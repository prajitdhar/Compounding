"""Small end-to-end regressions for compact, time-aware reduction.

All data and checkpoints are temporary; these tests never inspect production
Google extracts or the shared scratch workspace.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


GOOGLE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(GOOGLE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GOOGLE_DIRECTORY))


class CompactPartitionedAggregationTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("duckdb and pyarrow are required")
        from src import partitioned_parquet

        self.duckdb, self.pa, self.pq = duckdb, pa, pq
        self.partitioned = partitioned_parquet
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root / "sum.parquet"
        self.sources = []

    def options(self, **overrides):
        settings = dict(
            partitions=4, partition_batch_rows=4096, arrow_batch_rows=7,
            max_split_depth=3, threads=1, memory_limit="256MB",
            time_partitioning="auto", target_partition_rows=256,
            partition_sample_rows=1024, shuffle_buffer_mb=16,
            shuffle_row_group_rows=128, shuffle_file_rows=4096,
            shuffle_max_open_files=32, aggregation_workers=1,
            partial_aggregate=False,
        )
        settings.update(overrides)
        return self.partitioned.partition_options(SimpleNamespace(**settings))

    def write_source(self, rows, row_group_rows=128):
        path = self.root / f"input-{len(self.sources):02d}.parquet"
        self.pq.write_table(self.pa.Table.from_pylist(rows), path,
                            row_group_size=row_group_rows)
        self.sources.append(path)
        return path

    def run_sum(self, options=None, keys=("word", "time")):
        columns = [*keys, "count"]
        return self.partitioned.partitioned_sum(
            self.sources, self.destination, list(keys),
            "SELECT " + ", ".join(columns) + " FROM partition_input", columns,
            options=options or self.options(), work_root=self.root / "checkpoints",
            spill_root=self.root / "spill",
        )

    def assert_reference(self, keys=("word", "time")):
        grouping = ", ".join(keys)
        with self.duckdb.connect() as connection:
            expected = connection.execute(
                f"SELECT {grouping}, SUM(count)::BIGINT FROM read_parquet(?) "
                f"GROUP BY {grouping} ORDER BY {grouping}",
                [[str(path) for path in self.sources]],
            ).fetchall()
            actual = connection.execute(
                f"SELECT {grouping}, count FROM read_parquet(?) ORDER BY {grouping}",
                [str(self.destination)],
            ).fetchall()
        self.assertEqual(actual, expected)
        source_schema = self.pq.read_schema(self.sources[0])
        expected_schema = self.pa.schema([
            *(source_schema.field(key) for key in keys),
            self.pa.field("count", self.pa.int64()),
        ])
        self.assertTrue(self.pq.read_schema(self.destination).equals(
            expected_schema, check_metadata=False))

    @staticmethod
    def read_plan(result):
        return json.loads((Path(result["checkpoint_directory"]) /
                           "shuffle" / "plan.json").read_text())

    def test_compact_shuffle_combines_many_arrow_chunks_into_large_files(self):
        self.write_source([
            {"word": f"word-{index}", "time": 1900, "count": index % 5 + 1}
            for index in range(2048)
        ], row_group_rows=128)
        result = self.run_sum(self.options(time_partitioning="off"))
        shuffle = Path(result["checkpoint_directory"]) / "shuffle"
        files = list(shuffle.rglob("*.parquet"))
        # There are 293 Arrow input chunks, but at most one output file per
        # physical bucket is necessary for this small checkpoint batch.
        self.assertLessEqual(len(files), 4)
        self.assertGreater(len(files), 0)
        groups = []
        for path in files:
            with self.pq.ParquetFile(path) as parquet:
                groups.extend(parquet.metadata.row_group(index).num_rows
                              for index in range(parquet.num_row_groups))
        self.assertGreaterEqual(max(groups), 128)
        self.assert_reference()

    def test_parallel_partial_aggregation_preserves_counts_and_schema(self):
        for source in range(3):
            self.write_source([
                {"word": f"w{index % 11}", "time": 1820 + (index % 3) * 10,
                 "count": source + index % 7 + 1}
                for index in range(1024)
            ], row_group_rows=64)
        options = self.options(
            threads=2, aggregation_workers=2, partial_aggregate=True,
            arrow_batch_rows=128, partition_batch_rows=512,
            target_partition_rows=256,
        )
        worker_budgets = []
        aggregate_bucket = self.partitioned._aggregate_bucket

        def record_worker_budget(*args, **kwargs):
            worker_budgets.append(args[3])
            return aggregate_bucket(*args, **kwargs)

        with mock.patch.object(self.partitioned, "_aggregate_bucket",
                               record_worker_budget):
            observed = self.run_sum(options)
        self.assertEqual(observed["input_rows"], 3072)
        self.assertEqual(observed["count_sum"], sum(
            source + index % 7 + 1
            for source in range(3) for index in range(1024)))
        self.assert_reference()
        plan = self.read_plan(observed)
        self.assertEqual(plan["routing"]["type"], "time")
        self.assertEqual(plan["routing"]["column"], "time")
        self.assertTrue(worker_budgets)
        self.assertTrue(all(budget["threads"] == 1 and
                            budget["memory_limit"] == "128000000B"
                            for budget in worker_budgets))
        markers = (Path(observed["checkpoint_directory"]) / "shuffle").glob(
            "batch-*/complete.json")
        written = sum(json.loads(marker.read_text())["shuffled_rows"] for marker in markers)
        self.assertLess(written, observed["projected_rows"])

    def test_time_routing_plan_is_frozen_and_handles_an_unsampled_decade(self):
        self.write_source([
            {"word": f"w{index % 13}", "time": 1820, "count": index % 3 + 1}
            for index in range(1024)
        ])
        self.write_source([
            {"word": f"w{index % 17}", "time": 1930, "count": index % 5 + 1}
            for index in range(512)
        ])
        # A sample is an estimate, not an inventory of every decade. Freeze a
        # valid plan that missed 1930 and exercise the fallback route end to end.
        sampled_routing = {
            "type": "time", "column": "time",
            "partitions": [{"time": 1820, "buckets": 2,
                            "estimated_rows": 1024}],
        }
        with mock.patch.object(self.partitioned, "_plan_routing",
                               return_value=sampled_routing):
            observed = self.run_sum(self.options(
                partition_sample_rows=64, target_partition_rows=128,
            ))
        plan = self.read_plan(observed)
        routing = plan["routing"]
        self.assertEqual(routing["type"], "time")
        self.assertEqual(routing["column"], "time")
        times = {partition["time"] for partition in routing["partitions"]}
        self.assertIn(1820, times)
        self.assertNotIn(1930, times)
        self.assertTrue(all(partition["buckets"] >= 1 and
                            partition["estimated_rows"] >= 0
                            for partition in routing["partitions"]))
        self.assert_reference()
        plan_path = Path(observed["checkpoint_directory"]) / "shuffle" / "plan.json"
        original_bytes = plan_path.read_bytes()
        with mock.patch.object(self.partitioned, "_arrow_reader",
                               side_effect=AssertionError("completed shuffle was reread")):
            self.assertTrue(self.run_sum(self.options(
                partition_sample_rows=64, target_partition_rows=128))["reused"])
        self.assertEqual(plan_path.read_bytes(), original_bytes)

    def test_decade_column_gets_the_same_independent_time_routing(self):
        self.write_source([
            {"word": f"w{index % 11}", "decade": 1800 + (index % 4) * 10,
             "count": index % 3 + 1}
            for index in range(512)
        ])
        observed = self.run_sum(keys=("word", "decade"))
        self.assertEqual(self.read_plan(observed)["routing"]["column"], "decade")
        self.assert_reference(keys=("word", "decade"))

    def test_aggregation_without_time_key_uses_hash_routing(self):
        self.write_source([
            {"word": f"w{index % 11}", "time": 1900 + (index % 2) * 10,
             "count": index % 3 + 1}
            for index in range(256)
        ])
        observed = self.run_sum(keys=("word",))
        self.assertEqual(self.read_plan(observed)["routing"]["type"], "hash")
        self.assert_reference(keys=("word",))

    def test_empty_typed_route_batch_does_not_touch_the_writer(self):
        schema = self.pa.schema([
            ("word", self.pa.string()), ("time", self.pa.int64()),
            ("count", self.pa.int64()), ("__bucket", self.pa.uint64()),
        ])
        batch = self.pa.RecordBatch.from_arrays(
            [self.pa.array([], type=field.type) for field in schema], schema=schema)
        writer = mock.Mock()
        self.partitioned._route_batch(writer, batch, {"type": "time", "column": "time"})
        writer.append.assert_not_called()

    def test_null_time_values_are_not_dropped_or_merged_with_other_decades(self):
        self.write_source([
            {"word": f"w{index % 7}",
             "time": None if index % 3 == 0 else 1900 + (index % 2) * 10,
             "count": index % 5 + 1}
            for index in range(256)
        ])
        observed = self.run_sum()
        self.assertEqual(observed["input_rows"], 256)
        self.assert_reference()
        self.assertTrue(any(partition["time"] is None for partition in
                            self.read_plan(observed)["routing"]["partitions"]))

    def test_partial_sum_overflow_does_not_replace_an_existing_destination(self):
        self.write_source([
            {"word": "same", "time": 1900, "count": 2**63 - 1},
            {"word": "same", "time": 1900, "count": 1},
        ])
        self.destination.write_bytes(b"existing-canonical-output")
        with mock.patch.object(self.partitioned, "_concatenate") as publish:
            with self.assertRaises((self.pa.ArrowInvalid, OverflowError)):
                self.run_sum(self.options(partial_aggregate=True))
        publish.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b"existing-canonical-output")

    def test_source_time_hint_plans_without_executing_the_projection_lookup(self):
        self.write_source([
            {"word": f"w{index % 11}", "decade": 1820 + (index % 3) * 10,
             "count": index % 5 + 1}
            for index in range(512)
        ])
        options = self.options()
        schema, batches = self.partitioned._plan_batches(
            self.sources, ["word", "decade", "count"], options["batch_rows"])
        output_schema = self.pa.schema([
            ("word", self.pa.string()), ("time", self.pa.int64()),
            ("count", self.pa.int64()),
        ])
        connection = mock.Mock()
        connection.execute.side_effect = AssertionError("planner executed the lemma join")
        connection.register.side_effect = AssertionError("planner registered projection input")
        sampled_columns = []
        real_parquet_file = self.pq.ParquetFile

        class TrackColumns:
            def __init__(tracker, *args, **kwargs):
                tracker.parquet = real_parquet_file(*args, **kwargs)

            def __enter__(tracker):
                return tracker

            def __exit__(tracker, *args):
                tracker.parquet.close()

            def iter_batches(tracker, **kwargs):
                sampled_columns.append(kwargs["columns"])
                return tracker.parquet.iter_batches(**kwargs)

        with mock.patch.object(self.pq, "ParquetFile", TrackColumns):
            routing = self.partitioned._plan_routing(
                connection, schema, batches,
                "SELECT missing_lemma.word, decade AS time, count "
                "FROM partition_input JOIN unavailable_lemma_lookup missing_lemma USING(word)",
                ["word", "time"], output_schema,
                {**options, "source_time_column": "decade"}, options["partitions"],
            )
        self.assertTrue(sampled_columns)
        self.assertTrue(all(columns == ["decade"] for columns in sampled_columns))
        self.assertEqual(routing["column"], "time")
        self.assertEqual(routing["source_time_column"], "decade")
        connection.execute.assert_not_called()
        connection.register.assert_not_called()

    def test_source_decade_hint_supports_output_time_alias_end_to_end(self):
        self.write_source([
            {"word": f"w{index % 7}", "decade": 1820 + (index % 3) * 10,
             "count": index % 5 + 1}
            for index in range(256)
        ])
        observed = self.partitioned.partitioned_sum(
            self.sources, self.destination, ["word", "time"],
            "SELECT word, decade AS time, count FROM partition_input",
            ["word", "decade", "count"], options=self.options(),
            work_root=self.root / "checkpoints", spill_root=self.root / "spill",
            time_source_column="decade",
        )
        routing = self.read_plan(observed)["routing"]
        self.assertEqual(routing["column"], "time")
        self.assertEqual(routing["source_time_column"], "decade")
        with self.duckdb.connect() as connection:
            expected = connection.execute(
                "SELECT word,decade,SUM(count)::BIGINT FROM read_parquet(?) "
                "GROUP BY word,decade ORDER BY word,decade",
                [[str(path) for path in self.sources]],
            ).fetchall()
            actual = connection.execute(
                "SELECT word,time,count FROM read_parquet(?) ORDER BY word,time",
                [str(self.destination)],
            ).fetchall()
        self.assertEqual(actual, expected)
        self.assertEqual(self.pq.read_schema(self.destination).names, ["word", "time", "count"])

    def test_destination_lock_rejects_another_physical_workspace(self):
        self.write_source([{"word": "same", "time": 1900, "count": 1}])
        with self.partitioned._destination_lock(self.root / "checkpoints", self.destination):
            with self.assertRaisesRegex(RuntimeError, "Another finalizer is publishing"):
                self.run_sum(self.options(partitions=8))
        self.assertFalse(self.destination.exists())

    def test_shuffle_interruption_reuses_completed_batches_not_partial_files(self):
        self.write_source([
            {"word": f"w{index % 19}", "time": 1900 + (index % 2) * 10,
             "count": index % 5 + 1}
            for index in range(512)
        ], row_group_rows=64)
        options = self.options(partition_batch_rows=128, time_partitioning="off")
        real_reader = self.partitioned._arrow_reader
        calls = []

        def fail_second_batch(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise RuntimeError("simulated shuffle interruption")
            return real_reader(*args, **kwargs)

        with mock.patch.object(self.partitioned, "_arrow_reader", fail_second_batch):
            with self.assertRaisesRegex(RuntimeError, "simulated shuffle interruption"):
                self.run_sum(options)
        complete = list((self.root / "checkpoints").glob(
            "*/shuffle/batch-*/complete.json"))
        self.assertEqual(len(complete), 1)
        checkpoint_bytes = complete[0].read_bytes()
        checkpoint_mtime = complete[0].stat().st_mtime_ns
        with mock.patch.object(self.partitioned, "_arrow_reader", wraps=real_reader) as reader:
            self.run_sum(options)
        self.assertEqual(reader.call_count, 3)
        self.assertEqual(complete[0].read_bytes(), checkpoint_bytes)
        self.assertEqual(complete[0].stat().st_mtime_ns, checkpoint_mtime)
        self.assert_reference()

    def test_published_result_is_reused_after_physical_options_change(self):
        self.write_source([
            {"word": f"w{index % 7}", "time": 1900, "count": index + 1}
            for index in range(64)
        ])
        self.run_sum(self.options(time_partitioning="off"))
        original_mtime = self.destination.stat().st_mtime_ns
        changed = self.options(
            partitions=8, shuffle_file_rows=8192, time_partitioning="auto",
        )
        with mock.patch.object(self.partitioned, "_arrow_reader",
                               side_effect=AssertionError("compatible output was recomputed")):
            observed = self.run_sum(changed)
        self.assertTrue(observed["reused"])
        self.assertEqual(self.destination.stat().st_mtime_ns, original_mtime)
        self.assert_reference()

    def test_unsupported_checkpoint_version_is_not_adopted(self):
        self.write_source([
            {"word": f"w{index % 7}", "time": 1900, "count": index + 1}
            for index in range(64)
        ])
        with self.duckdb.connect() as connection:
            destination = str(self.destination).replace("'", "''")
            connection.execute(
                "COPY (SELECT word,time,SUM(count)::BIGINT AS count "
                f"FROM read_parquet(?) GROUP BY word,time) TO '{destination}' (FORMAT PARQUET)",
                [[str(path) for path in self.sources]],
            )
        specification = {
            "algorithm_version": 1,
            "implementation_sha256": "unsupported-implementation",
            "duckdb_version": self.duckdb.__version__,
            "pyarrow_version": self.pa.__version__,
            "sources": [self.partitioned._file_record(path) for path in self.sources],
            "identity": None, "keys": ["word", "time"],
            "query": "SELECT word, time, count FROM partition_input",
            "input_columns": ["word", "time", "count"],
            "required_identifiers": [], "preserve_count_mass": True,
            "partitions": 128, "batch_rows": 50000000,
        }
        fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        workspace = self.root / "checkpoints" / f"sum-{fingerprint[:24]}"
        workspace.mkdir(parents=True)
        (workspace / "specification.json").write_text(json.dumps(specification))
        statistics = {"rows": 7, "count_sum": 2080, "input_rows": 64,
                      "projected_rows": 64}
        (workspace / "published.json").write_text(json.dumps({
            "fingerprint": fingerprint, "statistics": statistics,
            "files": [self.partitioned._file_record(self.destination, self.root)],
        }))
        observed = self.run_sum()
        self.assertFalse(observed.get("reused", False))
        self.assertNotIn("reused_from", observed)
        self.assert_reference()


class AggregationWorkerBudgetTests(unittest.TestCase):
    def setUp(self):
        from src import partitioned_parquet
        self.partitioned = partitioned_parquet

    def test_total_memory_and_threads_are_divided_between_workers(self):
        options = self.partitioned.partition_options(SimpleNamespace(
            threads=5, aggregation_workers=2, memory_limit="1GB",
        ))
        per_worker = self.partitioned._aggregation_options(options)
        self.assertIsNot(per_worker, options)
        self.assertEqual(options["threads"], 5)
        self.assertEqual(options["memory_limit"], "1GB")
        self.assertEqual(per_worker["threads"], 2)
        self.assertEqual(per_worker["memory_limit"], "500000000B")

    def test_explicit_worker_count_cannot_overcommit_allocated_threads(self):
        with self.assertRaisesRegex(ValueError, "(?i)(worker|thread)"):
            options = self.partitioned.partition_options(SimpleNamespace(
                threads=1, aggregation_workers=2, memory_limit="1GB",
            ))
            self.partitioned._aggregation_options(options)


if __name__ == "__main__":
    unittest.main()
