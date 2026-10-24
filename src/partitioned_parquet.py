"""Resumable, disk-partitioned SUM aggregations with bounded input scans.

DuckDB hashes and projects Arrow streams; Python never builds a row dictionary
or a whole-dataframe aggregation. Equal output keys always share a partition.
After independent aggregation, a streaming concatenation is sufficient.
"""

import contextlib
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from collections import OrderedDict

from .source_info import write_json


ALGORITHM_VERSION = 2


def _quote(value):
    return str(value).replace("'", "''")


def _columns(names):
    return ", ".join('"' + name.replace('"', '""') + '"' for name in names)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_record(path, relative_to=None):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_checkpoint(path, root):
    """Reuse only atomically completed products whose sizes/mtimes still match."""
    try:
        value = json.loads(Path(path).read_text())
        for record in value.get("files", []):
            current = _file_record(Path(root) / record["path"], root)
            if current != record:
                return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def partition_options(args):
    options = {
        "partitions": int(getattr(args, "partitions", 128)),
        "batch_rows": int(getattr(args, "partition_batch_rows", 50000000)),
        "arrow_batch_rows": int(getattr(args, "arrow_batch_rows", 65536)),
        "max_split_depth": int(getattr(args, "max_split_depth", 4)),
        "threads": max(1, int(getattr(args, "threads", None)
                              or os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        "memory_limit": getattr(args, "memory_limit", None),
        "max_temp_directory_size": getattr(args, "max_temp_directory_size", "1TB"),
        "time_partitioning": getattr(args, "time_partitioning", "auto"),
        "target_partition_rows": int(getattr(args, "target_partition_rows", 250000000)),
        "partition_sample_rows": int(getattr(args, "partition_sample_rows", 1024)),
        "shuffle_buffer_mb": int(getattr(args, "shuffle_buffer_mb", 256)),
        "shuffle_row_group_rows": int(getattr(args, "shuffle_row_group_rows", 131072)),
        "shuffle_file_rows": int(getattr(args, "shuffle_file_rows", 2000000)),
        "shuffle_max_open_files": int(getattr(args, "shuffle_max_open_files", 512)),
        "aggregation_workers": int(getattr(args, "aggregation_workers", 1)),
        "partial_aggregate": bool(getattr(args, "partial_aggregate", False)),
    }
    if any(options[name] < 1 for name in
           ("partitions", "batch_rows", "arrow_batch_rows", "target_partition_rows",
            "partition_sample_rows", "shuffle_buffer_mb", "shuffle_row_group_rows",
            "shuffle_file_rows", "shuffle_max_open_files", "aggregation_workers")):
        raise ValueError("partition counts and batch row limits must be positive")
    if options["max_split_depth"] < 0:
        raise ValueError("max split depth must be nonnegative")
    if options["time_partitioning"] not in ("auto", "off"):
        raise ValueError("time partitioning must be auto or off")
    if options["aggregation_workers"] > options["threads"]:
        raise ValueError("--aggregation-workers cannot exceed --threads / allocated CPUs")
    return options


def add_partition_arguments(parser):
    parser.add_argument("--partitions", type=int, default=128,
                        help="baseline total bucket budget, or hash-only buckets (default: 128)")
    parser.add_argument("--partition-batch-rows", type=int, default=50000000,
                        help="maximum rows streamed per checkpointed shuffle batch")
    parser.add_argument("--arrow-batch-rows", type=int, default=65536,
                        help="maximum rows in a Python/Arrow input buffer")
    parser.add_argument("--max-split-depth", type=int, default=4,
                        help="recursively subdivide an OOM bucket up to this depth")
    parser.add_argument("--max-temp-directory-size", default="1TB",
                        help="DuckDB spill cap per connection; separate from shuffle storage")
    parser.add_argument("--work-directory",
                        help="resumable partitions; defaults to TEMP_DIRECTORY/partitioned")
    parser.add_argument("--time-partitioning", choices=("auto", "off"), default="auto",
                        help="separate integer decade/time grouping keys before hashing")
    parser.add_argument("--target-partition-rows", type=int, default=250000000,
                        help="estimated input rows per adaptive time/hash bucket")
    parser.add_argument("--partition-sample-rows", type=int, default=1024,
                        help="rows sampled per first/middle/last row-group anchor")
    parser.add_argument("--shuffle-buffer-mb", type=int, default=256,
                        help="shared Arrow bucket-buffer budget, not per bucket")
    parser.add_argument("--shuffle-row-group-rows", type=int, default=131072)
    parser.add_argument("--shuffle-file-rows", type=int, default=2000000)
    parser.add_argument("--shuffle-max-open-files", type=int, default=512)
    parser.add_argument("--aggregation-workers", type=int, default=1,
                        help="concurrent buckets; total memory and SQL threads are divided")
    parser.add_argument("--partial-aggregate", action="store_true",
                        help="optionally combine duplicate keys within bounded Arrow batches")


def _memory_bytes(value):
    """Parse DuckDB's decimal/binary memory units without relying on node RAM."""
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?I?B)?\s*",
                         str(value).upper())
    if not match:
        raise ValueError(f"Unsupported memory limit: {value!r}; use e.g. 192GB")
    unit = match.group(2) or "B"
    power = "BKMGTPE".index(unit[0])
    return int(float(match.group(1)) * (1024 if "I" in unit else 1000) ** power)


def _aggregation_options(options):
    """The supplied SQL memory/CPU allocation is a TOTAL, never per worker."""
    workers = options["aggregation_workers"]
    if workers > options["threads"]:
        raise ValueError("--aggregation-workers cannot exceed --threads / allocated CPUs")
    result = dict(options)
    result["threads"] = max(1, options["threads"] // workers)
    if options.get("memory_limit"):
        result["memory_limit"] = f"{_memory_bytes(options['memory_limit']) // workers}B"
    elif workers > 1:
        raise ValueError("Parallel aggregation requires an explicit total --memory-limit")
    # OOM subdivisions run sequentially inside their parent worker.
    result["aggregation_workers"] = 1
    result["shuffle_buffer_mb"] = max(1, options["shuffle_buffer_mb"] // workers)
    result["shuffle_max_open_files"] = max(1, options["shuffle_max_open_files"] // workers)
    return result


@contextlib.contextmanager
def connection_scope(options, spill_root, database=None, read_only=False):
    import duckdb

    spill_root = Path(spill_root)
    spill_root.mkdir(parents=True, exist_ok=True)
    # Every connection gets its own spill directory. Shuffle/checkpoint files
    # live outside it and survive query errors and Slurm timeouts.
    with tempfile.TemporaryDirectory(prefix="query-", dir=str(spill_root)) as spill:
        connection = duckdb.connect(str(database) if database else ":memory:",
                                    read_only=read_only)
        try:
            connection.execute(f"SET threads = {options['threads']}")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute(f"SET temp_directory = '{_quote(spill)}'")
            if options.get("memory_limit"):
                connection.execute(f"SET memory_limit = '{_quote(options['memory_limit'])}'")
            if options.get("max_temp_directory_size"):
                connection.execute(
                    f"SET max_temp_directory_size = '{_quote(options['max_temp_directory_size'])}'"
                )
            yield connection
        finally:
            connection.close()


def _plan_batches(sources, input_columns, max_rows):
    """Plan row-group slices once, without binding thousands of DuckDB files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = None
    batches, current, current_rows = [], [], 0
    for source in sources:
        with pq.ParquetFile(source) as parquet:
            selected = pa.schema([parquet.schema_arrow.field(name) for name in input_columns])
            if schema is None:
                schema = selected
            elif not schema.equals(selected, check_metadata=False):
                raise RuntimeError(f"Incompatible input schema in {source}")
            start = 0
            while start < parquet.num_row_groups:
                stop, rows = start, 0
                while stop < parquet.num_row_groups:
                    size = parquet.metadata.row_group(stop).num_rows
                    if stop > start and current_rows + rows + size > max_rows:
                        break
                    if current and current_rows + rows + size > max_rows:
                        break
                    rows += size
                    stop += 1
                    if current_rows + rows >= max_rows:
                        break
                if stop == start:
                    batches.append(current)
                    current, current_rows = [], 0
                    continue
                current.append({"path": str(source), "start": start,
                                "stop": stop, "rows": rows})
                current_rows += rows
                start = stop
                if current_rows >= max_rows:
                    batches.append(current)
                    current, current_rows = [], 0
    if current:
        batches.append(current)
    if schema is None:
        raise ValueError("No input sources supplied")
    return schema, batches


def _arrow_reader(schema, units, batch_rows, statistics):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    def read():
        for unit in units:
            with pq.ParquetFile(unit["path"]) as parquet:
                for batch in parquet.iter_batches(
                        batch_size=batch_rows, columns=schema.names,
                        row_groups=range(unit["start"], unit["stop"]),
                        use_threads=False):
                    statistics["input_rows"] += batch.num_rows
                    statistics["input_count_sum"] += int(
                        pc.sum(batch.column(batch.schema.get_field_index("count"))).as_py() or 0
                    )
                    yield batch
    return pa.RecordBatchReader.from_batches(schema, read())


def _plan_routing(connection, schema, batches, query, keys, output_schema, options,
                  partitions):
    """Freeze a stratified, bounded estimate; never scan all rows per decade.

    Estimates affect resource allocation only. Unseen times get one bucket and
    recursive splitting remains available if a sample underestimates a decade.
    The projection is applied to samples too, so aliases/filters are respected.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    column = next((name for name in ("decade", "time") if name in keys
                   and pa.types.is_integer(output_schema.field(name).type)), None)
    if options["time_partitioning"] == "off" or column is None:
        return {"type": "hash", "buckets": partitions}
    source_time = options.get("source_time_column")
    if source_time and (source_time not in schema.names
                        or not pa.types.is_integer(schema.field(source_time).type)):
        raise ValueError("time source hint must name an integer input column")
    estimates = {}
    for index, units in enumerate(batches):
        middle = units[len(units) // 2]
        anchors = dict.fromkeys([
            (units[0]["path"], units[0]["start"]),
            (middle["path"], (middle["start"] + middle["stop"] - 1) // 2),
            (units[-1]["path"], units[-1]["stop"] - 1),
        ])
        samples = []
        for path, row_group in anchors:
            with pq.ParquetFile(path) as parquet:
                sample = next(parquet.iter_batches(
                    batch_size=options["partition_sample_rows"],
                    columns=[source_time] if source_time else schema.names,
                    row_groups=[row_group], use_threads=False), None)
                if sample is not None:
                    samples.append(sample)
        if not samples:
            continue
        sample = pa.Table.from_batches(samples)
        if source_time:
            # Pipeline callers certify that output time is this input column,
            # unchanged. Planning reads ONLY it, avoiding repeated lemma joins.
            # Ignoring a projection filter merely overestimates resources.
            population = [(row["values"], row["counts"])
                          for row in pc.value_counts(sample[source_time]).to_pylist()]
        else:
            connection.register("partition_input", sample)
            try:
                population = connection.execute(
                    f"SELECT {_columns([column])}, COUNT(*) FROM ({query}) p "
                    f"GROUP BY {_columns([column])}"
                ).fetchall()
            finally:
                connection.unregister("partition_input")
        weight = sum(unit["rows"] for unit in units) / sample.num_rows
        for value, rows in population:
            estimates[value] = estimates.get(value, 0.0) + rows * weight
        if (index + 1) % 200 == 0:
            print(f"Planned time routing: sampled {index + 1}/{len(batches)} batches",
                  flush=True)
    total = sum(estimates.values())
    target = min(options["target_partition_rows"], total / partitions) if total else 1
    layout = [{"time": value, "estimated_rows": math.ceil(rows),
               "buckets": max(1, math.ceil(rows / max(1, target)))}
              for value, rows in sorted(estimates.items(),
                                        key=lambda item: (item[0] is None, item[0] or 0))]
    print(f"Planned {len(layout)} {column} partitions with "
          f"{sum(item['buckets'] for item in layout)} adaptive hash buckets", flush=True)
    return {"type": "time", "column": column, "partitions": layout,
            "sampling": "first/middle/last row-group anchors per checkpoint batch",
            "source_time_column": source_time,
            "unseen_time_buckets": 1}


class _BufferedPartitionWriter:
    """Compact files AND row groups under one shared Arrow buffer budget.

    A writer can remain open between flushes without retaining all bucket rows.
    If descriptor pressure closes a writer, rotation occurs at a buffered flush,
    not at every tiny input slice. The budget excludes the bounded input batch
    and transient concatenation/compression buffers, so it is not a process-RSS
    limit. No whole-corpus dictionaries/dataframes are built.
    """
    def __init__(self, root, schema, routing, options):
        self.root, self.schema, self.routing, self.options = Path(root), schema, routing, options
        self.buffers, self.buffer_rows, self.buffer_sizes = {}, {}, {}
        self.buffer_bytes = 0
        self.writers = OrderedDict()
        self.file_numbers, self.files = {}, []

    def append(self, key, table):
        if not table.num_rows:
            return
        self.buffers.setdefault(key, []).append(table)
        self.buffer_rows[key] = self.buffer_rows.get(key, 0) + table.num_rows
        size = table.get_total_buffer_size()
        self.buffer_sizes[key] = self.buffer_sizes.get(key, 0) + size
        self.buffer_bytes += size
        if self.buffer_rows[key] >= self.options["shuffle_row_group_rows"]:
            self._flush(key)
        while self.buffer_bytes > self.options["shuffle_buffer_mb"] * 1024**2:
            self._flush(max(self.buffer_sizes, key=self.buffer_sizes.get))

    def _close_writer(self, key):
        writer, path, unused_rows = self.writers.pop(key)
        writer.close()
        self.files.append(path)

    def _writer(self, key):
        import pyarrow.parquet as pq

        if key in self.writers:
            self.writers.move_to_end(key)
            return self.writers[key]
        if len(self.writers) >= self.options["shuffle_max_open_files"]:
            self._close_writer(next(iter(self.writers)))
        value, bucket = key
        directory = self.root
        if self.routing["type"] == "time":
            directory /= f"{self.routing['column']}={'null' if value is None else value}"
        directory /= f"__bucket={bucket}"
        directory.mkdir(parents=True, exist_ok=True)
        number = self.file_numbers.get(key, 0)
        self.file_numbers[key] = number + 1
        path = directory / f"fragment-{number:05d}.parquet"
        state = (pq.ParquetWriter(path, self.schema, compression="zstd",
                                 use_dictionary=True), path, 0)
        self.writers[key] = state
        return state

    def _flush(self, key):
        import pyarrow as pa

        table = pa.concat_tables(self.buffers.pop(key)).combine_chunks()
        self.buffer_bytes -= self.buffer_sizes.pop(key)
        self.buffer_rows.pop(key)
        offset = 0
        while offset < table.num_rows:
            writer, path, written = self._writer(key)
            rows = min(table.num_rows - offset, self.options["shuffle_file_rows"] - written)
            writer.write_table(table.slice(offset, rows),
                               row_group_size=self.options["shuffle_row_group_rows"])
            self.writers[key] = (writer, path, written + rows)
            offset += rows
            if written + rows >= self.options["shuffle_file_rows"]:
                self._close_writer(key)

    def finish(self):
        for key in list(self.buffers):
            self._flush(key)
        self.close()
        return self.files

    def close(self):
        for key in list(self.writers):
            self._close_writer(key)
        self.buffers.clear()
        self.buffer_sizes.clear()
        self.buffer_rows.clear()
        self.buffer_bytes = 0


def _combine_partial(table, keys):
    """Optional bounded map-side combine, with checked rather than wrapping SUM."""
    import pyarrow as pa

    schema = table.schema
    index = table.schema.get_field_index("count")
    wide = table.set_column(index, "count", table["count"].cast(pa.decimal128(38, 0)))
    combined = wide.group_by([*keys, "__bucket"], use_threads=False).aggregate([
        ("count", "sum")])
    combined = combined.rename_columns([
        "count" if name == "count_sum" else name for name in combined.column_names])
    return combined.select(schema.names).cast(schema, safe=True)


def _route_batch(writer, batch, routing):
    """Group bounded Arrow indices; take copies avoid retaining sliced parent buffers."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    table = pa.Table.from_batches([batch])
    if not table.num_rows:
        return
    buckets = table["__bucket"].to_numpy().astype(np.int64, copy=False)
    stride = int(buckets.max(initial=0)) + 1
    times = None
    if routing["type"] == "time":
        encoded = pc.dictionary_encode(table[routing["column"]]).combine_chunks()
        times = encoded.dictionary.to_pylist()
        codes = pc.fill_null(encoded.indices, -1).to_numpy().astype(np.int64, copy=False)
        route_ids = (codes + 1) * stride + buckets
    else:
        route_ids = buckets
    order = np.argsort(route_ids, kind="stable")
    ordered_ids = route_ids[order]
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(ordered_ids)) + 1,
                                 [len(order)]))
    output = table.select(writer.schema.names)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        route_id = int(ordered_ids[start])
        bucket = route_id % stride
        time_index = route_id // stride - 1
        value = times[time_index] if times is not None and time_index >= 0 else None
        # take() owns the selected arrays. slice() would retain every input batch
        # as long as even one small bucket buffer still referenced it.
        writer.append((value, bucket), output.take(pa.array(order[start:stop])))


def _shuffle(connection, sources, input_columns, query, keys, root, options,
             partitions, seed=0, required_identifiers=(), adaptive=False):
    """Stream bounded Arrow batches to independently checkpointed bucket files."""
    import pyarrow as pa

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    plan = None
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
    if plan is None:
        schema, batches = _plan_batches(sources, input_columns, options["batch_rows"])
        plan = {"batches": batches}
    else:
        # Input identity and versions are part of the enclosing workspace key.
        import pyarrow.parquet as pq
        with pq.ParquetFile(sources[0]) as parquet:
            schema = pa.schema([parquet.schema_arrow.field(name) for name in input_columns])
        batches = plan["batches"]

    connection.register("partition_input", pa.Table.from_batches([], schema=schema))
    try:
        output_schema = connection.execute(
            f"SELECT {_columns([*keys, 'count'])} FROM ({query}) p LIMIT 0"
        ).to_arrow_table().schema
    finally:
        connection.unregister("partition_input")

    if "routing" not in plan:
        plan["routing"] = (_plan_routing(connection, schema, batches, query, keys,
                                         output_schema, options, partitions)
                           if adaptive else {"type": "hash", "buckets": partitions})
        write_json(plan_path, plan)
    routing = plan["routing"]

    hash_args = _columns(keys)
    # Changing DuckDB version invalidates the enclosing workspace: its hash
    # implementation is deterministic within a version, not across versions.
    hashed = f"hash({seed}, {hash_args})"
    if routing["type"] == "time":
        time_column = _columns([routing["column"]])
        cases = []
        for item in routing["partitions"]:
            condition = (f"{time_column} IS NULL" if item["time"] is None
                         else f"{time_column} = {int(item['time'])}")
            cases.append(f"WHEN {condition} THEN {hashed} % {int(item['buckets'])}")
        bucket = "CASE " + " ".join(cases) + " ELSE 0::UBIGINT END" if cases else "0::UBIGINT"
    else:
        bucket = f"{hashed} % {partitions}"
    if required_identifiers:
        missing = " OR ".join(f'"{name}" IS NULL' for name in required_identifiers)
        bucket = (f"CASE WHEN {missing} THEN error('Strict lemmatization failed: "
                  f"missing lemma/POS join') ELSE {bucket} END")
    project = (f"SELECT {_columns([*keys, 'count'])}, {bucket} AS __bucket "
               f"FROM ({query}) projected")
    paths = {}
    statistics = {"input_rows": 0, "input_count_sum": 0, "projected_rows": 0,
                  "shuffled_rows": 0, "shuffle_files": 0}
    for index, units in enumerate(batches):
        batch_dir = root / f"batch-{index:06d}"
        marker = batch_dir / "complete.json"
        saved = _read_checkpoint(marker, batch_dir)
        if saved is None:
            temporary = root / f"batch-{index:06d}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            observed = {name: 0 for name in statistics}
            reader = _arrow_reader(schema, units, options["arrow_batch_rows"], observed)
            connection.register("partition_input", reader)
            writer = _BufferedPartitionWriter(temporary, output_schema, routing, options)
            projected = None
            try:
                projected = connection.execute(project).to_arrow_reader(options["arrow_batch_rows"])
                for batch in projected:
                    observed["projected_rows"] += batch.num_rows
                    if options["partial_aggregate"]:
                        combined = _combine_partial(pa.Table.from_batches([batch]), keys)
                        chunks = combined.to_batches(max_chunksize=options["arrow_batch_rows"])
                    else:
                        chunks = [batch]
                    for chunk in chunks:
                        observed["shuffled_rows"] += chunk.num_rows
                        _route_batch(writer, chunk, routing)
                files = writer.finish()
            finally:
                if projected is not None:
                    projected.close()
                writer.close()
                connection.unregister("partition_input")
                reader.close()
            observed["shuffle_files"] = len(files)
            saved = {**observed, "files": [_file_record(path, temporary) for path in files]}
            write_json(temporary / "complete.json", saved)
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
            os.replace(temporary, batch_dir)
            print(f"Shuffled {root.parent.name}: batch {index + 1}/{len(batches)} "
                  f"({observed['projected_rows']:,} projected rows, "
                  f"{observed['shuffled_rows']:,} written rows, "
                  f"{observed['shuffle_files']:,} files)", flush=True)
        for name in statistics:
            statistics[name] += saved[name]
        for record in saved["files"]:
            path = batch_dir / record["path"]
            relative = Path(record["path"])
            bucket_id = int(relative.parent.name.split("=", 1)[1])
            time_directory = relative.parent.parent.name if routing["type"] == "time" else None
            paths.setdefault((time_directory, bucket_id), []).append(path)
    return output_schema, paths, statistics


def _aggregate_bucket(sources, keys, node, options, spill_root, depth):
    import duckdb

    node = Path(node)
    node.mkdir(parents=True, exist_ok=True)
    saved = _read_checkpoint(node / "complete.json", node)
    if saved is not None:
        return [node / record["path"] for record in saved["files"]], saved["statistics"]
    result = node / "sum.parquet"
    temporary = node / "sum.parquet.tmp"
    if not (node / "split.json").exists():
        temporary.unlink(missing_ok=True)
        try:
            with connection_scope(options, spill_root) as connection:
                connection.execute(
                    f"COPY (SELECT {_columns(keys)}, SUM(count)::BIGINT AS count "
                    f"FROM read_parquet(?, hive_partitioning=false) "
                    f"GROUP BY {_columns(keys)}) TO '{_quote(temporary)}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)",
                    [[str(path) for path in sources]],
                )
                rows, mass = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(count), 0) FROM read_parquet(?)",
                    [str(temporary)],
                ).fetchone()
            os.replace(temporary, result)
            statistics = {"rows": int(rows), "count_sum": int(mass)}
            write_json(node / "complete.json", {
                "statistics": statistics, "files": [_file_record(result, node)]})
            print(f"Aggregated {node}: {rows:,} rows", flush=True)
            return [result], statistics
        except duckdb.OutOfMemoryException as error:
            temporary.unlink(missing_ok=True)
            if depth >= options["max_split_depth"]:
                raise RuntimeError(
                    f"Partition {node} still exceeds its memory/spill limit at depth {depth}; "
                    "increase --max-split-depth or the applicable limit and rerun. "
                    f"Completed partitions will be reused. DuckDB reported: {error}"
                ) from None
            print(f"Splitting OOM partition {node} into 16 smaller buckets", flush=True)
            write_json(node / "split.json", {"depth": depth + 1, "partitions": 16})

    query = f"SELECT {_columns([*keys, 'count'])} FROM partition_input"
    with connection_scope(options, spill_root) as connection:
        _, children, _ = _shuffle(connection, sources, [*keys, "count"], query,
                                  keys, node / "shuffle", options, 16, seed=depth + 1)
    files, statistics = [], {"rows": 0, "count_sum": 0}
    for (unused_time, bucket_id), paths in children.items():
        if not paths:
            continue
        child_files, child_stats = _aggregate_bucket(
            paths, keys, node / f"child-{bucket_id:04d}", options, spill_root, depth + 1
        )
        files.extend(child_files)
        for name in statistics:
            statistics[name] += child_stats[name]
    write_json(node / "complete.json", {
        "statistics": statistics, "files": [_file_record(path, node) for path in files]})
    return files, statistics


def _concatenate(files, destination, schema, batch_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    # OOM retries must not become late disk-full failures during publication.
    # This is a lower-bound estimate; footer overhead/recompression can vary.
    needed = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(destination.parent).free < needed:
        raise RuntimeError(
            f"Insufficient space to publish {destination}: approximately "
            f"{needed / 1024**3:.1f} GiB required for its atomic temporary file. "
            "Partition results are checkpointed and can be reused."
        )
    with pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True) as writer:
        for source in files:
            with pq.ParquetFile(source) as parquet:
                for batch in parquet.iter_batches(batch_size=batch_rows, use_threads=False):
                    table = pa.Table.from_batches([batch]).select(schema.names).cast(schema)
                    writer.write_table(table, row_group_size=batch_rows)
    os.replace(temporary, destination)


def _logical_specification(specification):
    """Only physical implementation changes may reuse a published table."""
    names = ("duckdb_version", "pyarrow_version", "sources", "identity", "keys", "query",
             "input_columns", "required_identifiers", "preserve_count_mass")
    return {name: specification.get(name) for name in names}


def _reuse_published(work_root, workspace, destination, specification, fingerprint):
    """Look only at direct checkpoint markers, never traverse source fragments."""
    wanted = _logical_specification(specification)
    for candidate in sorted(Path(work_root).glob(f"{Path(destination).stem}-*")):
        if candidate == workspace or not (candidate / "published.json").is_file():
            continue
        try:
            previous = json.loads((candidate / "specification.json").read_text())
        except (OSError, ValueError):
            continue
        if _logical_specification(previous) != wanted:
            continue
        if previous.get("algorithm_version") != ALGORITHM_VERSION:
            continue
        saved = _read_checkpoint(candidate / "published.json", Path(destination).parent)
        if not saved or saved.get("fingerprint") != _digest(previous):
            continue
        # A different destination with the same filename must not be adopted.
        expected = _file_record(destination, Path(destination).parent) if Path(destination).is_file() else None
        if saved.get("files") != [expected]:
            continue
        statistics = {**saved["statistics"], "reused_from": str(candidate),
                      "reused_algorithm_version": previous["algorithm_version"],
                      "reused_implementation_sha256": previous.get("implementation_sha256")}
        write_json(workspace / "specification.json", specification)
        write_json(workspace / "published.json", {
            "fingerprint": fingerprint, "statistics": statistics, "files": [expected]})
        print(f"Reused logically matching published {destination} from {candidate}", flush=True)
        return statistics
    return None


@contextlib.contextmanager
def _destination_lock(work_root, destination):
    """Protect one canonical path across different physical workspace versions."""
    import fcntl

    directory = Path(work_root) / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_digest(str(Path(destination).resolve()))[:24] + ".lock")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another finalizer is publishing {destination}") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _aggregate_partitions(partitions, keys, workspace, options, spill_root):
    worker_options = _aggregation_options(options)
    tasks = []
    for (time_directory, bucket_id), paths in partitions.items():
        node = workspace / time_directory if time_directory else workspace
        tasks.append((node / f"bucket-{bucket_id:04d}", paths))
    # Larger partitions start first. Publication order remains deterministic.
    tasks.sort(key=lambda item: (-len(item[1]), str(item[0])))
    print(f"Aggregating {len(tasks)} buckets with {options['aggregation_workers']} workers "
          f"({worker_options['threads']} SQL threads, "
          f"{worker_options.get('memory_limit') or 'automatic memory'} each)", flush=True)
    completed = {}

    def aggregate(task):
        node, paths = task
        return _aggregate_bucket(paths, keys, node, worker_options, spill_root, 0)

    workers = options["aggregation_workers"]
    if workers == 1:
        for task in tasks:
            completed[str(task[0])] = aggregate(task)
    else:
        # Submit only the concurrency limit, not thousands of queued tasks.
        # On failure, no new tasks start; in-flight buckets may finish/checkpoint.
        pending = {}
        iterator = iter(tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for task in iterator:
                pending[executor.submit(aggregate, task)] = task[0]
                if len(pending) == workers:
                    break
            try:
                while pending:
                    done, unused = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        node = pending.pop(future)
                        completed[str(node)] = future.result()
                    for unused in range(workers - len(pending)):
                        task = next(iterator, None)
                        if task is not None:
                            pending[executor.submit(aggregate, task)] = task[0]
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    files, statistics = [], {"rows": 0, "count_sum": 0}
    for node in sorted(completed):
        outputs, observed = completed[node]
        files.extend(outputs)
        for name in statistics:
            statistics[name] += observed[name]
    return files, statistics


def partitioned_sum(sources, destination, keys, query, input_columns, *,
                    options, work_root, spill_root, identity=None,
                    projection_database=None, required_identifiers=(),
                    preserve_count_mass=True, force=False, time_source_column=None,
                    projection_setup=None):
    """Project once, route on output keys, aggregate, then publish atomically."""
    import duckdb
    import pyarrow as pa

    options = {**options, "source_time_column": time_source_column}
    _aggregation_options(options)  # Validate total resource budgets before starting a shuffle.
    sources = sorted({Path(path).resolve() for path in sources})
    specification = {
        "algorithm_version": ALGORITHM_VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "duckdb_version": duckdb.__version__, "pyarrow_version": pa.__version__,
        "sources": [_file_record(path) for path in sources],
        "identity": identity, "keys": list(keys), "query": query,
        "input_columns": list(input_columns),
        "required_identifiers": list(required_identifiers),
        "preserve_count_mass": preserve_count_mass,
        "partitions": options["partitions"], "batch_rows": options["batch_rows"],
        "time_source_column": time_source_column,
        "layout_options": {name: options[name] for name in (
            "arrow_batch_rows", "time_partitioning", "target_partition_rows",
            "partition_sample_rows", "shuffle_buffer_mb", "shuffle_row_group_rows",
            "shuffle_file_rows", "shuffle_max_open_files", "partial_aggregate")},
    }
    fingerprint = _digest(specification)
    workspace = Path(work_root) / f"{Path(destination).stem}-{fingerprint[:24]}"
    if force:
        workspace = workspace.with_name(workspace.name + f"-force-{time.time_ns()}")
    workspace.mkdir(parents=True, exist_ok=True)
    # A cooperative lock prevents two finalizers from deleting/replacing each
    # other's scratch products. flock is released even if Slurm kills Python.
    import fcntl
    with _destination_lock(work_root, destination), (workspace / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using partition workspace {workspace}") from None
        published = _read_checkpoint(workspace / "published.json", Path(destination).parent)
        if published and published.get("fingerprint") == fingerprint:
            print(f"Reused {destination}", flush=True)
            return {**published["statistics"], "reused": True,
                    "checkpoint_directory": str(workspace)}
        if not force:
            reused = _reuse_published(work_root, workspace, destination, specification, fingerprint)
            if reused is not None:
                return {**reused, "reused": True, "checkpoint_directory": str(workspace)}
        write_json(workspace / "specification.json", specification)
        print(f"Partition workspace: {workspace}", flush=True)
        with connection_scope(options, spill_root, database=projection_database,
                              read_only=bool(projection_database)) as connection:
            # Connection-local functions must be installed before binding the
            # projection/planning queries. Their scientific identity belongs
            # in the caller's explicit identity, not in a callable's repr.
            if projection_setup is not None:
                projection_setup(connection)
            schema, partitions, shuffled = _shuffle(
                connection, sources, input_columns, query, keys,
                workspace / "shuffle", options, options["partitions"],
                required_identifiers=required_identifiers,
                adaptive=True,
            )
        files, statistics = _aggregate_partitions(
            partitions, keys, workspace, options, spill_root)
        if preserve_count_mass and statistics["count_sum"] != shuffled["input_count_sum"]:
            raise RuntimeError("Count conservation failed; canonical output was not replaced")
        _concatenate(files, destination, schema, options["arrow_batch_rows"])
        statistics.update({"input_rows": shuffled["input_rows"],
                           "projected_rows": shuffled["projected_rows"],
                           "shuffled_rows": shuffled["shuffled_rows"],
                           "shuffle_files": shuffled["shuffle_files"],
                           "routing": json.loads((workspace / "shuffle" / "plan.json").read_text())["routing"],
                           "aggregation_workers": options["aggregation_workers"]})
        write_json(workspace / "published.json", {
            "fingerprint": fingerprint, "statistics": statistics,
            "files": [_file_record(destination, Path(destination).parent)],
        })
        return {**statistics, "checkpoint_directory": str(workspace)}
