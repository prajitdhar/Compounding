import importlib.util
import contextlib
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "google_v3_pipeline.py"
SPEC = importlib.util.spec_from_file_location("google_v3_pipeline", str(MODULE_PATH))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class RecordParsingTests(unittest.TestCase):
    def test_parse_packed_v3_record(self):
        ngram, measurements = pipeline.parse_v3_record(
            "ice_NOUN cream_NOUN\t1901,12,7\t1909,4,3\n"
        )
        self.assertEqual(ngram, "ice_NOUN cream_NOUN")
        self.assertEqual(measurements[0], pipeline.Measurement(1901, 12, 7))
        self.assertEqual(measurements[1], pipeline.Measurement(1909, 4, 3))

    def test_decade_aggregation_keeps_only_google_match_count(self):
        measurements = [
            pipeline.Measurement(1901, 12, 7),
            pipeline.Measurement(1909, 4, 3),
            pipeline.Measurement(1910, 2, 2),
            pipeline.Measurement(1799, 100, 100),
        ]
        result = pipeline.aggregate_measurements(measurements, 1800, 2019, 10)
        self.assertEqual(result[1900], 16)
        self.assertEqual(result[1910], 2)
        self.assertNotIn(1790, result)


class TokenTests(unittest.TestCase):
    def test_surface_with_underscores_uses_final_suffix(self):
        token = pipeline.parse_tagged_token("New_York_NOUN")
        self.assertEqual(token.surface, "New_York")
        self.assertEqual(token.pos, "NOUN")
        self.assertTrue(pipeline.is_lexical(token))

    def test_pos_projection_is_not_lexical(self):
        token = pipeline.parse_tagged_token("_NOUN_")
        self.assertTrue(token.wildcard)
        self.assertFalse(pipeline.is_lexical(token))

    def test_boundaries_are_annotated_but_not_lexical(self):
        token = pipeline.parse_tagged_token("_START_")
        self.assertTrue(token.boundary)
        self.assertFalse(pipeline.is_lexical(token))


class EventExtractionTests(unittest.TestCase):
    @staticmethod
    def tokens(value):
        return [pipeline.parse_tagged_token(token) for token in value.split(" ")]

    def test_candidate_inventory_accepts_an_and_nn_only(self):
        self.assertTrue(pipeline.is_eligible_pair(*self.tokens("electric_ADJ car_NOUN")))
        self.assertTrue(pipeline.is_eligible_pair(*self.tokens("ice_NOUN cream_NOUN")))
        self.assertFalse(pipeline.is_eligible_pair(*self.tokens("drive_VERB car_NOUN")))
        self.assertFalse(pipeline.is_eligible_pair(*self.tokens("ice_NOUN _NOUN_")))

    def test_distance_one_uses_trigram_edges(self):
        events = pipeline.context_events(
            self.tokens("cold_ADJ water_NOUN freezes_VERB")
        )
        event = next(row for row in events if row["modifier_norm"] == "cold")
        self.assertEqual(event["head_norm"], "water")
        self.assertEqual(event["context_norm"], "freezes")
        self.assertEqual(event["side"], "R")
        self.assertEqual(event["distance"], 1)

    def test_distance_two_extracts_only_farthest_context(self):
        events = pipeline.context_events(
            self.tokens("quickly_ADV very_ADV cold_ADJ water_NOUN")
        )
        event = next(row for row in events if row["modifier_norm"] == "cold")
        self.assertEqual(event["context_norm"], "quickly")
        self.assertEqual(event["side"], "L")
        self.assertEqual(event["distance"], 2)

    def test_one_ngram_can_supply_two_different_pair_targets(self):
        events = pipeline.context_events(
            self.tokens("ice_NOUN cream_NOUN shop_NOUN owner_NOUN")
        )
        self.assertEqual(len(events), 2)
        observed = {
            (row["modifier_norm"], row["head_norm"], row["side"])
            for row in events
        }
        self.assertEqual(
            observed,
            {("ice", "cream", "R"), ("shop", "owner", "L")},
        )

    def test_bigram_context_targets_include_all_lexical_pos(self):
        events = pipeline.context_events(
            self.tokens("the_DET car_NOUN moves_VERB")
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["modifier_pos"], "DET")
        self.assertEqual(events[0]["head_pos"], "NOUN")

    def test_context_projection_is_rejected(self):
        events = pipeline.context_events(
            self.tokens("cold_ADJ water_NOUN _VERB_")
        )
        self.assertEqual(events, [])

    def test_general_word_context_uses_edges_and_distance(self):
        events = pipeline.word_context_events(
            self.tokens("water_NOUN can_VERB freeze_VERB")
        )
        self.assertEqual(
            {(row["word_norm"], row["context_norm"], row["side"], row["distance"])
             for row in events},
            {("water", "freeze", "R", 2), ("freeze", "water", "L", 2)},
        )

    def test_all_lexical_bigrams_are_retained_but_punctuation_is_not(self):
        self.assertEqual(
            len(pipeline.bigram_events(self.tokens("the_DET car_NOUN"))), 1
        )
        self.assertEqual(
            pipeline.bigram_events(self.tokens("car_NOUN ._.")), []
        )

    def test_single_bigram_stream_retains_cased_surfaces_and_pos(self):
        rows = pipeline.bigram_events(
            self.tokens("Running_VERB Dogs_NOUN")
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["modifier_surface"], "Running")
        self.assertEqual(rows[0]["head_surface"], "Dogs")
        self.assertEqual(rows[0]["modifier_pos"], "VERB")

    def test_bigram_supplies_both_general_constituent_directions(self):
        events = pipeline.word_context_events(
            self.tokens("cold_ADJ water_NOUN")
        )
        self.assertEqual(
            {
                (event["word_norm"], event["context_norm"], event["side"])
                for event in events
            },
            {("cold", "water", "R"), ("water", "cold", "L")},
        )

    def test_dependency_arrow_is_normalized_to_modifier_head_order(self):
        forward = pipeline.dependency_events("cream_NOUN=>ice_NOUN")
        reverse = pipeline.dependency_events("ice_NOUN<=cream_NOUN")
        self.assertEqual(len(forward), 1)
        self.assertEqual(len(reverse), 1)
        self.assertEqual(forward[0]["modifier_surface"], "ice")
        self.assertEqual(forward[0]["head_surface"], "cream")
        self.assertEqual(reverse[0]["modifier_surface"], "ice")
        self.assertEqual(reverse[0]["head_surface"], "cream")

    def test_ner_window_preserves_case_and_aligns_after_boundary_removal(self):
        events = pipeline.ner_snippet_events(
            self.tokens("_START_ visit_VERB New_ADJ York_NOUN today_NOUN")
        )
        self.assertEqual(len(events), 1)
        row = events[0]
        self.assertEqual(json.loads(row["tokens_json"]), ["visit", "New", "York", "today"])
        self.assertEqual(row["modifier_index"], 1)
        self.assertEqual(row["head_index"], 2)
        self.assertTrue(row["sentence_start"])
        self.assertFalse(row["sentence_end"])


class InputListTests(unittest.TestCase):
    def test_manifest_parser_keeps_only_gzip_links_and_uses_https(self):
        parser = pipeline.ExportIndexParser()
        parser.feed(
            '<a href="http://example.test/2-00000-of-00002.gz">a</a>'
            '<a href="http://example.test/index.html">index</a>'
        )
        self.assertEqual(
            parser.urls,
            ["https://example.test/2-00000-of-00002.gz"],
        )

    def test_target_pairs_accept_rating_table_columns(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        try:
            handle.write("score\thead\tmodifier\n1\tYork\tNew\n")
            handle.close()
            self.assertEqual(
                pipeline.load_target_pairs(handle.name),
                [("new", "york")],
            )
        finally:
            os.unlink(handle.name)

    def test_context_vocabulary_is_pos_specific_when_pos_is_present(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        try:
            handle.write(
                "norm\tpos\tlemma\tsurface_match_count\tlemma_match_count\n"
                "saw\tNOUN\tsaw\t10\t10\n"
            )
            handle.close()
            vocabulary = pipeline.load_context_vocabulary(handle.name)
            self.assertTrue(pipeline.vocabulary_contains(vocabulary, "saw", "NOUN"))
            self.assertFalse(pipeline.vocabulary_contains(vocabulary, "saw", "VERB"))
        finally:
            os.unlink(handle.name)

    def test_norm_only_context_vocabulary_is_accepted(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        try:
            handle.write("norm\tmatch_count\nfreeze\t10\n")
            handle.close()
            vocabulary = pipeline.load_context_vocabulary(handle.name)
            self.assertTrue(pipeline.vocabulary_contains(vocabulary, "freeze", "VERB"))
        finally:
            os.unlink(handle.name)

    def test_vocabulary_ranking_gives_decades_equal_weight(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reduced = root / "reduced"
            reduced.mkdir()
            unigrams = pa.Table.from_pylist([
                {"surface": "a", "pos": "NOUN", "decade": 1900, "count": 90},
                {"surface": "b", "pos": "NOUN", "decade": 1900, "count": 10},
                {"surface": "b", "pos": "NOUN", "decade": 1910, "count": 100},
            ], schema=pipeline.arrow_schema("unigrams", pa))
            lexicon = pa.Table.from_pylist([
                {"surface": "a", "norm": "a", "pos": "NOUN", "lemma": "a", "count": 90},
                {"surface": "b", "norm": "b", "pos": "NOUN", "lemma": "b", "count": 110},
            ], schema=pipeline.arrow_schema("lexicon", pa))
            pq.write_table(unigrams, reduced / "unigrams.parquet")
            pq.write_table(lexicon, reduced / "lexicon.parquet")
            destination = root / "contexts.tsv"
            pipeline.make_vocabulary(SimpleNamespace(
                output=str(root), vocab_output=str(destination), coverage_output=None,
                lexicon=None, top_k=1, context_pos="NOUN", threads=1,
                memory_limit=None, temp_directory=str(root / "tmp"),
            ))
            rows = destination.read_text().splitlines()
            self.assertEqual(rows[1].split("\t")[:3], ["b", "NOUN", "b"])
            self.assertTrue(Path(str(destination) + ".json").is_file())


class ReductionValidationTests(unittest.TestCase):
    def test_reduction_pools_side_distance_and_accepts_empty_completed_tasks(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for shard, side, distance, count in (
                    ("one", "L", 1, 2), ("two", "R", 3, 3),
                    ("empty", None, None, None)):
                source = str(root / f"{shard}.gz")
                entries.append({"order": 3, "shard": shard, "source": source})
                marker = pipeline.completion_marker(root, 3, shard)
                marker.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "schema_version": 2, "corpus_version": pipeline.CORPUS_VERSION,
                    "source": source, "order": 3, "shard": shard,
                    "year_min": 1800, "year_max": 2019, "bin_size": 10,
                    "context_pos": sorted(pipeline.DEFAULT_CONTEXT_POS),
                    "context_vocabulary_sha256": "test_data", "ner_pair_offset": 2,
                    "word_target_pos": "all_lexical",
                    "bigram_target_pos": "all_lexical",
                    "representation_version": 6,
                    "context_surface_storage": "normalized_context_keys",
                    "count_measure": "match_count",
                    "contexts_rows": 0 if count is None else 1,
                    "word_contexts_rows": 0,
                }
                marker.write_text(json.dumps(metadata))
                if count is not None:
                    task = pipeline.task_data_dir(root, 3, shard)
                    task.mkdir(parents=True)
                    row = {
                        "modifier_norm": "cold", "modifier_pos": "ADJ",
                        "head_norm": "water", "head_pos": "NOUN",
                        "context_norm": "freeze", "context_pos": "VERB",
                        "side": side, "distance": distance,
                        "decade": 1900, "count": count,
                    }
                    pq.write_table(pa.Table.from_pylist(
                        [row], schema=pipeline.arrow_schema("contexts", pa)
                    ), task / "contexts.parquet")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "corpus": pipeline.DEFAULT_CORPUS,
                "corpus_version": pipeline.CORPUS_VERSION,
                "entries": entries,
            }))
            pipeline.reduce_tables(SimpleNamespace(
                output=str(root), manifest=[str(manifest)], allow_incomplete=False,
                table=["contexts"], force=False, temp_directory=str(root / "tmp"),
                threads=1, memory_limit=None, database=None,
            ))
            rows = pq.read_table(root / "reduced" / "contexts.parquet").to_pylist()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["count"], 5)
            self.assertNotIn("side", rows[0])
            self.assertNotIn("distance", rows[0])

    def test_reduction_rejects_incompatible_time_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for order, bin_size in ((2, 10), (3, 20)):
                marker = pipeline.completion_marker(root, order, str(order))
                marker.parent.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "schema_version": 1,
                    "corpus_version": pipeline.CORPUS_VERSION,
                    "year_min": 1800,
                    "year_max": 2019,
                    "bin_size": bin_size,
                    "order": order,
                    "context_pos": sorted(pipeline.DEFAULT_CONTEXT_POS),
                    "context_vocabulary_sha256": None,
                    "ner_pair_offset": 2,
                }
                with open(marker, "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle)
            with self.assertRaisesRegex(RuntimeError, "incompatible bin_size"):
                pipeline.validate_reduction_inputs(root, [])


class SharedFeatureInputTests(unittest.TestCase):
    def test_strict_lemma_validation_rejects_null_identifiers(self):
        try:
            import duckdb
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("duckdb and pyarrow are required")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.parquet"
            pq.write_table(pa.table({"word": pa.array([None], type=pa.string()),
                                     "count": pa.array([1], type=pa.int64())}), path)
            connection = duckdb.connect()
            try:
                with self.assertRaisesRegex(RuntimeError, "Strict lemmatization failed"):
                    pipeline._reject_null_identifiers(connection, path, ["word"])
            finally:
                connection.close()
            self.assertFalse(path.exists())

    def test_google_adapter_is_readable_by_shared_coha_feature_loader(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reduced = root / "reduced"
            reduced.mkdir()

            def write(name, schema_name, rows):
                table = pa.Table.from_pylist(rows, schema=pipeline.arrow_schema(schema_name, pa))
                pq.write_table(table, reduced / f"{name}.parquet")

            count = {"decade": 1900, "count": 3}
            surface_pair = {"modifier_surface": "cold", "modifier_pos": "ADJ",
                            "head_surface": "Waters", "head_pos": "NOUN"}
            norm_pair = {"modifier_norm": "cold", "modifier_pos": "ADJ",
                         "head_norm": "waters", "head_pos": "NOUN"}
            write("bigrams", "bigrams", [
                {**surface_pair, **count},
                {**surface_pair, "head_surface": "water", "decade": 1900, "count": 5},
            ])
            write("dependencies", "dependencies", [
                {**surface_pair, **count, "direction": "head=>dependent"},
                {**surface_pair, "head_surface": "water", "decade": 1900,
                 "count": 5, "direction": "head=>dependent"},
            ])
            write("contexts", "contexts", [{**norm_pair,
                "context_norm": "freeze", "context_pos": "VERB",
                "side": "R", "distance": 1, **count},
                {**norm_pair, "head_norm": "water", "context_norm": "freezes",
                 "context_pos": "VERB", "side": "L", "distance": 2,
                 "decade": 1900, "count": 5}])
            write("word_contexts", "word_contexts", [{
                "word_norm": "water", "word_pos": "NOUN",
                "context_norm": "freeze", "context_pos": "VERB",
                "side": "R", "distance": 1, **count}])
            write("unigrams", "unigrams", [{
                "surface": "water", "pos": "NOUN", **count}])
            write("lexicon", "lexicon", [
                {"surface": "cold", "norm": "cold", "pos": "ADJ", "lemma": "cold", "count": 3},
                {"surface": "water", "norm": "water", "pos": "NOUN", "lemma": "water", "count": 3},
                {"surface": "Waters", "norm": "waters", "pos": "NOUN", "lemma": "water", "count": 3},
                {"surface": "freeze", "norm": "freeze", "pos": "VERB", "lemma": "freeze", "count": 3},
                {"surface": "freezes", "norm": "freezes", "pos": "VERB", "lemma": "freeze", "count": 5},
            ])
            (reduced / "reduction_manifest.json").write_text(json.dumps({
                "source_info": {
                    "stage": "reduce", "status": "complete",
                    "corpus": pipeline.corpus_record(),
                    "parameters": {"context_vocabulary_sha256": "test_data",
                                   "context_vocabulary": None},
                    "time": pipeline.time_record(1900, 1909, 10),
                }
            }))
            pipeline.make_feature_inputs(SimpleNamespace(
                output=str(root), feature_output=None, lexicon=None, allow_partial=False,
                threads=1, memory_limit=None, temp_directory=str(root / "tmp"),
                include_dependencies=True,
            ))
            feature_dir = root / "extracted" / "feature_inputs" / "local5"
            manifest = json.loads((feature_dir / "corpus_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(manifest["representation_version"], 5)
            self.assertEqual(manifest["word_target_pos"], "all_lexical")
            self.assertEqual(manifest["bigram_target_pos"], "all_lexical")
            self.assertEqual(set(manifest["table_files"]),
                             {"unigrams", "unigram_counts", "bigrams", "bigram_counts"})

            annotation_path = root / "extracted" / "tables" / "bigram_annotations.parquet"
            annotations = pq.read_table(annotation_path).to_pandas()
            self.assertEqual(annotations.iloc[0]["modifier"], "cold_ADJ")
            self.assertEqual(annotations.iloc[0]["head"], "water_NOUN")
            self.assertEqual(set(annotations["head_surface"]), {"Waters", "water"})
            self.assertTrue(annotations["pair_ner"].isna().all())
            self.assertTrue((root / "final" / "bigram_annotations.parquet").is_file())
            self.assertFalse((root / "extracted" / "tables" / "bigram_counts.parquet").exists())
            self.assertFalse((root / "final" / "bigram_counts.parquet").exists())
            self.assertEqual(list(root.rglob("bigram_counts.parquet")),
                             [feature_dir / "bigram_counts.parquet"])
            self.assertEqual(sorted(path.name for path in feature_dir.glob("*manifest.json")),
                             ["corpus_manifest.json"])
            self.assertFalse((root / "final" / "contexts.parquet").exists())
            self.assertTrue((root / "reduced" / "contexts.parquet").is_file())

            # Validate the reusable exported specification without importing a
            # downstream compositionality implementation or requiring SciPy.
            from src.source_info import validate_feature_input_manifest
            self.assertIs(validate_feature_input_manifest(manifest), manifest)
            exported = pq.read_table(feature_dir / "bigrams.parquet")
            self.assertEqual(exported.column_names,
                             ["modifier", "head", "time", "context", "count"])
            self.assertEqual(exported.to_pylist()[0]["modifier"], "cold_ADJ")
            canonical_bigrams = pq.read_table(root / "final" / "bigrams.parquet").to_pylist()
            self.assertEqual(canonical_bigrams, [{
                "modifier": "cold_ADJ", "head": "water_NOUN", "time": 1900,
                "context": "freeze_VERB", "count": 8,
            }])
            canonical_counts = pq.read_table(feature_dir / "bigram_counts.parquet").to_pylist()
            self.assertEqual(canonical_counts, [{
                "modifier": "cold_ADJ", "head": "water_NOUN", "time": 1900, "count": 8,
            }])
            dependencies = pq.read_table(root / "extracted" / "tables" / "dependencies.parquet").to_pylist()
            self.assertEqual(dependencies, [{
                "modifier": "cold_ADJ", "head": "water_NOUN", "time": 1900,
                "direction": "head=>dependent", "count": 8,
            }])

            feature_before = (feature_dir / "bigrams.parquet").read_bytes()
            pipeline.make_feature_inputs(SimpleNamespace(
                output=str(root), feature_output=None, tables_output=None,
                lexicon=None, allow_partial=False, annotations_only=True,
                threads=1, memory_limit=None, temp_directory=str(root / "tmp"),
                include_dependencies=False,
            ))
            self.assertEqual(
                (feature_dir / "bigrams.parquet").read_bytes(), feature_before
            )

    def test_ner_summary_is_occurrence_weighted_and_time_specific(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reduced = root / "reduced"
            reduced.mkdir()
            bigram = {
                "modifier_surface": "New", "modifier_pos": "ADJ",
                "head_surface": "York", "head_pos": "NOUN",
                "decade": 1900, "count": 10,
            }
            pq.write_table(pa.Table.from_pylist(
                [bigram], schema=pipeline.arrow_schema("bigrams", pa)
            ), reduced / "bigrams.parquet")
            snippets = [
                {**bigram, "tokens_json": '["in", "New", "York", "today"]',
                 "modifier_index": 1, "head_index": 2,
                 "sentence_start": True, "sentence_end": False, "count": 4},
                {**bigram, "tokens_json": '["a", "new", "york", "style"]',
                 "modifier_surface": "New", "head_surface": "York",
                 "modifier_index": 1, "head_index": 2,
                 "sentence_start": False, "sentence_end": False, "count": 2},
            ]
            pq.write_table(pa.Table.from_pylist(
                snippets, schema=pipeline.arrow_schema("ner_snippets", pa)
            ), reduced / "ner_snippets.parquet")
            predictions = [
                {"tokens_json": snippets[0]["tokens_json"], "modifier_index": 1,
                 "head_index": 2, "modifier_ner": "GPE", "head_ner": "GPE",
                 "pair_ner": "GPE", "same_entity_span": True},
                {"tokens_json": snippets[1]["tokens_json"], "modifier_index": 1,
                 "head_index": 2, "modifier_ner": "NONE", "head_ner": "NONE",
                 "pair_ner": "NONE", "same_entity_span": False},
            ]
            pq.write_table(pa.Table.from_pylist(
                predictions, schema=pipeline.arrow_schema("ner_predictions", pa)
            ), reduced / "ner_predictions.parquet")
            pipeline.ner_aggregate(SimpleNamespace(output=str(root), predictions=None))
            result = pq.read_table(reduced / "pair_ner_summary.parquet").to_pylist()[0]
            self.assertEqual(result["pair_ner"], "GPE")
            self.assertAlmostEqual(result["pair_ner_confidence"], 4 / 6)
            self.assertAlmostEqual(result["same_entity_span_fraction"], 4 / 6)
            self.assertAlmostEqual(result["ner_coverage"], 6 / 10)
            self.assertEqual(result["time"], 1900)


class PartitionedAggregationTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("duckdb and pyarrow are required")
        self.duckdb, self.pa, self.pq = duckdb, pa, pq
        from src import partitioned_parquet
        self.partitioned = partitioned_parquet
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sources = []
        for index in range(3):
            path = self.root / f"input-{index}.parquet"
            rows = [
                {"word": f"w{i % 19}", "time": 1900 + (i % 2) * 10,
                 "count": i + 1 + index} for i in range(100)
            ]
            self.pq.write_table(self.pa.Table.from_pylist(rows), path, row_group_size=10)
            self.sources.append(path)
        self.options = self.partitioned.partition_options(SimpleNamespace(
            partitions=4, partition_batch_rows=70, arrow_batch_rows=7,
            max_split_depth=3, threads=1, memory_limit="128MB",
        ))
        self.destination = self.root / "sum.parquet"

    def run_sum(self):
        return self.partitioned.partitioned_sum(
            self.sources, self.destination, ["word", "time"],
            "SELECT word, time, count FROM partition_input", ["word", "time", "count"],
            options=self.options, work_root=self.root / "checkpoints",
            spill_root=self.root / "spill",
        )

    def assert_reference(self):
        with self.duckdb.connect() as connection:
            expected = connection.execute(
                "SELECT word, time, SUM(count)::BIGINT FROM read_parquet(?) "
                "GROUP BY word,time ORDER BY word,time", [[str(p) for p in self.sources]],
            ).fetchall()
            actual = connection.execute(
                "SELECT word,time,count FROM read_parquet(?) ORDER BY word,time",
                [str(self.destination)],
            ).fetchall()
        self.assertEqual(actual, expected)

    def test_many_batches_match_global_sum_and_reuse_completed_output(self):
        observed = self.run_sum()
        self.assertEqual(observed["input_rows"], 300)
        self.assertEqual(observed["count_sum"], 15450)
        self.assert_reference()
        with mock.patch.object(self.partitioned, "_arrow_reader",
                               side_effect=AssertionError("completed source was reread")):
            reused = self.run_sum()
        self.assertTrue(reused["reused"])

    def test_interrupted_run_reuses_shuffle_and_completed_buckets(self):
        aggregate = self.partitioned._aggregate_bucket
        completed = []

        def interrupt_after_first(*args, **kwargs):
            if completed:
                raise RuntimeError("simulated interruption")
            result = aggregate(*args, **kwargs)
            completed.extend(result[0])
            return result

        with mock.patch.object(self.partitioned, "_aggregate_bucket",
                               side_effect=interrupt_after_first):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                self.run_sum()
        mtimes = {path: path.stat().st_mtime_ns for path in completed}
        with mock.patch.object(self.partitioned, "_arrow_reader",
                               side_effect=AssertionError("completed shuffle was reread")):
            self.run_sum()
        self.assertEqual(mtimes, {path: path.stat().st_mtime_ns for path in completed})
        self.assert_reference()

    def test_changed_source_invalidates_published_checkpoint(self):
        before = self.run_sum()
        self.pq.write_table(self.pa.Table.from_pylist([
            {"word": "new", "time": 1900, "count": 900},
        ]), self.sources[0])
        after = self.run_sum()
        self.assertNotEqual(before["checkpoint_directory"], after["checkpoint_directory"])
        self.assert_reference()

    def test_oom_recursively_splits_then_matches_global_sum(self):
        real_scope = self.partitioned.connection_scope
        failures = [2]

        class InjectOOM:
            def __init__(self, connection):
                self.connection = connection

            def __getattr__(self, name):
                return getattr(self.connection, name)

            def execute(proxy, query, *args, **kwargs):
                if query.startswith("COPY (SELECT") and "SUM(count)" in query and failures[0]:
                    failures[0] -= 1
                    raise self.duckdb.OutOfMemoryException("simulated OOM")
                return proxy.connection.execute(query, *args, **kwargs)

        @contextlib.contextmanager
        def injected_scope(*args, **kwargs):
            with real_scope(*args, **kwargs) as connection:
                yield InjectOOM(connection)

        with mock.patch.object(self.partitioned, "connection_scope", injected_scope):
            self.run_sum()
        self.assertEqual(failures[0], 0)
        self.assertEqual(len(list((self.root / "checkpoints").rglob("split.json"))), 2)
        self.assert_reference()

    def test_missing_lemma_aborts_before_replacing_canonical_output(self):
        self.destination.write_bytes(b"existing-output")
        with self.assertRaisesRegex(Exception, "Strict lemmatization failed"):
            self.partitioned.partitioned_sum(
                self.sources, self.destination, ["word", "time"],
                "SELECT NULL::VARCHAR AS word,time,count FROM partition_input",
                ["word", "time", "count"], options=self.options,
                work_root=self.root / "checkpoints", spill_root=self.root / "spill",
                required_identifiers=["word"],
            )
        self.assertEqual(self.destination.read_bytes(), b"existing-output")

    def test_empty_projection_publishes_typed_empty_table(self):
        observed = self.partitioned.partitioned_sum(
            self.sources, self.destination, ["word", "time"],
            "SELECT word,time,count FROM partition_input WHERE FALSE",
            ["word", "time", "count"], options=self.options,
            work_root=self.root / "checkpoints", spill_root=self.root / "spill",
            preserve_count_mass=False,
        )
        self.assertEqual(observed["rows"], 0)
        table = self.pq.read_table(self.destination)
        self.assertEqual(table.column_names, ["word", "time", "count"])
        self.assertEqual(table.num_rows, 0)


class CompletionReuseTests(unittest.TestCase):
    def test_empty_extraction_writes_marker_without_raw_task_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "empty-2.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write("untagged words\t1900,4,2\n")
            args = SimpleNamespace(
                output=str(root / "output"), year_min=1800, year_max=2019,
                bin_size=10, context_pos="ADJ,ADV,NOUN,NUM,VERB",
                context_vocab=None, ner_pair_offset=2, batch_size=10,
                progress_every=0, timeout=10, manifest=None,
                force=False, replace_incompatible=False,
            )
            pipeline.extract_source(args, str(source), 2, "empty")
            marker = pipeline.completion_marker(args.output, 2, "empty")
            self.assertTrue(marker.is_file())
            self.assertFalse(pipeline.task_data_dir(args.output, 2, "empty").exists())
            saved = json.loads(marker.read_text())
            self.assertEqual(saved["bigrams_rows"], 0)
            self.assertEqual(saved["word_contexts_rows"], 0)

    def test_extract_completion_requires_matching_settings_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "unigrams.parquet"
            output.write_bytes(b"parquet-placeholder")
            expected = {
                "schema_version": 1, "corpus_version": pipeline.CORPUS_VERSION,
                "source": "source.gz", "order": 1, "shard": "one",
                "year_min": 1820, "year_max": 2020, "bin_size": 10,
                "context_pos": sorted(pipeline.DEFAULT_CONTEXT_POS),
                "context_vocabulary_sha256": None, "ner_pair_offset": 2,
            }
            marker = root / "stats.json"
            marker.write_text(json.dumps(expected))
            self.assertEqual(
                pipeline.validate_completed_task(marker, expected, [output]),
                expected,
            )
            output.unlink()
            with self.assertRaisesRegex(RuntimeError, "does not match its outputs"):
                pipeline.validate_completed_task(marker, expected, [output])

    def test_extract_completion_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "stats.json"
            marker.write_text(json.dumps({"bin_size": 20}))
            with self.assertRaisesRegex(RuntimeError, "incompatible settings"):
                pipeline.validate_completed_task(marker, {"bin_size": 10})


if __name__ == "__main__":
    unittest.main()
