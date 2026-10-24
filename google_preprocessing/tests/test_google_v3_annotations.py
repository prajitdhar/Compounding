"""Annotation-only regressions using tiny datasets and a deterministic NER.

No production extracts, shared-scratch checkpoints, or downloaded model are
used. The four scientific feature-input tables must remain byte-identical.
"""

import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


GOOGLE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(GOOGLE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GOOGLE_DIRECTORY))


class AnnotationTestData(unittest.TestCase):
    """Shared miniature corpus; unknown evidence is deliberately retained."""

    def setUp(self):
        try:
            import duckdb
            import pyarrow as pa
            import pyarrow.parquet as pq
            import spacy
        except ImportError:
            self.skipTest("spaCy, DuckDB, and PyArrow are required")
        self.duckdb, self.pa, self.pq, self.spacy = duckdb, pa, pq, spacy
        spec = importlib.util.spec_from_file_location(
            "google_v3_annotations", GOOGLE_DIRECTORY / "google_v3_annotations.py")
        self.annotations = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.annotations
        spec.loader.exec_module(self.annotations)
        self.annotations.ensure_spacy_compatibility()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reduced = self.root / "reduced"
        self.reduced.mkdir()
        self.features = self.root / "extracted/feature_inputs/local5"
        self.features.mkdir(parents=True)
        self.tables = self.root / "extracted/tables"
        self.tables.mkdir(parents=True)
        self.work = self.root / "annotation-work"
        self.spill = self.root / "spill"
        self.model = self.root / "model"
        self.fake_ner = spacy.blank("en")
        # Register deterministic rule matching under the same pipeline name as
        # a trained NER component, so production loading/checkpoint code runs.
        ruler = self.fake_ner.add_pipe("entity_ruler", name="ner")
        ruler.add_patterns([{"label": "GPE", "pattern": [
            {"TEXT": "visit"}, {"TEXT": "New"}, {"TEXT": "York"}]}])
        self.fake_ner.to_disk(self.model)

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, name, rows, *, directory=None, schema=None):
        path = (directory or self.reduced) / f"{name}.parquet"
        self.pq.write_table(self.pa.Table.from_pylist(rows, schema=schema), path,
                            row_group_size=2)
        return path

    @staticmethod
    def pair(modifier="New", head="York", modifier_pos="ADJ", decade=1900):
        return {"modifier_surface": modifier, "modifier_pos": modifier_pos,
                "head_surface": head, "head_pos": "NOUN", "decade": decade}

    def snippet(self, words, count, *, decade=1900, start=False, end=False,
                modifier="New", head="York", modifier_pos="ADJ"):
        return {**self.pair(modifier, head, modifier_pos, decade),
                "tokens_json": json.dumps(words), "modifier_index": 1,
                "head_index": 2, "sentence_start": start,
                "sentence_end": end, "count": count}

    def write_test_data(self):
        self.write("bigrams", [
            {**self.pair(), "count": 20},
            {**self.pair(decade=1910), "count": 7},
            {**self.pair("Other", "Thing", "NOUN"), "count": 5},
            {**self.pair("very", "thing", "ADV"), "count": 9},
        ])
        self.write("ner_snippets", [
            self.snippet(["visit", "New", "York", "today"], 6, start=True),
            self.snippet(["ordinary", "New", "York", "words"], 2, end=True),
            self.snippet(["rare", "New", "York", "evidence"], 1),
            self.snippet(["visit", "New", "York", "today"], 3, decade=1910),
        ])
        self.write("dependencies", [
            {**self.pair(), "direction": "head=>dependent", "count": 30},
            {**self.pair(), "direction": "dependent<=head", "count": 5},
        ])
        self.write("lexicon", [
            {"surface": "New", "norm": "new", "pos": "ADJ", "lemma": "new", "count": 20},
            {"surface": "York", "norm": "york", "pos": "NOUN", "lemma": "york", "count": 20},
            {"surface": "Other", "norm": "other", "pos": "NOUN", "lemma": "other", "count": 5},
            {"surface": "Thing", "norm": "thing", "pos": "NOUN", "lemma": "thing", "count": 5},
        ])
        self.write("bigrams", [{"modifier": "new_ADJ", "head": "york_NOUN", "time": 1900,
                                "context": "visit_VERB", "count": 6}], directory=self.features)
        self.write("bigram_counts", [{"modifier": "new_ADJ", "head": "york_NOUN",
                                      "time": 1900, "count": 20}], directory=self.features)
        self.write("unigrams", [{"word": "new_ADJ", "time": 1900,
                                 "context": "visit_VERB", "count": 6}], directory=self.features)
        self.write("unigram_counts", [{"word": "new_ADJ", "time": 1900,
                                       "count": 20}], directory=self.features)
        annotations = [
            {"modifier": "new_ADJ", "head": "york_NOUN", "time": 1900,
             "modifier_surface": "New", "head_surface": "York",
             "modifier_pos": "ADJ", "head_pos": "NOUN", "count": 20},
            {"modifier": "new_ADJ", "head": "york_NOUN", "time": 1910,
             "modifier_surface": "New", "head_surface": "York",
             "modifier_pos": "ADJ", "head_pos": "NOUN", "count": 7},
            {"modifier": "other_NOUN", "head": "thing_NOUN", "time": 1900,
             "modifier_surface": "Other", "head_surface": "Thing",
             "modifier_pos": "NOUN", "head_pos": "NOUN", "count": 5},
        ]
        null_fields = {"modifier_ner": self.pa.string(),
                       "head_ner": self.pa.string(), "pair_ner": self.pa.string(),
                       "modifier_ner_confidence": self.pa.float64(),
                       "head_ner_confidence": self.pa.float64(),
                       "pair_ner_confidence": self.pa.float64(),
                       "same_entity_span_fraction": self.pa.float64(),
                       "ner_coverage": self.pa.float64(),
                       "classified_ner_count": self.pa.int64(),
                       "sentence_start_fraction": self.pa.float64(),
                       "sentence_end_fraction": self.pa.float64(),
                       "ner_distribution_json": self.pa.string()}
        base_schema = self.pa.Table.from_pylist(annotations).schema
        for row in annotations:
            row.update({name: None for name in null_fields})
        annotation_schema = self.pa.schema([
            *base_schema, *(self.pa.field(name, kind)
                            for name, kind in null_fields.items())])
        self.write("bigram_annotations", annotations, directory=self.tables,
                   schema=annotation_schema)
        self.feature_digests = {path: self.digest(path)
                                for path in self.features.glob("*.parquet")}
        self.feature_manifest_records = {
            path.stem: {"name": f"local5/{path.name}", "path": str(path),
                        "bytes": path.stat().st_size, "sha256": digest,
                        "format": "parquet", "columns": self.pq.read_schema(path).names}
            for path, digest in self.feature_digests.items()}
        (self.features / "corpus_manifest.json").write_text(json.dumps({
            "tables": self.feature_manifest_records,
            "table_files": {name: f"{name}.parquet"
                            for name in self.feature_manifest_records},
            "source_info": {"status": "complete", "parameters": {
                "context_vocabulary_sha256": "frozen-test-vocabulary"},
                "outputs": list(self.feature_manifest_records.values())}}))
        self.reduced_digests = {path: self.digest(path)
                                for path in self.reduced.glob("*.parquet")}

    def assert_features_unchanged(self):
        self.assertEqual({path: self.digest(path) for path in self.feature_digests},
                         self.feature_digests)
        self.assertEqual({path: self.digest(path) for path in self.reduced_digests},
                         self.reduced_digests)

    def doc(self, words, entities):
        from spacy.tokens import Doc, Span
        doc = Doc(self.fake_ner.vocab, words=words)
        doc.ents = [Span(doc, start, end, label=label)
                    for start, end, label in entities]
        return doc

    def read_rows(self, path):
        return self.pq.read_table(path).to_pylist()

    def run_cli(self, stage, *extra):
        """Exercise the same entrypoint that the Slurm wrapper invokes."""
        arguments = [stage, "--output", str(self.root),
                     "--work-directory", str(self.work),
                     "--temp-directory", str(self.spill),
                     "--model", str(self.model), "--shards", "2",
                     "--partitions", "2", "--threads", "1",
                     "--memory-limit", "256MB", "--min-snippet-count", "2",
                     "--arrow-batch-rows", "2", "--partition-batch-rows", "2",
                     *extra]
        return self.annotations.main(arguments)

    def predict_all(self):
        for shard in range(2):
            self.run_cli("predict", "--shard-index", str(shard),
                         "--ner-batch-size", "1", "--prediction-batch-rows", "1")


class EntityClassificationTests(AnnotationTestData):
    def test_pair_membership_requires_the_same_entity_span(self):
        doc = self.doc(["visit", "New", "York", "today"], [(1, 3, "GPE")])
        self.assertEqual(self.annotations.classify_pair_entities(doc, 1, 2), {
            "modifier_ner": "GPE", "head_ner": "GPE", "pair_ner": "GPE",
            "same_entity_span": True, "either_token_entity": True})

    def test_separate_entities_are_not_an_entity_pair(self):
        doc = self.doc(["visit", "New", "York", "today"],
                       [(1, 2, "PERSON"), (2, 3, "ORG")])
        result = self.annotations.classify_pair_entities(doc, 1, 2)
        self.assertEqual(result["modifier_ner"], "PERSON")
        self.assertEqual(result["head_ner"], "ORG")
        self.assertEqual(result["pair_ner"], "NONE")
        self.assertFalse(result["same_entity_span"])
        self.assertTrue(result["either_token_entity"])

    def test_one_entity_token_and_assessed_none_remain_distinct(self):
        doc = self.doc(["visit", "New", "York", "today"], [(1, 2, "GPE")])
        result = self.annotations.classify_pair_entities(doc, 1, 2)
        self.assertEqual(result["head_ner"], "NONE")
        self.assertEqual(result["pair_ner"], "NONE")
        self.assertFalse(result["same_entity_span"])
        self.assertTrue(result["either_token_entity"])
        empty = self.annotations.classify_pair_entities(
            self.doc(["visit", "New", "York", "today"], []), 1, 2)
        self.assertEqual(empty["modifier_ner"], "NONE")
        self.assertEqual(empty["head_ner"], "NONE")
        self.assertEqual(empty["pair_ner"], "NONE")
        self.assertFalse(empty["same_entity_span"])
        self.assertFalse(empty["either_token_entity"])


class AnnotationPipelineTests(AnnotationTestData):
    def setUp(self):
        super().setUp()
        self.write_test_data()

    def enriched(self):
        rows = self.read_rows(self.tables / "bigram_annotations.parquet")
        return {(row["modifier_surface"], row["head_surface"], row["time"]): row
                for row in rows}

    def test_end_to_end_weighted_ner_and_dependency_evidence(self):
        self.run_cli("prepare")
        predicted_docs = []
        original_pipe = self.fake_ner.pipe

        def record_pipe(docs, *args, **kwargs):
            materialized = list(docs)
            predicted_docs.extend(tuple(token.text for token in doc)
                                  for doc in materialized)
            return original_pipe(materialized, *args, **kwargs)

        with mock.patch("spacy.load", return_value=self.fake_ner), \
                mock.patch.object(self.fake_ner, "pipe", side_effect=record_pipe):
            self.predict_all()
        # The repeated visit snippet is predicted once across two decades;
        # below-policy evidence is not incorrectly treated as a NONE result.
        self.assertCountEqual(predicted_docs, [
            ("visit", "New", "York", "today"),
            ("ordinary", "New", "York", "words")])
        self.run_cli("aggregate")
        self.run_cli("export")
        published_rows = self.read_rows(self.tables / "bigram_annotations.parquet")
        self.assertEqual(len(published_rows), 3)
        self.assertEqual(sum(row["count"] for row in published_rows), 32)
        result = self.enriched()
        self.assertEqual(set(result), {("New", "York", 1900),
                                       ("New", "York", 1910),
                                       ("Other", "Thing", 1900)})
        row = result[("New", "York", 1900)]
        self.assertEqual(row["modifier"], "new_ADJ")
        self.assertEqual(row["head"], "york_NOUN")
        self.assertEqual(row["count"], 20)
        self.assertEqual(row["pair_ner"], "GPE")
        self.assertEqual(row["classified_ner_count"], 8)
        self.assertAlmostEqual(row["same_entity_span_fraction"], .75)
        self.assertAlmostEqual(row["either_token_entity_fraction"], .75)
        self.assertAlmostEqual(row["ner_coverage"], .4)
        self.assertEqual(row["ner_status"], "partial")
        self.assertEqual(json.loads(row["ner_distribution_json"]),
                         {"GPE": .75, "NONE": .25})
        self.assertAlmostEqual(row["sentence_start_fraction"], .75)
        self.assertAlmostEqual(row["sentence_end_fraction"], .25)
        unknown = result[("Other", "Thing", 1900)]
        self.assertEqual(unknown["ner_status"], "unavailable")
        self.assertIsNone(unknown["pair_ner"])
        self.assertIsNone(unknown["same_entity_span_fraction"])
        self.assertIsNone(unknown["ner_coverage"])
        self.assertEqual(self.digest(self.tables / "bigram_annotations.parquet"),
                         self.digest(self.root / "final/bigram_annotations.parquet"))
        manifest = json.loads((self.features / "corpus_manifest.json").read_text())
        self.assertEqual(manifest["source_info"]["parameters"]["context_vocabulary_sha256"],
                         "frozen-test-vocabulary")
        self.assertEqual(manifest["tables"], self.feature_manifest_records)
        self.assertEqual(manifest["table_files"], {
            name: f"{name}.parquet" for name in self.feature_manifest_records})
        self.assertEqual(manifest["source_info"]["outputs"],
                         list(self.feature_manifest_records.values()))
        annotation_paths = [self.tables / "bigram_annotations.parquet",
                            self.root / "final/bigram_annotations.parquet"]
        published = {path: (self.digest(path), path.stat().st_size, path.stat().st_mtime_ns)
                     for path in annotation_paths}
        with mock.patch.object(self.annotations, "_route_query",
                               side_effect=AssertionError("completed annotation shuffle reran")), \
                mock.patch.object(self.annotations.pp, "_concatenate",
                                  side_effect=AssertionError("completed annotation concatenation reran")):
            self.run_cli("export")
        self.assertEqual({path: (self.digest(path), path.stat().st_size, path.stat().st_mtime_ns)
                          for path in annotation_paths}, published)
        with self.assertRaisesRegex(RuntimeError, "(?i)(NER|downgrade)"):
            self.run_cli("export", "--dependencies-only")
        self.assert_features_unchanged()

    def test_dependencies_only_keeps_unknown_ner_and_preserves_features(self):
        self.run_cli("export", "--dependencies-only")
        rows = self.enriched()
        row = rows[("New", "York", 1900)]
        self.assertEqual(row["ner_status"], "not_run")
        self.assertIsNone(row["pair_ner"])
        self.assertTrue(row["dependency_evidence_observed"])
        self.assertEqual(row["dependency_head_to_dependent_count"], 30)
        self.assertEqual(row["dependency_dependent_to_head_count"], 5)
        # The arc population is not the adjacent-bigram population. A count
        # greater than count20 must not be clipped or advertised as probability.
        self.assertFalse(any("probability" in column for column in row))
        self.assert_features_unchanged()

    def test_aggregation_rejects_unfinished_prediction_shards(self):
        self.run_cli("prepare")
        with self.assertRaisesRegex((RuntimeError, ValueError),
                                    "(?i)(predict|shard|complete)"):
            self.run_cli("aggregate")

    def test_stale_feature_manifest_size_is_refused(self):
        path = self.features / "corpus_manifest.json"
        record = json.loads(path.read_text())
        record["tables"]["bigrams"]["bytes"] += 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "(?i)(size|bytes|source information|manifest)"):
            self.run_cli("export", "--dependencies-only")
        self.assert_features_unchanged()

    def test_stale_feature_manifest_path_is_refused(self):
        path = self.features / "corpus_manifest.json"
        record = json.loads(path.read_text())
        record["tables"]["bigrams"]["path"] = str(self.root / "wrong/bigrams.parquet")
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "(?i)(path|source information|manifest)"):
            self.run_cli("export", "--dependencies-only")
        self.assert_features_unchanged()

    def test_pinned_baseline_cannot_be_reused_for_another_dataset(self):
        self.run_cli("export", "--dependencies-only")
        other_root = self.root / "other-dataset"
        with self.assertRaisesRegex(RuntimeError, "(?i)(different|another).*dataset"):
            self.annotations._baseline(other_root, self.work)
        self.assert_features_unchanged()

    def test_null_dependency_direction_is_refused_before_publication(self):
        path = self.reduced / "dependencies.parquet"
        rows = self.read_rows(path)
        rows[0]["direction"] = None
        self.write("dependencies", rows)
        self.reduced_digests[path] = self.digest(path)
        before = self.digest(self.tables / "bigram_annotations.parquet")
        with self.assertRaisesRegex(RuntimeError, "(?i)(unknown|invalid).*dependency"):
            self.run_cli("export", "--dependencies-only")
        self.assertEqual(self.digest(self.tables / "bigram_annotations.parquet"), before)
        self.assert_features_unchanged()

    def test_below_threshold_evidence_is_unknown_not_assessed_none(self):
        path = self.reduced / "ner_snippets.parquet"
        rows = self.read_rows(path)
        rows.append(self.snippet(["rare", "Other", "Thing", "evidence"], 1,
                                 modifier="Other", head="Thing", modifier_pos="NOUN"))
        self.write("ner_snippets", rows)
        self.reduced_digests[path] = self.digest(path)
        self.run_cli("prepare")
        self.predict_all()
        self.run_cli("aggregate")
        self.run_cli("export")
        row = self.enriched()[("Other", "Thing", 1900)]
        self.assertEqual(row["ner_status"], "below_threshold")
        self.assertEqual(row["available_ner_count"], 1)
        self.assertEqual(row["classified_ner_count"], 0)
        self.assertIsNone(row["pair_ner"])
        self.assertIsNone(row["same_entity_span_fraction"])
        self.assert_features_unchanged()

    def test_completed_predictions_resume_without_rerunning_model(self):
        self.run_cli("prepare")
        self.predict_all()
        before = {path: self.digest(path) for path in self.work.rglob("*.parquet")}
        with mock.patch("spacy.load", side_effect=AssertionError("completed NER was rerun")):
            self.predict_all()
        self.assertEqual({path: self.digest(path) for path in before}, before)
        self.assert_features_unchanged()

    def test_interrupted_prediction_reuses_completed_batches(self):
        common = ["--shards", "1", "--partitions", "1"]
        self.run_cli("prepare", *common)
        original_pipe = self.fake_ner.pipe
        calls = []

        def interrupt_second(docs, *args, **kwargs):
            batch = list(docs)
            calls.append([tuple(token.text for token in doc) for doc in batch])
            if len(calls) == 2:
                raise RuntimeError("intentional test interruption")
            return original_pipe(batch, *args, **kwargs)

        options = [*common, "--shard-index", "0", "--ner-batch-size", "1",
                   "--prediction-batch-rows", "1"]
        with mock.patch("spacy.load", return_value=self.fake_ner), \
                mock.patch.object(self.fake_ner, "pipe", side_effect=interrupt_second):
            with self.assertRaisesRegex(RuntimeError, "intentional test interruption"):
                self.run_cli("predict", *options)
        resumed = []

        def record_resume(docs, *args, **kwargs):
            batch = list(docs)
            resumed.extend(tuple(token.text for token in doc) for doc in batch)
            return original_pipe(batch, *args, **kwargs)

        with mock.patch("spacy.load", return_value=self.fake_ner), \
                mock.patch.object(self.fake_ner, "pipe", side_effect=record_resume):
            self.run_cli("predict", *options)
        self.assertEqual(len(resumed), 1)
        self.assertNotIn(calls[0][0], resumed)
        self.assert_features_unchanged()

    def test_changed_model_cannot_reuse_old_predictions(self):
        self.run_cli("prepare")
        self.predict_all()
        self.fake_ner.get_pipe("ner").add_patterns([{
            "label": "ORG", "pattern": [{"TEXT": "ordinary"},
                {"TEXT": "New"}, {"TEXT": "York"}]}])
        self.fake_ner.to_disk(self.model)
        self.run_cli("prepare")
        calls = []
        original_load = self.spacy.load

        def record_load(*args, **kwargs):
            calls.append(args)
            return original_load(*args, **kwargs)

        with mock.patch("spacy.load", side_effect=record_load):
            try:
                self.predict_all()
            except (RuntimeError, ValueError) as error:
                # An explicit stale-cache refusal is as safe as a fresh model
                # workspace. Silent reuse of a changed model is never safe.
                self.assertRegex(str(error), "(?i)(model|identity|incompatible)")
            else:
                self.assertTrue(calls, "changed model silently reused predictions")
        self.assert_features_unchanged()


class AnnotationGpuTests(AnnotationTestData):
    """GPU policy is tested with mocks; no CUDA context is initialized."""

    def args(self, stage="predict", *extra):
        arguments = [stage, "--output", str(self.root),
                     "--work-directory", str(self.work),
                     "--temp-directory", str(self.spill),
                     "--model", str(self.model), "--threads", "1",
                     "--shards", "2", "--partitions", "2",
                     "--memory-limit", "256MB", "--min-snippet-count", "2"]
        if stage == "predict":
            arguments += ["--shard-index", "0"]
        return self.annotations.parser().parse_args([*arguments, *extra])

    @contextlib.contextmanager
    def fake_hardware(self, *, cuda_build="12.1", available=True, require_gpu=None):
        torch = SimpleNamespace(
            __version__="mock-torch", version=SimpleNamespace(cuda=cuda_build),
            set_num_threads=mock.Mock(), set_float32_matmul_precision=mock.Mock(),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True)),
            cuda=SimpleNamespace(
                is_available=mock.Mock(return_value=available),
                device_count=mock.Mock(return_value=2),
                set_device=mock.Mock(), current_device=mock.Mock(return_value=0),
                get_device_name=mock.Mock(return_value="mock-gpu"),
                get_device_capability=mock.Mock(return_value=(8, 0)),
                get_device_properties=mock.Mock(return_value=SimpleNamespace(
                    name="mock-gpu", total_memory=80 * 1024**3, major=8, minor=0))))
        cupy = SimpleNamespace(
            __version__="mock-cupy", cuda=SimpleNamespace(
                Device=mock.Mock(), runtime=SimpleNamespace(
                    driverGetVersion=mock.Mock(return_value=12040),
                    runtimeGetVersion=mock.Mock(return_value=12010),
                    getDeviceCount=mock.Mock(return_value=2))))
        torch_tensor, cupy_array = mock.Mock(), mock.Mock()
        torch_tensor.sum.return_value.item.return_value = 120
        cupy_array.sum.return_value.get.return_value = 120
        torch.float32, cupy.float32 = "mock-torch-float32", "mock-cupy-float32"
        torch.arange, cupy.arange = mock.Mock(return_value=torch_tensor), mock.Mock(return_value=cupy_array)
        with mock.patch.dict(sys.modules, {"torch": torch, "cupy": cupy}), \
                mock.patch("spacy.require_gpu", require_gpu or mock.Mock(return_value=True)) as selected, \
                mock.patch("thinc.api.set_gpu_allocator") as allocator:
            yield SimpleNamespace(torch=torch, cupy=cupy,
                                  require_gpu=selected, allocator=allocator)

    def test_device_defaults_are_cpu_and_logical_gpu_zero(self):
        args = self.args()
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.gpu_id, 0)

    def test_cuda_prediction_identity_is_distinct_without_gpu_initialization(self):
        prepared = {"identity": {"stage": "prepare", "immutable_keys": "same"}}
        model = {"name": "mock-test-NER", "fingerprint": "same-frozen-model"}
        with mock.patch("spacy.require_gpu",
                        side_effect=AssertionError("identity construction initialized CUDA")):
            cpu_path, cpu = self.annotations._prediction_workspace(
                self.args("predict", "--device", "cpu"), prepared, model)
            gpu_path, gpu = self.annotations._prediction_workspace(
                self.args("predict", "--device", "cuda"), prepared, model)
        self.assertNotEqual(cpu, gpu)
        self.assertNotEqual(cpu_path, gpu_path)

    def test_physical_gpu_id_does_not_change_semantic_prediction_identity(self):
        prepared = {"identity": {"stage": "prepare", "immutable_keys": "same"}}
        model = {"name": "mock-test-NER", "fingerprint": "same-frozen-model"}
        zero = self.annotations._prediction_workspace(
            self.args("predict", "--device", "cuda", "--gpu-id", "0"), prepared, model)
        one = self.annotations._prediction_workspace(
            self.args("predict", "--device", "cuda", "--gpu-id", "1"), prepared, model)
        self.assertEqual(zero, one)

    def test_negative_gpu_id_is_rejected_before_stage_work(self):
        with self.assertRaisesRegex(ValueError, "(?i)(gpu|nonnegative)"):
            self.run_cli("prepare", "--gpu-id", "-1")

    def test_cuda_requires_gpu_before_loading_and_records_device_properties(self):
        events = []

        def require_gpu(index):
            events.append(("gpu", index))
            return True

        def load(*unused_args, **unused_kwargs):
            events.append(("load", None))
            return self.fake_ner

        args = self.args("predict", "--device", "cuda", "--gpu-id", "1")
        with self.fake_hardware(require_gpu=mock.Mock(side_effect=require_gpu)) as hardware, \
                mock.patch("spacy.load", side_effect=load):
            model = self.annotations._load_model(args)
        self.assertIs(model, self.fake_ner)
        self.assertEqual(events, [("gpu", 1), ("load", None)])
        hardware.allocator.assert_called_once_with("pytorch")
        hardware.torch.set_float32_matmul_precision.assert_called_once_with("highest")
        record = args._ner_runtime
        self.assertEqual(record["device"], "cuda")
        self.assertEqual(record["gpu_id"], 1)
        self.assertEqual(record["gpu_name"], "mock-gpu")
        self.assertEqual(record["compute_capability"], [8, 0])
        self.assertEqual(record["gpu_memory_bytes"], 80 * 1024**3)
        self.assertEqual(record["cuda_driver_version"], 12040)
        self.assertEqual(record["cuda_runtime_version"], 12010)
        self.assertEqual(record["torch_version"], "mock-torch")
        self.assertEqual(record["cupy_version"], "mock-cupy")
        self.assertEqual(record["cuda_build"], "12.1")

    def test_failed_cuda_selection_never_falls_back_to_cpu_or_loads_model(self):
        args = self.args("predict", "--device", "cuda")
        failure = mock.Mock(side_effect=ValueError("mock CUDA selection failure"))
        with self.fake_hardware(require_gpu=failure), \
                mock.patch("spacy.require_cpu") as cpu, mock.patch("spacy.load") as load:
            with self.assertRaisesRegex((RuntimeError, ValueError), "(?i)(cuda|gpu)"):
                self.annotations._load_model(args)
        load.assert_not_called()
        cpu.assert_not_called()

    def test_cpu_only_torch_build_is_refused_in_cuda_mode(self):
        args = self.args("predict", "--device", "cuda")
        with self.fake_hardware(cuda_build=None) as hardware, \
                mock.patch("spacy.load") as load:
            with self.assertRaisesRegex(RuntimeError, "(?i)(cuda|torch|gpu)"):
                self.annotations._load_model(args)
        load.assert_not_called()
        hardware.require_gpu.assert_not_called()

    def test_unavailable_cuda_device_is_refused_before_model_load(self):
        args = self.args("predict", "--device", "cuda")
        with self.fake_hardware(available=False), mock.patch("spacy.load") as load:
            with self.assertRaisesRegex(RuntimeError, "(?i)(cuda|gpu)"):
                self.annotations._load_model(args)
        load.assert_not_called()

    def test_cpu_model_load_does_not_initialize_cuda(self):
        args = self.args("predict", "--device", "cpu")
        with mock.patch("spacy.require_gpu", side_effect=AssertionError("CPU initialized CUDA")), \
                mock.patch("torch.cuda.is_available", side_effect=AssertionError("CPU queried CUDA")), \
                mock.patch("torch.cuda.get_device_properties", side_effect=AssertionError("CPU initialized CUDA")), \
                mock.patch("spacy.load", return_value=self.fake_ner):
            self.assertIs(self.annotations._load_model(args), self.fake_ner)
        self.assertEqual(args._ner_runtime["device"], "cpu")

    def test_cuda_finalize_requires_no_gpu_allocation_or_model_load(self):
        self.write_test_data()
        self.run_cli("prepare", "--device", "cuda")
        with mock.patch.object(self.annotations, "_load_model", return_value=self.fake_ner):
            for shard in range(2):
                self.run_cli("predict", "--device", "cuda", "--shard-index", str(shard),
                             "--ner-batch-size", "1", "--prediction-batch-rows", "1")
        with mock.patch.object(self.annotations, "_configure_device",
                               side_effect=AssertionError("finalize initialized GPU")), \
                mock.patch.object(self.annotations, "_load_model",
                                  side_effect=AssertionError("finalize loaded NER model")), \
                mock.patch("spacy.require_gpu", side_effect=AssertionError("finalize initialized GPU")), \
                mock.patch("torch.cuda.get_device_properties",
                           side_effect=AssertionError("finalize initialized GPU")):
            for shard in range(2):
                self.run_cli("predict", "--device", "cuda", "--shard-index", str(shard),
                             "--ner-batch-size", "1", "--prediction-batch-rows", "1")
            self.run_cli("aggregate", "--device", "cuda")
            self.run_cli("export", "--device", "cuda")
        self.assert_features_unchanged()

    def test_gpu_check_cpu_smoke_uses_tiny_model_without_writing_data(self):
        with mock.patch("spacy.require_gpu", side_effect=AssertionError("CPU check initialized CUDA")):
            result = self.run_cli("gpu-check", "--device", "cpu", "--smoke-test")
        self.assertEqual(result["device"], "cpu")
        self.assertIn("ner", result["pipeline"])
        self.assertTrue(result["smoke_prediction"]["same_entity_span"])
        self.assertEqual(list(self.root.rglob("*.parquet")), [])

    def test_gpu_check_cuda_validates_both_kernel_backends_and_model_smoke(self):
        with self.fake_hardware() as hardware, \
                mock.patch("spacy.load", return_value=self.fake_ner):
            result = self.run_cli("gpu-check", "--device", "cuda", "--smoke-test")
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["cuda_kernel_check"], "passed")
        self.assertTrue(result["smoke_prediction"]["same_entity_span"])
        hardware.torch.arange.assert_called_once_with(
            16, dtype=hardware.torch.float32, device="cuda:0")
        hardware.cupy.arange.assert_called_once_with(16, dtype=hardware.cupy.float32)
        self.assertEqual(list(self.root.rglob("*.parquet")), [])

    def test_gpu_check_cuda_kernel_failure_is_fatal_without_cpu_fallback(self):
        with self.fake_hardware() as hardware, \
                mock.patch("spacy.load", return_value=self.fake_ner), \
                mock.patch("spacy.require_cpu") as cpu:
            hardware.torch.arange.side_effect = RuntimeError("mock incompatible CUDA driver")
            with self.assertRaisesRegex(RuntimeError, "(?i)CUDA kernel check failed"):
                self.run_cli("gpu-check", "--device", "cuda", "--smoke-test")
        cpu.assert_not_called()

    def test_export_requires_the_corpus_manifest(self):
        self.write_test_data()
        (self.features / "corpus_manifest.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing completed corpus manifest"):
            self.annotations._feature_snapshot(self.root)
        self.assert_features_unchanged()

    def test_prepare_identity_rejects_any_source_or_code_change(self):
        current = {"stage": "prepare", "version": self.annotations.VERSION,
                   "sources": [{"path": "source", "bytes": 11, "mtime_ns": 22}],
                   "parameters": {"shards": 2, "batch_rows": 100, "arrow_batch_rows": 10},
                   "code_sha256": "current-code", "partition_backend_sha256": "frozen-backend",
                   "duckdb": "frozen-duckdb", "pyarrow": "frozen-pyarrow"}
        self.assertTrue(self.annotations._compatible_prepare_identity(current, current))
        mutations = [("unknown_code", lambda saved: saved.update(code_sha256="unapproved-code")),
                     ("changed_source", lambda saved: saved["sources"][0].update(bytes=12)),
                     ("changed_backend", lambda saved: saved.update(partition_backend_sha256="other")),
                     ("changed_version", lambda saved: saved.update(version=current["version"] + 1)),
                     ("changed_shards", lambda saved: saved["parameters"].update(shards=3)),
                     ("changed_duckdb", lambda saved: saved.update(duckdb="other")),
                     ("changed_arrow", lambda saved: saved.update(pyarrow="other"))]
        for name, mutate in mutations:
            with self.subTest(name=name):
                saved = json.loads(json.dumps(current))
                mutate(saved)
                self.assertFalse(self.annotations._compatible_prepare_identity(current, saved))

    def test_changed_prepare_code_is_rejected_without_metadata_rewrites(self):
        self.write_test_data()
        self.run_cli("prepare")
        pointer_path = self.work / "prepare.json"
        pointer = json.loads(pointer_path.read_text())
        workspace = Path(pointer["workspace"])
        incompatible = dict(pointer["identity"], code_sha256="changed-code")
        pointer["identity"] = incompatible
        pointer_path.write_text(json.dumps(pointer))
        complete_path = workspace / "complete.json"
        complete = json.loads(complete_path.read_text())
        complete["identity"] = incompatible
        complete_path.write_text(json.dumps(complete))
        (workspace / "identity.json").write_text(json.dumps(incompatible))
        before = {path: path.read_bytes() for path in self.work.rglob("*.json")}
        with self.assertRaisesRegex(RuntimeError, "code/shards changed"):
            self.annotations._prepared(self.args())
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assert_features_unchanged()


@unittest.skipUnless((GOOGLE_DIRECTORY / "slurm/run_google_v3_ner.slurm").is_file(),
                     "Optional local batch template is not distributed")
class AnnotationSlurmTests(unittest.TestCase):
    """Run the wrapper with mocked environment setup and Python execution."""

    def run_wrapper(self, **overrides):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "conda/bin").mkdir(parents=True)
            (root / "conda/bin/activate").write_text("")
            (root / "scripts").mkdir()
            (root / "scripts/google_v3_annotations.py").write_text("")
            environment = dict(os.environ)
            for name in ("SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "MODE",
                         "SHARD_INDEX", "NER_SHARDS", "DEPENDENCIES_ONLY",
                         "NER_DEVICE", "NER_GPU_ID", "CONDA_ENV", "GPU_SMOKE_TEST",
                         "CUDA_MODULE"):
                environment.pop(name, None)
            environment.update({"CONDA_ROOT": str(root / "conda"), "PYTHON_BIN": "python",
                                "SCRIPT_DIR": str(root / "scripts"),
                                "OUTPUT_ROOT": str(root / "output"),
                                "TEMP_DIRECTORY": str(root / "spill"),
                                "WORK_DIRECTORY": str(root / "work"),
                                "NER_MODEL": "test-model", "NER_SHARDS": "2",
                                **{name: str(value) for name, value in overrides.items()}})
            script = """
module() { printf 'mock-module:%s\\n' "$*" >&2; }
conda() { printf 'mock-conda:%s\\n' "$*" >&2; }
nvidia-smi() { printf 'mock-nvidia-smi:%s\\n' "$*" >&2; return "${MOCK_NVIDIA_SMI_STATUS:-0}"; }
python() { printf '%s\\0' "$@"; printf '\\n'; }
source "$1"
"""
            completed = subprocess.run(
                ["bash", "-c", script, "wrapper-test",
                 str(GOOGLE_DIRECTORY / "slurm/run_google_v3_ner.slurm")],
                env=environment, text=True, capture_output=True, check=False,
                timeout=30)
            calls = [line.rstrip("\0").split("\0")
                     for line in completed.stdout.splitlines() if line]
            return completed, calls

    def test_default_prepare_and_common_resource_options(self):
        result, calls = self.run_wrapper(SLURM_CPUS_PER_TASK=2,
                                         DUCKDB_MEMORY_LIMIT="8GB")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "prepare")
        arguments = dict(zip(calls[0][3::2], calls[0][4::2]))
        self.assertEqual(arguments["--threads"], "2")
        self.assertEqual(arguments["--memory-limit"], "8GB")
        self.assertEqual(arguments["--model"], "test-model")
        self.assertEqual(arguments["--shards"], "2")
        self.assertEqual(arguments["--device"], "cpu")
        self.assertEqual(arguments["--gpu-id"], "0")

    def test_array_index_takes_precedence_over_explicit_shard(self):
        result, calls = self.run_wrapper(MODE="predict", SLURM_ARRAY_TASK_ID=1,
                                         SHARD_INDEX=0, NER_BATCH_SIZE=7,
                                         PREDICTION_BATCH_ROWS=9)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = dict(zip(calls[0][3::2], calls[0][4::2]))
        self.assertEqual(calls[0][2], "predict")
        self.assertEqual(arguments["--shard-index"], "1")
        self.assertEqual(arguments["--ner-batch-size"], "7")
        self.assertEqual(arguments["--prediction-batch-rows"], "9")

    def test_dependencies_only_skips_ner_aggregation(self):
        result, calls = self.run_wrapper(MODE="finalize", DEPENDENCIES_ONLY=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "export")
        self.assertEqual(calls[0][-1], "--dependencies-only")

    def test_all_runs_ordered_prepare_shards_aggregate_and_export(self):
        result, calls = self.run_wrapper(MODE="all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[2] for call in calls],
                         ["prepare", "predict", "predict", "aggregate", "export"])
        shards = [dict(zip(call[3::2], call[4::2]))["--shard-index"]
                  for call in calls if call[2] == "predict"]
        self.assertEqual(shards, ["0", "1"])

    def test_array_prepare_is_rejected_before_any_pipeline_execution(self):
        result, calls = self.run_wrapper(MODE="prepare", SLURM_ARRAY_TASK_ID=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Only MODE=predict", result.stderr)
        self.assertEqual(calls, [])

    def test_cuda_uses_activated_environment_and_passes_device_to_every_stage(self):
        result, calls = self.run_wrapper(MODE="all", NER_DEVICE="cuda")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("mock-conda:", result.stderr)
        self.assertEqual([call[2] for call in calls],
                         ["prepare", "predict", "predict", "aggregate", "export"])
        for call in calls:
            arguments = dict(zip(call[3::2], call[4::2]))
            self.assertEqual(arguments["--device"], "cuda")
            self.assertEqual(arguments["--gpu-id"], "0")

    def test_cuda_explicit_environment_and_gpu_id_overrides_are_preserved(self):
        result, calls = self.run_wrapper(MODE="predict", NER_DEVICE="cuda",
                                         CONDA_ENV="custom-gpu-env", NER_GPU_ID=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("mock-conda:", result.stderr)
        arguments = dict(zip(calls[0][3::2], calls[0][4::2]))
        self.assertEqual(arguments["--device"], "cuda")
        self.assertEqual(arguments["--gpu-id"], "1")

    def test_check_mode_passes_gpu_smoke_test_without_running_pipeline_stages(self):
        result, calls = self.run_wrapper(MODE="check", NER_DEVICE="cuda",
                                         GPU_SMOKE_TEST=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "gpu-check")
        self.assertEqual(calls[0][-1], "--smoke-test")
        arguments = dict(zip(calls[0][3:-1:2], calls[0][4:-1:2]))
        self.assertEqual(arguments["--device"], "cuda")
        self.assertEqual(arguments["--gpu-id"], "0")

    def test_invalid_device_is_rejected_before_environment_activation(self):
        result, calls = self.run_wrapper(NER_DEVICE="invalid-device")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown NER_DEVICE", result.stderr)
        self.assertNotIn("mock-conda:", result.stderr)
        self.assertEqual(calls, [])

    def test_cuda_module_load_is_limited_to_gpu_using_modes(self):
        for mode in ("predict", "check", "all", "prepare", "finalize"):
            with self.subTest(mode=mode):
                result, unused_calls = self.run_wrapper(MODE=mode, NER_DEVICE="cuda", CUDA_MODULE="cuda/test")
                self.assertEqual(result.returncode, 0, result.stderr)
                if mode in ("predict", "check", "all"):
                    self.assertIn("mock-module:load cuda/test", result.stderr)
                else:
                    self.assertNotIn("mock-module:load cuda/test", result.stderr)

    def test_cuda_module_can_be_overridden_or_explicitly_disabled(self):
        result, unused_calls = self.run_wrapper(MODE="predict", NER_DEVICE="cuda",
                                                CUDA_MODULE="site-cuda/12.3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mock-module:load site-cuda/12.3", result.stderr)
        result, unused_calls = self.run_wrapper(MODE="predict", NER_DEVICE="cuda",
                                                CUDA_MODULE="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("mock-module:load", result.stderr)

    def test_failed_nvidia_diagnostic_still_runs_python_cuda_preflight(self):
        result, calls = self.run_wrapper(MODE="check", NER_DEVICE="cuda",
                                         MOCK_NVIDIA_SMI_STATUS=7)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nvidia-smi diagnostic failed", result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "gpu-check")


if __name__ == "__main__":
    unittest.main()
