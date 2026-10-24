"""Bounded missing-key lemmatization without changing frequencies/vocabulary."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "google_v3_pipeline.py"
SPEC = importlib.util.spec_from_file_location("google_v3_pipeline", str(MODULE_PATH))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class LemmaCoverageTests(unittest.TestCase):
    def setUp(self):
        try:
            import spacy
            import duckdb
            import pyarrow.parquet as pq
            from spacy.lookups import Lookups
        except ImportError:
            self.skipTest("spaCy, DuckDB and PyArrow required")
        self.spacy, self.duckdb, self.pq = spacy, duckdb, pq
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        (self.model / "vocab").mkdir(parents=True)
        (self.model / "meta.json").write_text(json.dumps({"name": "test-rules"}))
        self.lookups = Lookups()
        self.lookups.add_table("lemma_rules", {"noun": [["s", ""]], "verb": [["ed", ""]]})
        self.lookups.add_table("lemma_exc", {
            "noun": {"axes": ["axis"]}, "verb": {"axes": ["axe"], "went": ["go"]}})
        self.lookups.add_table("lemma_index", {"noun": ["cat", "dog", "axis"],
                                                "verb": ["go", "axe", "walk"]})
        directory = self.model / "lemmatizer" / "lookups"
        directory.mkdir(parents=True)
        self.lookups.to_disk(directory)
        self.reduced = self.root / "reduced"
        self.reduced.mkdir()
        self.lexicon = self.write("lexicon", [{
            "surface": "dog", "norm": "dog", "pos": "NOUN", "lemma": "dog", "count": 7}])
        self.rule_key = "lemmatizer/lookups/lookups.bin"
        self.sidecar = Path(str(self.lexicon) + ".json")
        self.sidecar.write_text(json.dumps({
            "model": str(self.model), "spacy_version": spacy.__version__,
            "source_info": {"model": {"name": str(self.model), "files_sha256": {
                self.rule_key: pipeline.file_sha256(self.model / self.rule_key)}}}}))

    def write(self, name, rows):
        path = self.reduced / f"{name}.parquet"
        sink = pipeline.ParquetSink(path, name, 2)
        for row in rows:
            sink.append(row)
        sink.close()
        return path

    def test_google_pos_and_lexical_marker_use_rules_without_prediction(self):
        coverage = pipeline._ExportLemmatizer(self.lexicon)
        self.assertEqual(coverage.lemma("Axes", "NOUN"), "axis")
        self.assertEqual(coverage.lemma("Axes", "VERB"), "axe")
        self.assertEqual(coverage.lemma("went", "VERB"), "go")
        for pos in ["NOUN", "VERB", "PRT"]:
            self.assertEqual(coverage.lemma("<s=>", pos), "<s=>")
        self.assertEqual(coverage.lemma("ＯＮ", "PRT"), "on")
        with self.assertRaisesRegex(RuntimeError, "unsupported Google POS"):
            coverage.lemma("dog", "INVALID")
        with self.assertRaisesRegex(RuntimeError, "null form/POS"):
            coverage.lemma(None, "NOUN")

    def test_cache_and_spacy_string_store_are_bounded(self):
        coverage = pipeline._ExportLemmatizer(self.lexicon, cache_size=2, reset_after=2)
        coverage.lemma("dogs", "NOUN")
        first_vocab = coverage.vocab
        coverage.lemma("cats", "NOUN")
        self.assertEqual(coverage.lemma("DOGS", "NOUN"), "dog")
        self.assertEqual(coverage.cache_hits, 1)
        coverage.lemma("axes", "NOUN")
        self.assertIsNot(coverage.vocab, first_vocab)
        self.assertEqual(len(coverage.cache), 2)
        self.assertEqual(coverage.computed, 3)

    def test_mismatched_rules_fail_before_export(self):
        saved = json.loads(self.sidecar.read_text())
        saved["source_info"]["model"]["files_sha256"][self.rule_key] = "wrong-hash"
        self.sidecar.write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, "rules differ"):
            pipeline._ExportLemmatizer(self.lexicon)

    def test_mismatched_spacy_version_fails(self):
        saved = json.loads(self.sidecar.read_text())
        saved["spacy_version"] = "different-version"
        self.sidecar.write_text(json.dumps(saved))
        with self.assertRaisesRegex(RuntimeError, "spaCy version differs"):
            pipeline._ExportLemmatizer(self.lexicon)

    def test_no_model_fails_only_for_missing_joins(self):
        self.sidecar.unlink()
        coverage = pipeline._ExportLemmatizer(self.lexicon)
        with self.duckdb.connect() as connection:
            coverage.register(connection)
            # Known keys must never invoke Python/rules, even with no model.
            self.assertEqual(connection.execute("SELECT COALESCE('dog', "
                "google_rule_lemma('dogs', 'NOUN'))").fetchone()[0], "dog")
            with self.assertRaisesRegex(Exception, "Supply --lemma-model"):
                connection.execute("SELECT google_rule_lemma('dogs','NOUN')").fetchall()
        explicit = pipeline._ExportLemmatizer(self.lexicon, model=str(self.model))
        self.assertEqual(explicit.lemma("dogs", "NOUN"), "dog")

    def test_full_export_higher_order_and_stale_unigram_keys_preserves_counts(self):
        count = {"decade": 1900, "count": 3}
        pair = {"modifier_surface": "Cats", "modifier_pos": "NOUN",
                "head_surface": "Dogs", "head_pos": "NOUN"}
        norm_pair = {"modifier_norm": "cats", "modifier_pos": "NOUN",
                     "head_norm": "dogs", "head_pos": "NOUN"}
        self.write("bigrams", [{**pair, **count}])
        self.write("contexts", [{**norm_pair, "context_norm": "went", "context_pos": "VERB",
                                 "side": "R", "distance": 1, **count}])
        self.write("word_contexts", [{"word_norm": "axes", "word_pos": "NOUN",
                                      "context_norm": "went", "context_pos": "VERB",
                                      "side": "R", "distance": 1, **count}])
        self.write("unigrams", [{"surface": "dog", "pos": "NOUN", **count},
                                 {"surface": "CATS", "pos": "NOUN", **count},
                                 {"surface": "cat", "pos": "NOUN", "decade": 1900, "count": 2}])
        self.write("dependencies", [{**pair, "direction": "head=>dependent", **count}])
        (self.reduced / "reduction_manifest.json").write_text(json.dumps({"source_info": {
            "stage": "reduce", "status": "complete", "corpus": pipeline.corpus_record(),
            "parameters": {"context_vocabulary_sha256": "unchanged", "context_vocabulary": None},
            "time": pipeline.time_record(1900, 1909, 10)}}))
        before = {path: path.read_bytes() for path in self.reduced.iterdir()}
        args = SimpleNamespace(output=str(self.root), feature_output=None, lexicon=None,
            allow_partial=False, threads=1, memory_limit="256MB", temp_directory=str(self.root / "tmp"),
            include_dependencies=True, partitions=2, partition_batch_rows=2, arrow_batch_rows=2)
        pipeline.make_feature_inputs(args)
        final = self.root / "final"
        self.assertEqual(self.pq.read_table(final / "bigrams.parquet").to_pylist(), [{
            "modifier": "cat_NOUN", "head": "dog_NOUN", "time": 1900,
            "context": "go_VERB", "count": 3}])
        counts_path = self.root / "extracted/feature_inputs/local5/bigram_counts.parquet"
        self.assertEqual(self.pq.read_table(counts_path).to_pylist(), [{
            "modifier": "cat_NOUN", "head": "dog_NOUN", "time": 1900, "count": 3}])
        self.assertEqual(self.pq.read_table(final / "unigrams.parquet").to_pylist(), [{
            "word": "axis_NOUN", "time": 1900, "context": "go_VERB", "count": 3}])
        unigram_counts = self.pq.read_table(final / "unigram_counts.parquet").to_pylist()
        self.assertEqual(sorted(unigram_counts, key=lambda row: row["word"]), [
            {"word": "cat_NOUN", "time": 1900, "count": 5},
            {"word": "dog_NOUN", "time": 1900, "count": 3}])
        annotation = self.pq.read_table(final / "bigram_annotations.parquet").to_pylist()[0]
        self.assertEqual((annotation["modifier"], annotation["head"]), ("cat_NOUN", "dog_NOUN"))
        self.assertEqual((annotation["modifier_surface"], annotation["head_surface"]), ("Cats", "Dogs"))
        dependencies = self.pq.read_table(self.root / "extracted/tables/dependencies.parquet").to_pylist()
        self.assertEqual(dependencies[0]["modifier"], "cat_NOUN")
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
        manifest_path = self.root / "extracted/feature_inputs/local5/corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        parameters = manifest["source_info"]["parameters"]
        self.assertEqual(parameters["context_vocabulary_sha256"], "unchanged")
        self.assertEqual(parameters["lemma_coverage"]["model"]["files_sha256"][self.rule_key],
                         pipeline.file_sha256(self.model / self.rule_key))
        pipeline.make_feature_inputs(args)
        resumed = json.loads(manifest_path.read_text())
        self.assertTrue(all(resumed["source_info"]["statistics"]["tables"][name]["reused"]
                            for name in ["bigrams", "bigram_counts", "unigrams", "unigram_counts"]))


if __name__ == "__main__":
    unittest.main()
