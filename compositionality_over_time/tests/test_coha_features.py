"""COHA extraction, fixed vocabulary, and bounded feature compatibility."""

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import test_features as feature_tests
from test_features import FeatureTestData, GOOGLE_DIRECTORY


class CohaFeatureTests(FeatureTestData):
    compare_configuration = feature_tests.GoogleFeatureParityTests.compare_configuration
    long = feature_tests.GoogleFeatureParityTests.long

    def change_manifest(self, **updates):
        path = self.input / "corpus_manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(updates)
        path.write_text(json.dumps(manifest))
        return manifest

    def test_corpus_and_context_semantics_change_preparation_identity(self):
        args = self.args("prepare")
        original = self.google._specification(args)
        manifest = self.change_manifest()
        manifest["source_info"]["corpus"] = {"name": "COHA", "release": "synthetic"}
        self.change_manifest(source_info=manifest["source_info"])
        local = self.google._specification(args)
        self.change_manifest(context_mode="sentence")
        sentence = self.google._specification(args)
        self.assertEqual(len({item["fingerprint"] for item in (original, local, sentence)}), 3)
        self.assertEqual(sentence["identity"]["corpus_id"], "coha")
        self.assertEqual(sentence["identity"]["context_mode"], "sentence")

    def test_google_sentence_and_conflicting_context_modes_are_rejected(self):
        self.change_manifest(context_mode="sentence")
        with self.assertRaisesRegex(ValueError, "Google features support local5 only"):
            self.google._inputs(self.args("prepare"))
        self.change_manifest(context_mode="local5", mode="sentence")
        with self.assertRaisesRegex(ValueError, "Conflicting mode and context_mode"):
            self.google._inputs(self.args("prepare"))

    def test_shared_loader_uses_all_pos_only_for_untagged(self):
        tagged = self.shared.load_inputs(self.input, tagged=True)
        untagged = self.shared.load_inputs(self.input, tagged=False)
        self.assertFalse(tagged["compounds"].modifier.str.endswith("_VERB").any())
        selected = untagged["compound_counts"].loc[
            lambda frame: (frame.modifier == "red") & (frame["head"] == "car") & (frame.time == 1820)]
        self.assertEqual(selected["count"].item(), 116)
        contexts = untagged["compounds"].loc[
            lambda frame: (frame.modifier == "red") & (frame["head"] == "car") &
                          (frame.time == 1820) & (frame.context == "extra_ADJ")]
        self.assertEqual(contexts["count"].item(), 200)

    def test_cumulative_workflows_reject_coha_before_reading_study_files(self):
        prepared = self.root / "coha-preparation"
        prepared.mkdir()
        identity = {"corpus": {"name": "COHA"}, "context_mode": "local5"}
        (prepared / "identity.json").write_text(json.dumps(identity))
        (prepared / "complete.json").write_text(json.dumps(
            {"fingerprint": self.google._digest(identity), "files": []}))
        module = importlib.util.spec_from_file_location(
            "coha_cumulative_guard", GOOGLE_DIRECTORY / "cumulative_features.py")
        cumulative = importlib.util.module_from_spec(module)
        module.loader.exec_module(cumulative)
        with self.assertRaisesRegex(ValueError, "Cumulative workflows support Google local5"):
            cumulative.read_specification(SimpleNamespace(prepared_root=prepared))

    def test_canonical_coha_exports_and_vocab_feed_both_feature_modes(self):
        try:
            import spacy
            from spacy.tokens import Doc
        except ImportError:
            self.skipTest("spaCy is required for the COHA extraction integration")
        preprocessing = GOOGLE_DIRECTORY.parent / "coha_preprocessing"
        if str(preprocessing) not in sys.path:
            sys.path.insert(0, str(preprocessing))
        import coha_pipeline as pipeline
        from src.count_store import CountStore
        from vocabulary import build_vocabulary

        nlp = spacy.blank("en")
        store = CountStore(self.root / "counts.sqlite")
        self.addCleanup(store.close)
        for name, columns in pipeline.schemas().items():
            store.create(name, columns)
        examples = [
            (["red", "car", "sees", "the", "blue", "car", "near", "town", "."],
             ["ADJ", "NOUN", "VERB", "DET", "ADJ", "NOUN", "ADP", "NOUN", "PUNCT"]),
            (["red", "car", "moves", "through", "the", "blue", "town", "."],
             ["VERB", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN", "PUNCT"]),
        ]
        for time in (1820, 1830):
            for index, (words, pos) in enumerate(examples):
                doc = Doc(nlp.vocab, words=words, pos=pos, lemmas=words,
                          sent_starts=[True] + [False] * (len(words) - 1))
                record = {"doc_id": str(index), "document_uid": f"{time}-{index}",
                          "fragment_id": 0, "time": time, "domain": "fic",
                          "normalized_start": 0, "left_gap": False, "right_gap": False,
                          "left_gap_length": 0, "right_gap_length": 0}
                pipeline.add_document(store, doc, record, io.StringIO(), "map")

        for mode in ("local5", "sentence"):
            self.input = self.root / mode
            pipeline.canonical_inputs(store, self.input, mode, {})
            build_vocabulary(SimpleNamespace(input=self.input, output=self.vocabulary,
                size=0, threads=1, memory_limit="256MB", max_temp_directory_size="1GB",
                temp_directory=self.spill))
            self.input_bytes = {path: path.read_bytes() for path in self.input.glob("*.parquet")}
            unused_manifest, unused_paths, records = self.google._inputs(self.args("prepare"))
            self.assertTrue(all(record["sha256"] for record in records.values()))
            for tagged in (True, False):
                with self.subTest(mode=mode, tagged=tagged), contextlib.redirect_stdout(io.StringIO()):
                    self.compare_configuration(tagged=tagged, temporal=10, cutoffs=(0,),
                                               families=("features", "temporal"))
                    manifest = json.loads((self.output /
                        f"features_PPMI_{'Tagged' if tagged else 'UnTagged'}_ctx3_10_manifest.json").read_text())
                    self.assertEqual(manifest["corpus"], "coha")
                    self.assertEqual(manifest["context_mode"], mode)
                    self.assertEqual(manifest["source_info"]["corpus"]["name"], "COHA")
                    frame = self.long("features", "CompoundAware", tagged=tagged)
                    selected = frame.loc[(frame.modifier == ("red_ADJ" if tagged else "red")) &
                                         (frame["head"] == ("car_NOUN" if tagged else "car"))]
                    self.assertEqual(set(selected.comp_freq), {1 if tagged else 2})
            bigrams = self.pd.read_parquet(self.input / "bigrams.parquet")
            far_context = bigrams.loc[(bigrams.modifier == "red_ADJ") &
                                     (bigrams["head"] == "car_NOUN") &
                                     (bigrams.context == "town_NOUN")]
            self.assertEqual(far_context.empty, mode == "local5")
