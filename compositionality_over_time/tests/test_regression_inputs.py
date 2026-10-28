"""Verified, immutable CSV snapshots of small synthetic Google feature grids.

The rating roster is read-only; generated features, manifests and
snapshots are temporary test data. No corpus scan, R fitting, or Slurm submission is needed.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


GOOGLE_DIRECTORY = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = GOOGLE_DIRECTORY / "feature_helpers"
RATINGS_DIRECTORY = GOOGLE_DIRECTORY / "tests" / "test_data" / "ratings"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.append(str(SOURCE_DIRECTORY))
if str(GOOGLE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GOOGLE_DIRECTORY))


class RegressionInputTestData(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import pandas as pd
            import pyarrow
        except ImportError:
            self.skipTest("NumPy, Pandas, and PyArrow are required")
        self.np, self.pd = np, pd
        path = GOOGLE_DIRECTORY / "regression_inputs.py"
        if not path.is_file():
            self.skipTest("Google regression snapshot adapter is not installed")
        spec = importlib.util.spec_from_file_location("regression_inputs", path)
        self.adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.adapter
        spec.loader.exec_module(self.adapter)
        from coha_feature_pipeline import read_ratings
        self.read_ratings = read_ratings
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        ratings_environment = mock.patch.dict("os.environ", {"RATINGS_DIR": str(RATINGS_DIRECTORY)})
        ratings_environment.start()
        self.addCleanup(ratings_environment.stop)
        self.features_root = self.root / "features"
        self.mode = self.features_root / "local5"
        self.mode.mkdir(parents=True)
        self.snapshot_root = self.root / "snapshot"
        self.manifests = {}
        self.frames = {}
        self.context_hash = hashlib.sha256(b"one frozen lemma_POS vocabulary").hexdigest()
        self.make_grid()

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def artifact(self, path):
        path = Path(path)
        return {"name": path.name, "path": str(path.resolve()),
                "bytes": path.stat().st_size, "sha256": self.digest(path)}

    def ratings(self, tagging):
        args = SimpleNamespace(tag=tagging == "Tagged", **{
            name: RATINGS_DIRECTORY / filename for name, filename in (
                ("reddy90", "reddy_90.txt"), ("cordeiro90", "cordeiro_90.txt"),
                ("cordeiro100", "cordeiro_100.txt"))})
        return self.read_ratings(args)

    def make_grid(self):
        for tagging in ("Tagged", "UnTagged"):
            ratings = self.ratings(tagging)
            for width in (10000, 50):
                times = (0,) if width == 10000 else (1800, 1850)
                for kind in (("features",) if width == 10000 else ("features", "temporal")):
                    frame = ratings.copy()
                    for time in times:
                        if kind == "features":
                            frame[f"sim_cpf_beta:{time}"] = self.np.float64("0.12345678901234567")
                            frame[f"beta:{time}"] = 0.5
                            frame[f"comp_freq:{time}"] = self.np.arange(len(frame), dtype=float) + 1
                            frame[f"eligible:{time}"] = 1
                            frame.loc[frame.index[0], f"sim_cpf_beta:{time}"] = self.np.nan
                        else:
                            for feature in ("change_comp", "change_mod", "change_head"):
                                frame[f"{feature}:{time}"] = 0.3141592653589793
                                frame.loc[frame.index[0], f"{feature}:{time}"] = self.np.nan
                    csv_name = self.adapter.feature_filename(
                        kind, "Agnostic", "PPMI", tagging, 5000, width, 0)
                    source = self.mode / Path(csv_name).with_suffix(".parquet").name
                    frame.to_parquet(source, index=False)
                    self.frames[source.name] = frame
                    key = (kind, tagging, width)
                    parameters = {
                        "context_size": 5000, "temporal": width, "cutoffs": [0],
                        "weights": ["PPMI"], "families": ["features", "temporal"],
                        "conditions": ["CompoundAgnostic"], "format": "parquet",
                        "tagged_targets": tagging == "Tagged", "ppmi": True,
                        "feature_definition_version": 3, "beta_policy": "clipped_unit_interval",
                        "compound_population": ("adj_noun_and_noun_noun" if tagging == "Tagged"
                                                else "all_lexical_bigrams"),
                        "context_mode": "local5", "frequency_source": "independent raw occurrence counts",
                        "context_vocabulary": {"requested_size": 5000, "actual_size": 5000,
                                               "sha256": self.context_hash,
                                               "fixed_across_time_bins_and_settings": True}}
                    manifest = {
                        "schema_version": 3, **parameters,
                        "prepare_fingerprint": hashlib.sha256(tagging.encode()).hexdigest(),
                        "source_info": {
                            "schema_version": 1, "stage": kind, "status": "complete",
                            "corpus": {"name": "Google Books Ngrams", "release": "20200217"},
                            "inputs": [], "parameters": parameters,
                            "time": {"column": "time", "bin_sizes": [width]},
                            "software": {"python": sys.version.split()[0], "packages": {}},
                            "outputs": [self.artifact(source)], "statistics": {}, "execution": {}}}
                    manifest_path = self.mode / f"{kind}_PPMI_{tagging}_ctx5000_{width}_manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    self.manifests[key] = manifest_path

    def args(self, *extra):
        values = ["snapshot", "--features-root", str(self.features_root),
                  "--snapshot-root", str(self.snapshot_root), "--vocabs", "5000",
                  "--weightings", "PPMI", "--settings", "Agnostic",
                  "--taggings", "Tagged", "UnTagged", "--timespans", "10000", "50",
                  "--cutoffs", "0", *extra]
        return self.adapter.parse_args(values)

    def snapshot(self, args=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.adapter.snapshot(args or self.args())

    def csv_files(self):
        return sorted((self.snapshot_root / "local5").glob("*.csv"))

    def mutate_manifest(self, key, change):
        path = self.manifests[key]
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def source(self, kind="features", tagging="UnTagged", width=50):
        name = self.adapter.feature_filename(kind, "Agnostic", "PPMI", tagging, 5000, width, 0)
        return self.mode / Path(name).with_suffix(".parquet").name

    def replace_source(self, kind, tagging, width, frame):
        path = self.source(kind, tagging, width)
        frame.to_parquet(path, index=False)
        self.mutate_manifest((kind, tagging, width),
                             lambda value: value["source_info"].update(outputs=[self.artifact(path)]))


class RegressionSnapshotTests(RegressionInputTestData):
    def test_snapshot_accepts_the_direct_feature_output_directory(self):
        direct = self.root / "analysis-feature-output"
        self.mode.rename(direct)
        self.features_root = direct
        state = self.snapshot()
        self.assertTrue(all(Path(record["source"]).parent == direct
                            for record in state["identity"]["files"].values()))
        self.adapter.verify_snapshot(self.snapshot_root)

    def test_coha_context_snapshots_preserve_identity_and_reject_cross_study_reuse(self):
        for mode in ("local5", "sentence"):
            with self.subTest(context_mode=mode):
                if self.mode.name != mode:
                    destination = self.features_root / mode
                    self.mode.rename(destination)
                    self.mode = destination
                self.snapshot_root = self.root / f"snapshot-{mode}"
                for key, previous in list(self.manifests.items()):
                    path = self.mode / previous.name
                    value = json.loads(path.read_text())
                    value.update(corpus="coha", context_mode=mode)
                    block = value["source_info"]
                    block["corpus"] = {"name": "COHA", "release": None}
                    block["parameters"]["corpus"] = "coha"
                    block["parameters"]["context_mode"] = mode
                    for record in block["outputs"]:
                        record["path"] = str(self.mode / Path(record["path"]).name)
                    path.write_text(json.dumps(value))
                    self.manifests[key] = path
                state = self.snapshot(self.args("--corpus", "coha", "--context-mode", mode))
                self.assertEqual(state["identity"]["corpus"], "coha")
                self.assertEqual(state["identity"]["context_mode"], mode)
                self.assertEqual(len(list((self.snapshot_root / mode).glob("*.csv"))), 6)
                self.adapter.verify_snapshot(self.snapshot_root, corpus="coha", context_mode=mode)
                with self.assertRaisesRegex(ValueError, "differs"):
                    self.adapter.verify_snapshot(self.snapshot_root, corpus="google")
                with self.assertRaisesRegex(ValueError, "differs"):
                    self.adapter.verify_snapshot(self.snapshot_root,
                        context_mode="sentence" if mode == "local5" else "local5")

    def test_snapshot_rejects_wrong_or_contradictory_corpus_manifests(self):
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            self.snapshot(self.args("--corpus", "coha"))
        self.mutate_manifest(("features", "Tagged", 10000),
                             lambda value: value.update(corpus="coha"))
        with self.assertRaisesRegex(ValueError, "contradictory"):
            self.snapshot()
        self.assertFalse((self.snapshot_root / "READY").exists())

    def test_filename_matches_latest_coha_r_specification(self):
        self.assertEqual(self.adapter.feature_filename(
            "features", "Aware", "RAW", "Tagged", 3000, 50, 10),
            "features_CompoundAware_withSetting_RAW_Tagged_ctx3000_50_10_na.csv")

    def test_snapshot_preserves_source_roster_missingness_and_float_precision(self):
        originals = {path: self.digest(path) for path in self.mode.glob("*.parquet")}
        self.snapshot()
        self.assertEqual(len(self.csv_files()), 6)
        self.assertTrue((self.snapshot_root / "READY").is_file())
        self.assertTrue((self.snapshot_root / "snapshot.json").is_file())
        for path in self.csv_files():
            expected = self.frames[path.with_suffix(".parquet").name].reset_index(drop=True)
            actual = self.pd.read_csv(path, na_values=["NA", ""], float_precision="round_trip")
            self.pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                              check_exact=True)
            self.assertFalse(any(name.startswith("Unnamed:") for name in actual))
        self.assertEqual(originals, {path: self.digest(path) for path in originals})
        for tagging, count in (("Tagged", 363), ("UnTagged", 287)):
            self.assertEqual(len(self.ratings(tagging)), count)

    def test_snapshot_records_canonical_pair_policy_without_changing_source(self):
        before = {path: self.digest(path) for path in self.manifests.values()}
        self.snapshot()
        manifest = json.loads((self.snapshot_root / "snapshot.json").read_text())
        self.assertEqual(manifest["compound_pair_policy"], {
            "Tagged": "adj_noun_or_noun_noun", "UnTagged": "all_lexical_bigrams"})
        self.assertEqual(manifest["identity"]["feature_definition_version"], 3)
        self.assertEqual(manifest["identity"]["beta_policy"], "clipped_unit_interval")
        self.assertEqual(before, {path: self.digest(path) for path in before})

    def test_repeat_snapshot_reuses_identical_csv_bytes_and_mtimes(self):
        self.snapshot()
        saved = {path: (self.digest(path), path.stat().st_mtime_ns) for path in self.csv_files()}
        self.snapshot()
        self.assertEqual(saved, {path: (self.digest(path), path.stat().st_mtime_ns) for path in saved})
        self.adapter.verify_snapshot(self.snapshot_root)

    def test_verify_refuses_changed_snapshot_csv(self):
        self.snapshot()
        path = self.csv_files()[0]
        path.write_bytes(path.read_bytes() + b"\nchanged\n")
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.verify_snapshot(self.snapshot_root)
        changed = path.read_bytes()
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()
        self.assertEqual(path.read_bytes(), changed)

    def test_rejects_historical_feature_definition(self):
        def historical(value):
            value["feature_definition_version"] = 1
            value["source_info"]["parameters"]["feature_definition_version"] = 1
        self.mutate_manifest(("features", "UnTagged", 50), historical)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()
        self.assertFalse((self.snapshot_root / "READY").exists())

    def test_rejects_unclipped_version_two_features(self):
        def incompatible(value):
            value["feature_definition_version"] = 2
            value["source_info"]["parameters"]["feature_definition_version"] = 2
            value.pop("beta_policy")
            value["source_info"]["parameters"].pop("beta_policy")
        self.mutate_manifest(("features", "UnTagged", 50), incompatible)
        with self.assertRaisesRegex(ValueError, "Incompatible/incomplete feature manifest"):
            self.snapshot()
        self.assertFalse(self.snapshot_root.exists())

    def test_rejects_missing_or_unclipped_beta_policy(self):
        key = ("features", "Tagged", 50)
        path = self.manifests[key]
        original = path.read_text()
        for policy in (None, "unclipped_analytic"):
            with self.subTest(policy=policy):
                value = json.loads(original)
                value["beta_policy"] = policy
                value["source_info"]["parameters"]["beta_policy"] = policy
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Incompatible/incomplete feature manifest"):
                    self.snapshot()
                self.assertFalse(self.snapshot_root.exists())

    def test_rejects_out_of_range_beta_despite_clipped_manifest(self):
        path = self.source("features", "Tagged", 50)
        original = self.pd.read_parquet(path)
        for beta in (-0.001, 1.001, self.np.inf):
            with self.subTest(beta=beta):
                frame = original.copy()
                frame.loc[0, "beta:1800"] = beta
                self.replace_source("features", "Tagged", 50, frame)
                with self.assertRaisesRegex(ValueError, "Unbounded beta"):
                    self.snapshot()
                self.assertFalse((self.snapshot_root / "READY").exists())
                # Conversion may have frozen earlier files before reaching the
                # invalid table; each independent source identity uses a new root.
                self.snapshot_root = self.root / f"snapshot-next-{beta}"

    def test_verify_rejects_incompatible_snapshot_despite_valid_receipt_hashes(self):
        self.snapshot()
        state_path = self.snapshot_root / "snapshot.json"
        ready_path = self.snapshot_root / "READY"
        original_state = json.loads(state_path.read_text())
        for version, policy in ((2, None), (3, "unclipped_analytic")):
            with self.subTest(version=version, policy=policy):
                state = json.loads(json.dumps(original_state))
                state["identity"]["feature_definition_version"] = version
                state["identity"]["beta_policy"] = policy
                state["fingerprint"] = self.adapter._digest(state["identity"])
                state_path.write_text(json.dumps(state), encoding="utf-8")
                ready_path.write_text(json.dumps({
                    "fingerprint": state["fingerprint"],
                    "snapshot_sha256": self.digest(state_path)}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Incompatible or incomplete regression snapshot"):
                    self.adapter.verify_snapshot(self.snapshot_root)

    def test_rejects_pos_restricted_untagged_population(self):
        def restricted(value):
            value["compound_population"] = "adj_noun_and_noun_noun"
            value["source_info"]["parameters"]["compound_population"] = value["compound_population"]
        self.mutate_manifest(("features", "UnTagged", 50), restricted)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_rejects_all_pos_tagged_population(self):
        def untagged(value):
            value["compound_population"] = "all_lexical_bigrams"
            value["source_info"]["parameters"]["compound_population"] = value["compound_population"]
        self.mutate_manifest(("features", "Tagged", 50), untagged)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_missing_manifest_is_not_an_unverified_conversion(self):
        self.manifests[("temporal", "Tagged", 50)].unlink()
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            self.snapshot()

    def test_incomplete_manifest_is_rejected(self):
        self.mutate_manifest(("features", "Tagged", 50),
                             lambda value: value["source_info"].update(status="partial"))
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_changed_parquet_hash_is_rejected_before_ready(self):
        path = self.source()
        frame = self.pd.read_parquet(path)
        frame.loc[0, "comp_freq:1800"] += 2
        frame.to_parquet(path, index=False)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()
        self.assertFalse((self.snapshot_root / "READY").exists())

    def test_preparation_identity_must_agree_across_time_bins(self):
        self.mutate_manifest(("features", "UnTagged", 50),
                             lambda value: value.update(prepare_fingerprint="a different preparation"))
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_frozen_vocabulary_identity_must_agree_across_tag_modes(self):
        def vocabulary(value):
            value["context_vocabulary"]["sha256"] = "f" * 64
            value["source_info"]["parameters"]["context_vocabulary"]["sha256"] = "f" * 64
        self.mutate_manifest(("features", "UnTagged", 50), vocabulary)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_duplicate_rating_rows_are_rejected_even_with_updated_hash(self):
        frame = self.pd.read_parquet(self.source())
        self.replace_source("features", "UnTagged", 50,
                            self.pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_missing_original_rating_is_not_silently_dropped(self):
        frame = self.pd.read_parquet(self.source())
        self.replace_source("features", "UnTagged", 50, frame.iloc[1:].copy())
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_missing_modified_rating_is_not_rewritten_or_imputed(self):
        frame = self.pd.read_parquet(self.source())
        self.assertTrue((frame.compound == "baby_blue").any())
        self.replace_source("features", "UnTagged", 50,
                            frame.loc[frame.compound != "baby_blue"].copy())
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_static_temporal_rating_values_must_match(self):
        frame = self.pd.read_parquet(self.source("temporal", "UnTagged", 50))
        frame.loc[0, "avgHead"] += 0.1
        self.replace_source("temporal", "UnTagged", 50, frame)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()

    def test_row_order_can_differ_without_changing_rating_identity(self):
        frame = self.pd.read_parquet(self.source("temporal", "UnTagged", 50))
        frame = frame.sample(frac=1, random_state=291).reset_index(drop=True)
        self.replace_source("temporal", "UnTagged", 50, frame)
        self.snapshot()
        self.adapter.verify_snapshot(self.snapshot_root)

    def test_existing_snapshot_cannot_mix_a_changed_source_generation(self):
        self.snapshot()
        preserved = {path: self.digest(path) for path in self.csv_files()}
        frame = self.pd.read_parquet(self.source())
        frame.loc[0, "comp_freq:1800"] += 2
        self.replace_source("features", "UnTagged", 50, frame)
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot()
        self.assertEqual(preserved, {path: self.digest(path) for path in preserved})

    def test_existing_snapshot_cannot_extend_its_grid_implicitly(self):
        self.snapshot()
        with self.assertRaises((ValueError, RuntimeError)):
            self.snapshot(self.args("--cutoffs", "0", "10"))

    def test_source_change_during_csv_publication_is_detected(self):
        source = self.source()
        original = self.pd.DataFrame.to_csv
        changed = False
        def change_during_write(frame, *args, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                source.write_bytes(source.read_bytes() + b"source changed during conversion")
            return original(frame, *args, **kwargs)
        with mock.patch.object(self.pd.DataFrame, "to_csv", change_during_write):
            with self.assertRaises((ValueError, RuntimeError)):
                self.snapshot()
        self.assertTrue(changed)
        self.assertFalse((self.snapshot_root / "READY").exists())

    def test_cli_grid_lists_accept_comma_separation_and_deduplicate(self):
        args = self.adapter.parse_args([
            "snapshot", "--features-root", str(self.features_root),
            "--snapshot-root", str(self.snapshot_root), "--vocabs", "5000,5000",
            "--weightings", "PPMI,PPMI", "--settings", "Agnostic,Agnostic",
            "--taggings", "Tagged,UnTagged", "--timespans", "10000,50,50",
            "--cutoffs", "0,0"])
        self.snapshot(args)
        self.assertEqual(len(self.csv_files()), 6)

    def test_interrupted_conversion_resumes_without_rewriting_completed_csvs(self):
        original = self.pd.DataFrame.to_csv
        writes = 0
        def interrupt_third_write(frame, *args, **kwargs):
            nonlocal writes
            writes += 1
            if writes == 3:
                raise RuntimeError("synthetic interrupted conversion")
            return original(frame, *args, **kwargs)
        with mock.patch.object(self.pd.DataFrame, "to_csv", interrupt_third_write):
            with self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
                self.snapshot()
        self.assertFalse((self.snapshot_root / "READY").exists())
        completed = {path: (self.digest(path), path.stat().st_mtime_ns)
                     for path in self.csv_files()}
        self.assertEqual(len(completed), 2)
        self.snapshot()
        self.assertEqual(len(self.csv_files()), 6)
        self.assertEqual(completed, {path: (self.digest(path), path.stat().st_mtime_ns)
                                     for path in completed})
        self.adapter.verify_snapshot(self.snapshot_root)

    def test_same_size_source_change_during_publication_is_detected(self):
        # Mutate the FIRST source after it has been loaded: no subsequent source
        # read accidentally exposes the corruption before the final race guard.
        source = self.source("features", "Tagged", 10000)
        size = source.stat().st_size
        original = self.pd.DataFrame.to_csv
        changed = False
        def same_size_change(frame, *args, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                data = bytearray(source.read_bytes())
                data[20] ^= 1
                source.write_bytes(data)
            return original(frame, *args, **kwargs)
        with mock.patch.object(self.pd.DataFrame, "to_csv", same_size_change):
            with self.assertRaises((ValueError, RuntimeError)):
                self.snapshot()
        self.assertTrue(changed)
        self.assertEqual(source.stat().st_size, size)
        self.assertFalse((self.snapshot_root / "READY").exists())

    def test_selected_verification_checks_requested_files_not_other_grid_cells(self):
        self.snapshot()
        selected = self.adapter.feature_filename("features", "Agnostic", "PPMI",
                                                 "Tagged", 5000, 10000, 0)
        unused = next(path for path in self.csv_files() if path.name != selected)
        unused.write_bytes(unused.read_bytes() + b"changed unrelated cell")
        self.adapter.verify_snapshot(self.snapshot_root, selected_names=[selected])
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.verify_snapshot(self.snapshot_root)
        with self.assertRaises((ValueError, RuntimeError)):
            self.adapter.verify_snapshot(self.snapshot_root, selected_names=["not-a-grid-file.csv"])

    def test_main_verification_accepts_an_explicit_grid_subset(self):
        self.snapshot()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.adapter.main(["verify", "--snapshot-root", str(self.snapshot_root),
                               "--taggings", "Tagged", "--timespans", "10000"])
        self.assertIn("Verified 1", output.getvalue())

    def test_dry_run_has_no_snapshot_writes(self):
        self.snapshot(self.args("--dry-run"))
        self.assertFalse(self.snapshot_root.exists())

    def test_invalid_grid_is_rejected_before_snapshot_writes(self):
        for option, value in (("--vocabs", "0"), ("--timespans", "0"),
                              ("--cutoffs", "-1"), ("--settings", "Unknown"),
                              ("--taggings", "NoTags"), ("--weightings", "TFIDF")):
            with self.subTest(option=option):
                with self.assertRaises((ValueError, RuntimeError)):
                    self.snapshot(self.args(option, value))
                self.assertFalse(self.snapshot_root.exists())

    def test_contradictory_scientific_source_info_is_rejected(self):
        key = ("features", "UnTagged", 50)
        path = self.manifests[key]
        original = path.read_text()
        contradictions = {
            "feature_definition_version": 1,
            "beta_policy": "unclipped_analytic",
            "compound_population": "adj_noun_and_noun_noun",
            "tagged_targets": True,
            "ppmi": False,
            "context_mode": "sentence",
            "context_size": 3000,
            "temporal": 10,
            "context_vocabulary": {"sha256": "a different vocabulary"}}
        for name, value in contradictions.items():
            with self.subTest(field=name):
                manifest = json.loads(original)
                manifest["source_info"]["parameters"][name] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Contradictory scientific source information"):
                    self.snapshot()
                self.assertFalse(self.snapshot_root.exists())
        path.write_text(original, encoding="utf-8")

    def test_ready_reuse_rejects_same_size_source_rewrite_with_unchanged_manifest(self):
        self.snapshot()
        preserved = {path: (self.digest(path), path.stat().st_mtime_ns)
                     for path in self.csv_files()}
        source = self.source("features", "Tagged", 10000)
        old_size = source.stat().st_size
        data = bytearray(source.read_bytes())
        data[20] ^= 1
        source.write_bytes(data)
        self.assertEqual(source.stat().st_size, old_size)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.snapshot()
        self.assertEqual(preserved, {path: (self.digest(path), path.stat().st_mtime_ns)
                                     for path in preserved})
        # A verified frozen snapshot remains independently usable even if live
        # source feature files are gone, changing, or corrupt.
        self.adapter.verify_snapshot(self.snapshot_root)

    def test_final_verification_failure_clears_ready_then_resume_reuses_csvs(self):
        with mock.patch.object(self.adapter, "verify_snapshot",
                               side_effect=RuntimeError("synthetic final verification failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic final verification"):
                self.snapshot()
        self.assertFalse((self.snapshot_root / "READY").exists())
        saved = {path: (self.digest(path), path.stat().st_mtime_ns)
                 for path in self.csv_files()}
        self.assertEqual(len(saved), 6)
        self.snapshot()
        self.assertTrue((self.snapshot_root / "READY").is_file())
        self.assertEqual(saved, {path: (self.digest(path), path.stat().st_mtime_ns)
                                for path in saved})


if __name__ == "__main__":
    unittest.main()
