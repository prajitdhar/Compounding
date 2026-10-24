"""Tiny immutable test inputs for the scoped cumulative-prefix extension."""

import contextlib
import importlib.util
import io
import json
import sys
from unittest import mock

import numpy as np
from scipy import sparse

import test_cumulative_features as static_test_data
from test_features import FeatureTestData, GOOGLE_DIRECTORY


class GoogleCumulativeTemporalTests(FeatureTestData):
    # Reuse only test data utilities, not the static driver's inherited tests.
    snapshot_files = static_test_data.GoogleCumulativeFeatureTests.snapshot_files
    assert_immutable_inputs = static_test_data.GoogleCumulativeFeatureTests.assert_immutable_inputs
    raw_oracle = static_test_data.GoogleCumulativeFeatureTests.raw_oracle
    assert_frames_equal = static_test_data.GoogleCumulativeFeatureTests.assert_frames_equal

    def setUp(self):
        super().setUp()
        for path in self.ratings.values():
            ratings = self.pd.read_csv(path, sep="\t")
            ratings["is_original"] = True
            ratings.to_csv(path, sep="\t", index=False)
        for name in ("cumulative_features", "cumulative_temporal"):
            spec = importlib.util.spec_from_file_location(name, GOOGLE_DIRECTORY / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(self, "cumulative" if name.endswith("features") else "temporal", module)
        self.modern = self.root / "modern"
        self.work = self.root / "cumulative-work"
        self.cumulative_output = self.root / "temporal-output"
        self.prepared = None
        self.specification = None
        self.production_guard = self.temporal._require_configuration
        patch = mock.patch.object(self.temporal, "_require_configuration", side_effect=self.validate_test_data_configuration)
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def validate_test_data_configuration(args):
        # Only a test may replace the production 30000/cutoff-500 guard.
        if (args.context_size != 3 or args.condition != "CompoundAware"
                or args.weight != "PPMI" or args.time_step != 10):
            raise ValueError("Invalid configuration for tiny test data")

    def setup_prepared(self):
        with contextlib.redirect_stdout(io.StringIO()):
            args = self.args("prepare", "--untagged")
            self.prepared = self.google.prepare(args)
            self.specification = self.google._specification(args)
            self.google.run_features(self.args(
                "features", "--untagged", "--context-size", "3", "--temporal", "10000",
                "--weights", "PPMI", "--families", "features", "--conditions", "CompoundAware",
                "--cutoffs", "0", "20", "20000"))
        local = self.modern / "local5"
        local.mkdir(parents=True)
        for path in self.output.glob("*.parquet"):
            self.pd.read_parquet(path).to_csv(local / path.with_suffix(".csv").name, index=False)
        self.immutable = self.snapshot_files(self.prepared)
        self.immutable.update(self.snapshot_files(self.modern))
        self.source_files = {path for path in self.input.rglob("*") if path.is_file()}
        self.source_files.update([self.vocabulary, *self.ratings.values()])
        self.immutable.update({path: (self.digest(path), path.stat().st_mtime_ns)
                               for path in self.source_files})

    def driver_args(self, command="features", *extra):
        return self.cumulative.parse_args([
            command, "--prepared-root", str(self.prepared or self.cache / "unprepared"),
            "--modern-root", str(self.modern), "--work-root", str(self.work),
            "--output-root", str(self.cumulative_output), "--temp-directory", str(self.spill),
            "--context-size", "3", "--condition", "CompoundAware", "--cutoff", "0",
            "--threads", "1", "--memory-limit", "256MB", *extra])

    def run_temporal(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            self.temporal.prepare(self.driver_args("prepare", *extra))
            return self.temporal.run_features(self.driver_args("features", *extra))

    def product_path(self, endpoint, *, cutoff=0, long=False, csv=False):
        return self.cumulative_output / "local5" / f"through_{endpoint}" / (
            "features_CompoundAware_withSetting_PPMI_UnTagged_ctx3_10000_"
            f"{cutoff}_na{'_long' if long else ''}.{'csv' if csv else 'parquet'}")

    def long(self, endpoint, *, cutoff=0):
        return self.pd.read_parquet(self.product_path(endpoint, cutoff=cutoff, long=True))

    @staticmethod
    def dense_ppmi(matrix, background):
        raw = matrix.toarray()
        denominator = raw.sum(axis=1, keepdims=True) * np.asarray(background)[None, :]
        result = np.zeros(raw.shape)
        mask = (raw > 0) & (denominator > 0)
        result[mask] = np.maximum(np.log2(raw[mask] * np.sum(background) / denominator[mask]), 0)
        return result

    def temporal_oracle(self, endpoint, cutoff=0):
        raw, bg, _, _, minimum, mod_ids, head_ids = self.raw_oracle(endpoint)
        result = self.specification["pairs"][["modifier", "head"]].copy()
        for column in self.temporal.TEMPORAL_COLUMNS:
            result[column] = np.nan
        if endpoint == 1820:
            return result
        previous, old_bg, _, _, old_minimum, _, _ = self.raw_oracle(endpoint - 10)
        eligible = (minimum > cutoff) & (old_minimum > cutoff)
        for column, code, ids in zip(self.temporal.TEMPORAL_COLUMNS,
                                     (2, 3, 4), (None, mod_ids, head_ids)):
            current = self.dense_ppmi(raw[code], bg[0])
            old = self.dense_ppmi(previous[code], old_bg[0])
            if ids is not None:
                current, old = current[ids], old[ids]
            norm = np.linalg.norm(current, axis=1) * np.linalg.norm(old, axis=1)
            values = np.full(len(norm), np.nan)
            valid = (norm > 0) & eligible
            values[valid] = np.sum(current[valid] * old[valid], axis=1) / norm[valid]
            result[column] = values
        return result

    def assert_temporal_oracle(self, endpoint, cutoff=0):
        frame = self.long(endpoint, cutoff=cutoff)
        columns = ["modifier", "head", *self.temporal.TEMPORAL_COLUMNS]
        expected = self.temporal_oracle(endpoint, cutoff).merge(
            frame[["modifier", "head"]], on=["modifier", "head"], validate="one_to_one")
        self.assert_frames_equal(frame[columns], expected[columns])

    def test_exact_production_defaults_and_guard_reject_any_other_configuration(self):
        values = ["plan", "--prepared-root", str(self.cache), "--modern-root", str(self.modern),
                  "--work-root", str(self.work), "--output-root", str(self.cumulative_output),
                  "--temp-directory", str(self.spill)]
        with mock.patch.object(self.temporal, "_require_configuration", side_effect=self.production_guard):
            args = self.temporal.parse_args(values)
            self.assertEqual((args.context_size, args.condition, args.weight, args.cutoff, args.time_step),
                             (30000, "CompoundAware", "PPMI", 500, 10))
            for option, value in (("--context-size", "5000"), ("--condition", "CompoundPooled"),
                                  ("--weight", "RAW"), ("--cutoff", "0"), ("--time-step", "20")):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises((ValueError, SystemExit)):
                        self.temporal.parse_args([*values, option, value])

    def test_cosines_are_similarities_and_strict_both_prefix_cutoffs_apply(self):
        current = {"minimum": np.array([501, 501, 501, 500, 501]),
                   "compound": sparse.csr_matrix([[1, 0], [0, 0], [1, 0], [1, 0], [1, 0]])}
        previous = {"minimum": np.array([501, 501, 501, 501, 500]),
                    "compound": sparse.csr_matrix([[1, 1], [1, 0], [0, 1], [1, 0], [1, 0]])}
        for role in ("modifier", "head"):
            current[role], previous[role] = current["compound"], previous["compound"]
        result = self.temporal._temporal_values(current, previous, 500)
        for column in self.temporal.TEMPORAL_COLUMNS:
            np.testing.assert_allclose(result[column], [1 / np.sqrt(2), np.nan, 0, np.nan, np.nan],
                                       equal_nan=True)
        self.assertTrue(self.temporal._temporal_values(current, None, 500).isna().all().all())

    def test_every_endpoint_matches_independent_prefix_weighting_and_static_parity(self):
        self.setup_prepared()
        self.run_temporal()
        for endpoint in (1820, 1830, 1840, 1850):
            with self.subTest(endpoint=endpoint):
                self.assert_temporal_oracle(endpoint)
        final = self.pd.read_csv(self.product_path(1850, csv=True))
        modern = self.pd.read_csv(self.modern / "local5" / self.product_path(1850, csv=True).name)
        self.temporal._static_parity(final, modern)
        setting = ["sim_bw_settings_modifier:0", "perc_token_mod:0", "perc_type_mod:0",
                   "sim_bw_settings_head:0", "perc_token_head:0", "perc_type_head:0"]
        self.assertTrue(set(setting).issubset(final.columns))
        self.assertFalse(final[setting].isna().all().all())
        self.assert_immutable_inputs()

    def test_count_cache_identity_and_files_are_identical_to_static_study(self):
        self.setup_prepared()
        args = self.driver_args("prepare")
        with contextlib.redirect_stdout(io.StringIO()):
            old_work = self.cumulative.prepare(args)
        before = self.snapshot_files(old_work)
        old_specification = self.cumulative.read_specification(args)
        new_specification = self.temporal.read_specification(args)
        self.assertEqual(new_specification["count_input_identity"], old_specification["input_identity"])
        self.assertEqual(new_specification["work"], old_work)
        self.assertNotEqual(new_specification["input_identity"], old_specification["input_identity"])
        with mock.patch.object(self.cumulative, "_first_seen_query", side_effect=AssertionError("cache rescanned")):
            self.run_temporal()
        self.assertEqual(self.snapshot_files(old_work), before)
        material = json.loads(new_specification["input_identity_material"])
        self.assertEqual(material["temporal_policy"], self.temporal.TEMPORAL_POLICY)
        self.assertEqual(material["full_input_first_decade"], 1820)
        self.assert_immutable_inputs()

    def test_thinned_endpoints_load_hidden_exact_lag_without_lookahead(self):
        self.setup_prepared()
        spans = []
        original = self.google._load_vectors

        def record_span(specification, start, width, size, args):
            spans.append((start, start + width))
            return original(specification, start, width, size, args)

        with mock.patch.object(self.google, "_load_vectors", side_effect=record_span):
            self.run_temporal("--end-decades", "1820", "1850")
        self.assertEqual(spans, [(0, 1830), (1830, 1850), (1850, 1860)])
        self.assert_temporal_oracle(1850)
        manifest = json.loads((self.product_path(1850).parent / "cumulative_manifest.json").read_text())
        self.assertEqual(manifest["lag_endpoint_decade"], 1840)
        self.assertFalse(self.product_path(1840).exists())
        self.assert_immutable_inputs()

    def test_lag_prefix_may_fall_in_a_decade_without_any_population_observations(self):
        manifest_path = self.input / "corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name in self.shared.FEATURE_TABLES:
            frame = self.pd.read_parquet(self.input / f"{name}.parquet")
            path = self.write_table(name, frame.loc[frame.time != 1830].to_dict("records"))
            manifest["tables"][name].update(bytes=path.stat().st_size, sha256=self.digest(path))
        manifest_path.write_text(json.dumps(manifest))
        self.input_bytes = {path: path.read_bytes() for path in self.input_bytes}
        self.setup_prepared()
        args = self.driver_args()
        specification = self.temporal.read_specification(args)
        self.assertEqual(specification["available"], [1820, 1840, 1850])
        self.assertEqual(self.temporal._lag_endpoint(specification, 1840), 1830)
        self.assertIsNone(self.temporal._lag_endpoint(specification, 1820))
        self.run_temporal()
        self.assert_temporal_oracle(1840)
        manifest = json.loads((self.product_path(1840).parent / "cumulative_manifest.json").read_text())
        self.assertEqual(manifest["lag_endpoint_decade"], 1830)
        self.assert_immutable_inputs()

    def test_future_vector_perturbations_do_not_change_already_published_prefixes(self):
        self.setup_prepared()
        original = self.google._load_vectors

        def perturb_future(specification, start, width, size, args):
            raw, background = original(specification, start, width, size, args)
            if start <= 1850 < start + width:
                perturbation = sparse.csr_matrix(([1000000.0], ([0], [0])), shape=raw[2].shape)
                raw[2] = raw[2] + perturbation
                background[0][0] += 1000000
            return raw, background

        # A deliberately corrupted future must fail the final frozen-reference
        # parity check, but cannot influence any earlier temporal prefix.
        with mock.patch.object(self.google, "_load_vectors", side_effect=perturb_future):
            with self.assertRaisesRegex(ValueError, "Final cumulative feature differs"):
                self.run_temporal()
        for endpoint in (1820, 1830, 1840):
            self.assert_temporal_oracle(endpoint)
        self.assertFalse((self.cumulative_output / "features_complete.json").exists())
        self.assert_immutable_inputs()

    def test_newly_eligible_prefix_has_no_temporal_values_until_lag_also_qualifies(self):
        self.setup_prepared()
        self.run_temporal("--cutoff", "20")
        frame = self.long(1840, cutoff=20)
        jump = frame.loc[(frame.modifier == "jump") & (frame["head"] == "rope")].iloc[0]
        self.assertEqual(jump.comp_freq, 35)
        self.assertTrue(jump[list(self.temporal.TEMPORAL_COLUMNS)].isna().all())
        following = self.long(1850, cutoff=20)
        jump = following.loc[(following.modifier == "jump") & (following["head"] == "rope")].iloc[0]
        self.assertTrue(np.isfinite(jump.change_comp))
        self.assert_temporal_oracle(1840, cutoff=20)
        self.assert_temporal_oracle(1850, cutoff=20)
        self.assert_immutable_inputs()

    def test_empty_endpoints_keep_all_three_columns_and_full_rating_roster(self):
        self.setup_prepared()
        self.run_temporal("--cutoff", "20000")
        for endpoint in (1820, 1830, 1840, 1850):
            long = self.long(endpoint, cutoff=20000)
            self.assertTrue(long.empty)
            self.assertTrue(set(self.temporal.TEMPORAL_COLUMNS).issubset(long.columns))
            wide = self.pd.read_parquet(self.product_path(endpoint, cutoff=20000))
            columns = [column + ":0" for column in self.temporal.TEMPORAL_COLUMNS]
            self.assertTrue(wide[columns].isna().all().all())
            self.assertEqual(len(wide), len(self.specification["ratings"]))
        self.assert_immutable_inputs()

    def test_interrupted_run_reuses_valid_endpoints_and_reconstructs_lag(self):
        self.setup_prepared()
        original = self.temporal._write_endpoint

        def interrupt(specification, endpoint, frame, args):
            result = original(specification, endpoint, frame, args)
            if endpoint == 1830:
                raise InterruptedError("synthetic interruption")
            return result

        with mock.patch.object(self.temporal, "_write_endpoint", side_effect=interrupt):
            with self.assertRaises(InterruptedError):
                self.run_temporal()
        before = {path: data for path, data in self.snapshot_files(self.cumulative_output).items()
                  if "through_1820" in str(path) or "through_1830" in str(path)}
        written = []

        def resume(specification, endpoint, frame, args):
            written.append(endpoint)
            return original(specification, endpoint, frame, args)

        with mock.patch.object(self.temporal, "_write_endpoint", side_effect=resume):
            self.run_temporal()
        self.assertEqual(written, [1840, 1850])
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns) for path in before}, before)
        self.assert_temporal_oracle(1840)
        self.assert_temporal_oracle(1850)
        all_before = self.snapshot_files(self.cumulative_output)
        with mock.patch.object(self.google, "_load_vectors", side_effect=AssertionError("completed work reread")):
            self.run_temporal()
        self.assertEqual(self.snapshot_files(self.cumulative_output), all_before)
        self.assert_immutable_inputs()

    def test_metadata_and_file_tampering_are_detected_and_compatible_repairs_work(self):
        self.setup_prepared()
        self.run_temporal()
        manifest_path = self.product_path(1840).parent / "cumulative_manifest.json"
        value = json.loads(manifest_path.read_text())
        value["lag_endpoint_decade"] = 1820
        manifest_path.write_text(json.dumps(value))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Invalid"):
            self.temporal.check(self.driver_args("check"))
        self.run_temporal()
        self.assertEqual(json.loads(manifest_path.read_text())["lag_endpoint_decade"], 1830)
        path = self.product_path(1850, csv=True)
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            self.temporal.check(self.driver_args("check"))
        self.run_temporal()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.temporal.check(self.driver_args("check"))["status"], "complete")
        self.assert_immutable_inputs()

    def test_failed_repair_removes_stale_completion_receipt(self):
        self.setup_prepared()
        self.run_temporal()
        path = self.product_path(1840, csv=True)
        path.write_bytes(path.read_bytes() + b"\n")
        with mock.patch.object(self.temporal, "_write_endpoint", side_effect=InterruptedError("repair failed")):
            with self.assertRaises(InterruptedError):
                self.run_temporal()
        self.assertFalse((self.cumulative_output / "features_complete.json").exists())

    def test_manifest_configuration_mirrors_cannot_be_re_adopted_after_tampering(self):
        self.setup_prepared()
        self.run_temporal()
        args = self.driver_args()
        specification = self.temporal.read_specification(args)
        path = self.product_path(1840).parent / "cumulative_manifest.json"
        original = json.loads(path.read_text())
        for field, value in (("condition", "CompoundPooled"), ("cutoff", 500),
                             ("ppmi", False), ("tagged", True), ("context_size", 4),
                             ("full_input_first_decade", 1810), ("schema_version", 2),
                             ("final_modern_parity_verified", True),
                             ("static_parity_excluded_columns", ["perc_type_head:0"])):
            with self.subTest(field=field):
                path.write_text(json.dumps({**original, field: value}))
                self.assertIsNone(self.temporal._valid_manifest(path, specification, 1840, args))
        path.write_text(json.dumps(original))
        self.run_temporal()
        self.assert_immutable_inputs()

    def test_force_does_not_overwrite_static_output_or_retarget_existing_endpoints(self):
        self.setup_prepared()
        with contextlib.redirect_stdout(io.StringIO()):
            self.cumulative.prepare(self.driver_args("prepare"))
            self.cumulative.run_features(self.driver_args())
        before = self.snapshot_files(self.cumulative_output)
        with self.assertRaisesRegex(RuntimeError, "incompatible identity"):
            self.run_temporal("--force")
        self.assertEqual(self.snapshot_files(self.cumulative_output), before)
        alternate = self.root / "separate-temporal-output"
        self.run_temporal("--output-root", str(alternate))
        before = self.snapshot_files(alternate)
        with self.assertRaisesRegex(RuntimeError, "different endpoints"):
            self.run_temporal("--output-root", str(alternate), "--end-decades", "1820", "1850", "--force")
        self.assertEqual(self.snapshot_files(alternate), before)
        self.assert_immutable_inputs()

    def test_final_parity_excludes_only_the_three_added_temporal_columns(self):
        self.setup_prepared()
        self.run_temporal()
        modern = self.pd.read_csv(self.modern / "local5" / self.product_path(1850, csv=True).name)
        frame = self.pd.read_csv(self.product_path(1850, csv=True))
        frame["change_comp:0"] = 0.234
        self.temporal._static_parity(frame, modern)
        frame["perc_token_head:0"] += 0.1
        with self.assertRaisesRegex(ValueError, "perc_token_head"):
            self.temporal._static_parity(frame, modern)
        with self.assertRaisesRegex(ValueError, "missing required temporal"):
            self.temporal._static_parity(frame.drop(columns=["change_mod:0"]), modern)
