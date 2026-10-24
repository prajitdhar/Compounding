"""Cutoff-zero adapter checks, including immutable cutoff-500 drivers."""

import contextlib
import importlib.util
import io
import json
import sys
from unittest import mock

import numpy as np
from scipy import sparse

import test_cumulative_temporal as temporal_test_data
from test_features import FeatureTestData, GOOGLE_DIRECTORY


class GoogleCumulativeTemporalCutoffZeroTests(FeatureTestData):
    snapshot_files = temporal_test_data.GoogleCumulativeTemporalTests.snapshot_files
    assert_immutable_inputs = temporal_test_data.GoogleCumulativeTemporalTests.assert_immutable_inputs
    raw_oracle = temporal_test_data.GoogleCumulativeTemporalTests.raw_oracle
    assert_frames_equal = temporal_test_data.GoogleCumulativeTemporalTests.assert_frames_equal
    setup_prepared = temporal_test_data.GoogleCumulativeTemporalTests.setup_prepared
    driver_args = temporal_test_data.GoogleCumulativeTemporalTests.driver_args
    product_path = temporal_test_data.GoogleCumulativeTemporalTests.product_path
    long = temporal_test_data.GoogleCumulativeTemporalTests.long
    dense_ppmi = staticmethod(temporal_test_data.GoogleCumulativeTemporalTests.dense_ppmi)
    temporal_oracle = temporal_test_data.GoogleCumulativeTemporalTests.temporal_oracle
    assert_temporal_oracle = temporal_test_data.GoogleCumulativeTemporalTests.assert_temporal_oracle

    def setUp(self):
        super().setUp()
        for path in self.ratings.values():
            ratings = self.pd.read_csv(path, sep="\t")
            ratings["is_original"] = True
            ratings.to_csv(path, sep="\t", index=False)
        for name, attr in (("cumulative_features", "cumulative"),
                           ("cumulative_temporal", "temporal"),
                           ("cumulative_temporal_cutoff0", "extension")):
            spec = importlib.util.spec_from_file_location(name, GOOGLE_DIRECTORY / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            setattr(self, attr, module)
        self.modern = self.root / "modern"
        self.work = self.root / "cumulative-work"
        self.cumulative_output = self.root / "temporal-output"
        self.prepared = None
        self.specification = None
        self.production_guard = self.extension._require_configuration
        patch = mock.patch.object(self.extension, "_require_configuration", side_effect=self.validate_test_data_configuration)
        patch.start()
        self.addCleanup(patch.stop)
        old_names = ("cumulative_temporal.py", "cumulative_features.py",
                     "features.py")
        self.original_sources = {GOOGLE_DIRECTORY / name: self.digest(GOOGLE_DIRECTORY / name)
                                 for name in old_names}

    @staticmethod
    def validate_test_data_configuration(args):
        if (args.context_size != 3 or args.condition != "CompoundAware"
                or args.weight != "PPMI" or args.time_step != 10 or args.cutoff != 0):
            raise ValueError("Invalid cutoff-zero configuration for tiny test data")

    def run_temporal(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            self.extension.prepare(self.driver_args("prepare", *extra))
            return self.extension.run_features(self.driver_args("features", *extra))

    def test_defaults_are_cutoff_zero_and_explicit_other_scopes_are_rejected(self):
        values = ["plan", "--prepared-root", str(self.cache), "--modern-root", str(self.modern),
                  "--work-root", str(self.work), "--output-root", str(self.cumulative_output),
                  "--temp-directory", str(self.spill)]
        with mock.patch.object(self.extension, "_require_configuration", side_effect=self.production_guard):
            args = self.extension.parse_args(values)
            self.assertEqual((args.context_size, args.condition, args.weight, args.cutoff, args.time_step),
                             (30000, "CompoundAware", "PPMI", 0, 10))
            for option, value in (("--context-size", "5000"), ("--condition", "CompoundPooled"),
                                  ("--weight", "RAW"), ("--cutoff", "500"), ("--time-step", "20")):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises((ValueError, SystemExit)):
                        self.extension.parse_args([*values, option, value])

    def test_scoped_overrides_are_restored_after_success_and_failure(self):
        old_guard, old_specification = self.temporal._require_configuration, self.temporal.read_specification
        with self.extension._cutoff_zero_scope():
            self.assertIs(self.temporal._require_configuration, self.extension._require_configuration)
            self.assertIsNot(self.temporal.read_specification, old_specification)
        self.assertIs(self.temporal._require_configuration, old_guard)
        self.assertIs(self.temporal.read_specification, old_specification)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with self.extension._cutoff_zero_scope():
                raise RuntimeError("interrupted")
        self.assertIs(self.temporal._require_configuration, old_guard)
        self.assertIs(self.temporal.read_specification, old_specification)
        # Importing/using this adapter does not relax the old cutoff-500 guard.
        with self.assertRaisesRegex(ValueError, "cutoff=500"):
            self.temporal._require_configuration(self.driver_args())

    def test_zero_cutoff_still_requires_positive_counts_in_both_prefixes(self):
        current = {"minimum": np.array([1, 0, 1, 1]),
                   "compound": sparse.csr_matrix([[1, 0], [1, 0], [1, 0], [0, 0]])}
        previous = {"minimum": np.array([1, 1, 0, 1]),
                    "compound": sparse.csr_matrix([[1, 1], [1, 0], [1, 0], [1, 0]])}
        for role in ("modifier", "head"):
            current[role], previous[role] = current["compound"], previous["compound"]
        values = self.temporal._temporal_values(current, previous, 0)
        for column in self.temporal.TEMPORAL_COLUMNS:
            np.testing.assert_allclose(values[column], [1 / np.sqrt(2), np.nan, np.nan, np.nan],
                                       equal_nan=True)

    def test_count_cache_is_reused_but_feature_identity_includes_the_adapter(self):
        self.setup_prepared()
        args = self.driver_args("prepare")
        with contextlib.redirect_stdout(io.StringIO()):
            old_work = self.cumulative.prepare(args)
        before = self.snapshot_files(old_work)
        static = self.cumulative.read_specification(args)
        specification = self.extension.read_specification(args)
        self.assertEqual(specification["count_input_identity"], static["input_identity"])
        self.assertEqual(specification["work"], old_work)
        material = json.loads(specification["input_identity_material"])
        self.assertEqual(material["cutoff"], 0)
        self.assertEqual(material["temporal_policy"], self.temporal.TEMPORAL_POLICY)
        self.assertEqual(material["temporal_software"]["code_sha256"]["cumulative_temporal.py"],
                         self.digest(GOOGLE_DIRECTORY / "cumulative_temporal.py"))
        extension_sources = material["cutoff_extension_software"]["code_sha256"]
        self.assertEqual(extension_sources["cumulative_temporal_cutoff0.py"],
                         self.digest(GOOGLE_DIRECTORY / "cumulative_temporal_cutoff0.py"))
        with mock.patch.object(self.cumulative, "_first_seen_query", side_effect=AssertionError("cache rescanned")):
            self.run_temporal()
        self.assertEqual(self.snapshot_files(old_work), before)
        self.assertEqual({path: self.digest(path) for path in self.original_sources}, self.original_sources)
        self.assert_immutable_inputs()

    def test_all_endpoints_match_independent_oracle_and_completed_run_is_reusable(self):
        self.setup_prepared()
        self.run_temporal()
        for endpoint in (1820, 1830, 1840, 1850):
            with self.subTest(endpoint=endpoint):
                self.assert_temporal_oracle(endpoint)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.extension.check(self.driver_args("check"))["status"], "complete")
        before = self.snapshot_files(self.cumulative_output)
        with mock.patch.object(self.google, "_load_vectors", side_effect=AssertionError("completed work reread")):
            self.run_temporal()
        self.assertEqual(self.snapshot_files(self.cumulative_output), before)
        self.assert_immutable_inputs()

    def test_incompatible_temporal_output_is_not_overwritten_even_with_force(self):
        self.setup_prepared()
        # A bypass for test data lets the old implementation emit an old-format
        # identity at zero, as an adversarial same-root source information collision.
        with mock.patch.object(self.temporal, "_require_configuration", side_effect=self.validate_test_data_configuration):
            with contextlib.redirect_stdout(io.StringIO()):
                self.temporal.prepare(self.driver_args("prepare"))
                self.temporal.run_features(self.driver_args())
        before = self.snapshot_files(self.cumulative_output)
        with self.assertRaisesRegex(RuntimeError, "incompatible identity"):
            self.run_temporal("--force")
        self.assertEqual(self.snapshot_files(self.cumulative_output), before)
        self.assert_immutable_inputs()


if __name__ == "__main__":
    import unittest
    unittest.main()
