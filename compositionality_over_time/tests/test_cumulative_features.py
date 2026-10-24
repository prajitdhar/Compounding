"""Cumulative snapshots use exact types and freshly weighted raw populations.

Only tiny temporary test data are read or written. The prepared Google cache and
the modern reference are immutable inputs to the cumulative driver.
"""

import contextlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse

TEST_DIRECTORY = Path(__file__).resolve().parent
if str(TEST_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TEST_DIRECTORY))
from test_features import FeatureTestData, GOOGLE_DIRECTORY


class GoogleCumulativeFeatureTests(FeatureTestData):
    def setUp(self):
        super().setUp()
        # Production ratings carry this row identity; add it to the tiny shared
        # test data before its preparation hashes are frozen.
        for path in self.ratings.values():
            ratings = self.pd.read_csv(path, sep="\t")
            ratings["is_original"] = True
            ratings.to_csv(path, sep="\t", index=False)
        spec = importlib.util.spec_from_file_location(
            "cumulative_features",
            GOOGLE_DIRECTORY / "cumulative_features.py")
        self.cumulative = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.cumulative
        spec.loader.exec_module(self.cumulative)
        self.modern = self.root / "modern"
        self.work = self.root / "cumulative-work"
        self.cumulative_output = self.root / "cumulative-output"
        self.prepared = None
        self.specification = None

    def setup_prepared(self, *, size=3, cutoffs=(0, 10, 20, 20000)):
        """Make the same immutable cache and frozen CSV reference as production."""
        args = self.args("prepare", "--untagged")
        with contextlib.redirect_stdout(io.StringIO()):
            self.prepared = self.google.prepare(args)
            self.specification = self.google._specification(args)
            self.google.run_features(self.args(
                "features", "--untagged", "--context-size", str(size),
                "--temporal", "10000", "--weights", "PPMI", "--families", "features",
                "--conditions", "CompoundPooled", "--cutoffs", *map(str, cutoffs)))
        modern_local5 = self.modern / "local5"
        modern_local5.mkdir(parents=True)
        for path in self.output.glob("*.parquet"):
            self.pd.read_parquet(path).to_csv(modern_local5 / path.with_suffix(".csv").name,
                                            index=False)
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
            "--output-root", str(self.cumulative_output),
            "--temp-directory", str(self.spill), "--context-size", "3",
            "--threads", "1", "--memory-limit", "256MB", *extra])

    def run_cumulative(self, *extra):
        args = self.driver_args("features", *extra)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cumulative.prepare(self.driver_args("prepare", *extra))
            result = self.cumulative.run_features(args)
        return result

    def product_path(self, endpoint, *, cutoff=0, size=3, long=False, csv=False):
        suffix = "_long" if long else ""
        extension = "csv" if csv else "parquet"
        return self.cumulative_output / "local5" / f"through_{endpoint}" / (
            f"features_CompoundPooled_withSetting_PPMI_UnTagged_ctx{size}"
            f"_10000_{cutoff}_na{suffix}.{extension}")

    def long(self, endpoint, *, cutoff=0, size=3):
        return self.pd.read_parquet(self.product_path(endpoint, cutoff=cutoff, size=size, long=True))

    def row(self, endpoint, modifier="red", head="car", *, cutoff=0, size=3):
        frame = self.long(endpoint, cutoff=cutoff, size=size)
        return frame.loc[(frame.modifier == modifier) & (frame["head"] == head)].iloc[0]

    def snapshot_files(self, directory):
        return {path: (self.digest(path), path.stat().st_mtime_ns)
                for path in directory.rglob("*") if path.is_file()}

    def assert_immutable_inputs(self):
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns)
                          for path in self.immutable}, self.immutable)
        self.assertEqual(set(self.snapshot_files(self.prepared)) |
                         set(self.snapshot_files(self.modern)),
                         set(self.immutable) - self.source_files)
        self.assert_inputs_unchanged()

    def raw_oracle(self, endpoint, size=3):
        """Aggregate original tiny raw tables independently of cached summaries."""
        tables = {}
        for name in self.shared.FEATURE_TABLES:
            frame = self.pd.read_parquet(self.input / f"{name}.parquet")
            frame = frame.loc[(frame.time <= endpoint) & (frame["count"] > 0)].copy()
            for column in ("word", "modifier", "head"):
                if column in frame:
                    frame[column] = frame[column].str.replace(r"_[A-Z]+$", "", regex=True)
            tables[name] = frame
        pairs, words = self.specification["pairs"], self.specification["words"]
        pair_ids = {(r.modifier, r.head): int(r.target) for r in pairs.itertuples(index=False)}
        word_ids = {r.word: int(r.target) for r in words.itertuples(index=False)}
        counts = tables["bigram_counts"].groupby(["modifier", "head"], as_index=False)["count"].sum()
        word_counts = tables["unigram_counts"].groupby("word")["count"].sum()
        modifier = counts.groupby("modifier")["count"].sum()
        head = counts.groupby("head")["count"].sum()
        mod_prod = counts.groupby("modifier")["head"].nunique()
        head_prod = counts.groupby("head")["modifier"].nunique()
        records = []

        def add(kind, target, value):
            records.append({"kind": kind, "target": str(target), "partner": "",
                            "time": 0, "count": int(value)})

        for kind in (0, 1):
            add(kind, "", counts["count"].sum())
        add(6, "", word_counts.sum())
        add(103, "", counts.modifier.nunique())
        add(104, "", counts["head"].nunique())
        add(112, "", len(counts))
        for row in counts.itertuples(index=False):
            target = pair_ids.get((row.modifier, row.head))
            if target is not None:
                for kind in (2, 9):
                    add(kind, target, row.count)
        for word, target in word_ids.items():
            for kind, values in ((3, modifier), (4, head), (5, word_counts),
                                 (7, modifier), (8, head), (113, mod_prod), (114, head_prod)):
                if word in values:
                    add(kind, word, values[word])
        frequencies = self.pd.DataFrame(records)
        controls, occurrence, minimum, mod_ids, head_ids = self.google._controls(
            self.specification, frequencies, 0)
        matrices = {code: np.zeros((len(pairs) if code == 2 else len(words), size))
                    for code in (2, 3, 4, 5)}
        backgrounds = {code: np.zeros(size) for code in (0, 1)}
        context_ids = {r.context: int(r.rank) - 1
                       for r in self.specification["vocabulary"].head(size).itertuples(index=False)}
        for row in tables["bigrams"].itertuples(index=False):
            context = context_ids.get(row.context)
            if context is None:
                continue
            backgrounds[0][context] += row.count
            pair = pair_ids.get((row.modifier, row.head))
            if pair is not None:
                matrices[2][pair, context] += row.count
            for code, word in ((3, row.modifier), (4, row.head)):
                target = word_ids.get(word)
                if target is not None:
                    matrices[code][target, context] += row.count
        for row in tables["unigrams"].itertuples(index=False):
            context = context_ids.get(row.context)
            if context is None:
                continue
            backgrounds[1][context] += row.count
            target = word_ids.get(row.word)
            if target is not None:
                matrices[5][target, context] += row.count
        raw = {kind: sparse.csr_matrix(matrix) for kind, matrix in matrices.items()}
        return raw, backgrounds, controls, occurrence, minimum, mod_ids, head_ids

    def feature_oracle(self, endpoint, *, size=3, cutoff=0):
        raw, background, controls, occurrence, minimum, mod_ids, head_ids = self.raw_oracle(endpoint, size)
        pooled = raw[3] + raw[4]
        compound = self.google.weight_matrix(raw[2], background[0], ppmi=True)
        constituent = self.google.weight_matrix(pooled, 2 * background[0], ppmi=True)
        general = self.google.weight_matrix(raw[5], background[1], ppmi=True)
        frame = self.specification["pairs"][["modifier", "head"]].copy()
        frame["time"] = 0
        frame = self.pd.concat([frame, self.google.static_features(
            compound, constituent[mod_ids], constituent[head_ids])], axis=1)
        frame = frame.merge(controls["CompoundPooled"], on=self.google.PAIR,
                            how="left", validate="one_to_one")
        comparison = self.google.setting_features(
            constituent, general, pooled, raw[5], occurrence["CompoundPooled"][0],
            occurrence["CompoundAgnostic"][0], "constituent")
        for suffix, ids in (("mod", mod_ids), ("head", head_ids)):
            values = comparison.iloc[ids].reset_index(drop=True).add_suffix(f"_{suffix}")
            frame = self.pd.concat([frame, values], axis=1)
        return frame.loc[minimum > cutoff].reset_index(drop=True)

    def assert_frames_equal(self, actual, expected):
        keys = [column for column in ("modifier", "head", "word", "time", "source")
                if column in actual and column in expected]
        self.pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-8, atol=1e-10, check_like=True)

    def test_defaults_select_the_existing_pooled_ppmi_untagged_model(self):
        arguments = ["plan", "--prepared-root", str(self.cache),
                     "--modern-root", str(self.modern), "--work-root", str(self.work),
                     "--output-root", str(self.cumulative_output),
                     "--temp-directory", str(self.spill)]
        args = self.cumulative.parse_args(arguments)
        self.assertEqual(args.context_size, 5000)
        self.assertEqual(args.cutoff, 0)
        self.assertEqual(args.time_step, 10)
        self.assertEqual(args.condition, "CompoundPooled")
        self.assertEqual(args.weight, "PPMI")

    def test_every_endpoint_matches_independent_raw_population_math(self):
        self.setup_prepared()
        self.run_cumulative()
        for endpoint in (1820, 1830, 1840, 1850):
            with self.subTest(endpoint=endpoint):
                self.assert_frames_equal(self.long(endpoint), self.feature_oracle(endpoint))
        self.assert_immutable_inputs()

    def test_all_pos_tokens_accumulate_but_productivity_counts_each_type_once(self):
        self.setup_prepared()
        self.run_cumulative()
        for endpoint, comp, modifier, head in (
                (1820, 116, 137, 1142), (1830, 184, 215, 2237),
                (1840, 254, 324, 3356), (1850, 303, 387, 4424)):
            with self.subTest(endpoint=endpoint):
                row = self.row(endpoint)
                self.assertEqual(row.comp_freq, comp)
                self.assertEqual(row.assoc_pair_freq, comp)
                self.assertEqual(row.mod_freq, modifier)
                self.assertEqual(row.head_freq, head)
                self.assertEqual(row.mod_prod, 2)
                self.assertEqual(row.head_prod, 3)
                self.assertAlmostEqual(row.mod_family_size, np.log2(2 / 10))
                self.assertAlmostEqual(row.head_family_size, np.log2(3 / 10))
                self.assertAlmostEqual(row.mod_family_size_new, np.log2(2 / 8))
                self.assertAlmostEqual(row.head_family_size_new, np.log2(3 / 9))
        self.assert_immutable_inputs()

    def test_ranked_context_prefix_deduplicates_surface_spellings_before_weighting(self):
        self.setup_prepared(size=2)
        self.run_cumulative("--context-size", "2")
        vocabulary = self.pd.read_parquet(self.prepared / "contexts.parquet")
        self.assertEqual(vocabulary.head(2).context.tolist(), ["beta_VERB", "alpha_NOUN"])
        self.assertEqual(len(vocabulary), 6)
        for endpoint in (1820, 1830, 1840, 1850):
            with self.subTest(endpoint=endpoint):
                self.assert_frames_equal(self.long(endpoint, size=2),
                                         self.feature_oracle(endpoint, size=2))
        self.assert_immutable_inputs()

    def test_last_endpoint_is_numerically_identical_to_frozen_modern_features(self):
        self.setup_prepared()
        self.run_cumulative("--end-decades", "1820", "1830", "1850")
        self.assert_numeric_output_equal(
            self.product_path(1850, long=True),
            self.output / self.product_path(1850, long=True).name)
        self.assert_numeric_output_equal(
            self.product_path(1850), self.output / self.product_path(1850).name)
        self.assert_immutable_inputs()

    def test_pre_attestation_pair_is_absent_until_its_first_raw_count(self):
        for name in ("bigrams", "bigram_counts"):
            frame = self.pd.read_parquet(self.input / f"{name}.parquet")
            before = (frame.modifier == "jump_ADJ") & (frame["head"] == "rope_NOUN") & (frame.time < 1830)
            self.write_table(name, frame.loc[~before].to_dict("records"))
        manifest_path = self.input / "corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name in ("bigrams", "bigram_counts"):
            path = self.input / f"{name}.parquet"
            manifest["tables"][name].update(bytes=path.stat().st_size, sha256=self.digest(path))
        manifest_path.write_text(json.dumps(manifest))
        self.input_bytes = {path: path.read_bytes() for path in self.input_bytes}
        self.setup_prepared()
        self.run_cumulative()
        first = self.long(1820)
        self.assertFalse(((first.modifier == "jump") & (first["head"] == "rope")).any())
        wide = self.pd.read_csv(self.product_path(1820, csv=True))
        before = wide.loc[(wide.modifier == "jump") & (wide["head"] == "rope")]
        self.assertEqual(len(before), 3)
        self.assertTrue(before["eligible:0"].fillna(0).eq(0).all())
        self.assertTrue(before["sim_cpf_beta:0"].isna().all())
        self.assertEqual(self.row(1830, "jump", "rope").comp_freq, 15)
        self.assert_immutable_inputs()

    def test_cutoff_is_strict_and_pair_becomes_eligible_as_raw_counts_accumulate(self):
        self.setup_prepared()
        self.run_cumulative("--cutoff", "20")
        for endpoint in (1820, 1830):
            frame = self.long(endpoint, cutoff=20)
            self.assertFalse(((frame.modifier == "jump") & (frame["head"] == "rope")).any())
        self.assertEqual(self.row(1840, "jump", "rope", cutoff=20).comp_freq, 35)
        self.assert_frames_equal(self.long(1840, cutoff=20),
                                 self.feature_oracle(1840, cutoff=20))
        self.assert_immutable_inputs()

    def test_empty_epochs_keep_feature_columns_and_all_wide_ratings(self):
        self.setup_prepared()
        self.run_cumulative("--cutoff", "20000")
        rating_keys = ["modifier", "head", "source", "is_adj", "rating"]
        ratings = self.specification["ratings"][rating_keys]
        for endpoint in (1820, 1830, 1840, 1850):
            with self.subTest(endpoint=endpoint):
                long = self.long(endpoint, cutoff=20000)
                self.assertTrue(long.empty)
                self.assertTrue(set(self.google._feature_columns("CompoundPooled")).issubset(long.columns))
                wide = self.pd.read_csv(self.product_path(endpoint, cutoff=20000, csv=True))
                self.assert_frames_equal(wide[rating_keys], ratings)
                self.assertTrue(wide["eligible:0"].fillna(0).eq(0).all())
                self.assertTrue(wide["sim_cpf_beta:0"].isna().all())
        self.assert_immutable_inputs()

    def test_completed_outputs_resume_without_rewriting_epochs_or_cache(self):
        self.setup_prepared()
        self.run_cumulative()
        before = self.snapshot_files(self.cumulative_output)
        with mock.patch.object(self.cumulative, "_write_endpoint",
                               side_effect=AssertionError("completed snapshot was recomputed")):
            self.run_cumulative()
        self.assertEqual(self.snapshot_files(self.cumulative_output), before)
        self.assert_immutable_inputs()

    def test_first_appearance_preparation_reuses_durable_bucket_summaries(self):
        self.setup_prepared()
        args = self.driver_args("prepare")
        with contextlib.redirect_stdout(io.StringIO()):
            work = self.cumulative.prepare(args)
        before = self.snapshot_files(work)
        with mock.patch.object(self.cumulative, "_first_seen_query",
                               side_effect=AssertionError("exact types were rescanned")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cumulative.prepare(args), work)
        self.assertEqual(self.snapshot_files(work), before)
        self.assert_immutable_inputs()

    def test_interrupted_feature_run_resumes_after_durable_endpoints(self):
        self.setup_prepared()
        original = self.cumulative._write_endpoint

        def interrupt_after_publication(specification, endpoint, frame, args):
            record = original(specification, endpoint, frame, args)
            if endpoint == 1830:
                raise InterruptedError("synthetic interruption after durable endpoint")
            return record

        with mock.patch.object(self.cumulative, "_write_endpoint",
                               side_effect=interrupt_after_publication):
            with self.assertRaisesRegex(InterruptedError, "durable endpoint"):
                self.run_cumulative()
        self.assertFalse((self.cumulative_output / "features_complete.json").exists())
        before = {}
        for endpoint in (1820, 1830):
            before.update(self.snapshot_files(
                self.cumulative_output / "local5" / f"through_{endpoint}"))
        written = []

        def record_remaining(specification, endpoint, frame, args):
            written.append(endpoint)
            return original(specification, endpoint, frame, args)

        with mock.patch.object(self.cumulative, "_write_endpoint", side_effect=record_remaining):
            self.run_cumulative()
        self.assertEqual(written, [1840, 1850])
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns)
                          for path in before}, before)
        self.assert_frames_equal(self.long(1840), self.feature_oracle(1840))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cumulative.check(self.driver_args("check"))["status"], "complete")
        self.assert_immutable_inputs()

    def test_unseen_endpoint_and_incompatible_output_identity_are_refused(self):
        self.setup_prepared()
        with self.assertRaisesRegex(ValueError, "(?i)(endpoint|decade|available)"):
            self.run_cumulative("--end-decades", "1820", "1860")
        self.run_cumulative()
        before = self.snapshot_files(self.cumulative_output)
        with self.assertRaisesRegex((ValueError, RuntimeError), "(?i)(identity|configuration|different|incompatible|force)"):
            self.run_cumulative("--cutoff", "10")
        self.assertEqual(self.snapshot_files(self.cumulative_output), before)
        self.assert_immutable_inputs()

    def test_incompatible_population_cache_and_invalid_configuration_are_refused(self):
        self.setup_prepared()
        incompatible = self.root / "incompatible-prepared"
        shutil.copytree(self.prepared, incompatible)
        path = incompatible / "identity.json"
        identity = json.loads(path.read_text())
        identity["compound_population"] = "adj_noun_and_noun_noun"
        path.write_text(json.dumps(identity))
        # Keep this deliberately old population internally consistent, so the
        # semantic population check is exercised rather than just a bad digest.
        completion = incompatible / "complete.json"
        receipt = json.loads(completion.read_text())
        receipt["fingerprint"] = self.google._digest(identity)
        completion.write_text(json.dumps(receipt))
        with self.assertRaisesRegex((ValueError, RuntimeError), "(?i)(population|all.?pos|all.lexical|identity)"):
            self.cumulative.read_specification(self.driver_args("plan", "--prepared-root", str(incompatible)))
        for option, value in (("--time-step", "0"), ("--cutoff", "-1"),
                              ("--context-size", "0")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises((SystemExit, ValueError)):
                    self.cumulative.parse_args([
                        "plan", "--prepared-root", str(self.prepared),
                        "--modern-root", str(self.modern), "--work-root", str(self.work),
                        "--output-root", str(self.cumulative_output),
                        "--temp-directory", str(self.spill), option, value])
        self.assert_immutable_inputs()
