"""Numerical parity with the shared COHA calculations on tiny Google inputs.

All files/checkpoints are temporary. No production corpus, ratings, or Slurm job
is read or changed; Google vocabulary rank is supplied rather than recomputed.
"""

import contextlib
import hashlib
import importlib.util
import io
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
SOURCE_DIRECTORY = GOOGLE_DIRECTORY / "feature_helpers"
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.append(str(SOURCE_DIRECTORY))
if str(GOOGLE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GOOGLE_DIRECTORY))


class FeatureTestData(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("DuckDB, Pandas, and PyArrow are required")
        self.duckdb, self.pd, self.pa, self.pq = duckdb, pd, pa, pq
        import coha_features
        import coha_feature_pipeline
        self.shared, self.shared_driver = coha_features, coha_feature_pipeline
        spec = importlib.util.spec_from_file_location(
            "features", GOOGLE_DIRECTORY / "features.py")
        self.google = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.google
        spec.loader.exec_module(self.google)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "inputs"
        self.input.mkdir()
        self.output = self.root / "google-output"
        self.cache = self.root / "cache"
        self.spill = self.root / "spill"
        self.vocabulary = self.root / "contexts.tsv"
        self.ratings = {}
        self.write_test_data()

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_table(self, name, rows):
        frame = self.pd.DataFrame(rows).sample(frac=1, random_state=17).reset_index(drop=True)
        path = self.input / f"{name}.parquet"
        self.pq.write_table(self.pa.Table.from_pandas(frame, preserve_index=False), path,
                            row_group_size=7)
        return path

    def write_test_data(self):
        # Two independent occurrence populations and two event matrices. Context
        # event mass is deliberately not usable as occurrence frequency.
        counts, compounds, words, word_counts = [], [], [], []
        pair_series = {
            ("red_ADJ", "car_NOUN"): {1820: 12, 1830: 11, 1850: 14},
            ("red_NOUN", "car_NOUN"): {1820: 4, 1830: 7, 1850: 5},
            ("car_NOUN", "red_NOUN"): {1820: 6, 1830: 2, 1840: 9, 1850: 4},
            ("blue_ADJ", "car_NOUN"): {1820: 20, 1830: 25, 1840: 40, 1850: 15},
            ("red_ADJ", "bus_NOUN"): {1820: 15, 1830: 8, 1840: 30, 1850: 10},
            ("ghost_ADJ", "ship_NOUN"): {1820: 12, 1830: 12},
            ("jump_ADJ", "rope_NOUN"): {1820: 5, 1830: 15, 1840: 15, 1850: 15},
            ("red_VERB", "car_NOUN"): {1820: 100, 1830: 50, 1840: 70, 1850: 30},
            ("the_DET", "car_NOUN"): {1820: 1000, 1830: 1000, 1840: 1000, 1850: 1000},
            ("move_VERB", "the_DET"): {1820: 70, 1830: 70, 1840: 70, 1850: 70},
        }
        for (modifier, head), series in pair_series.items():
            for time, count in series.items():
                counts.append({"modifier": modifier, "head": head, "time": time, "count": count})
        patterns = {
            ("red_ADJ", "car_NOUN"): {1820: (3, 4), 1830: (6, 2), 1850: (1, 5)},
            ("red_NOUN", "car_NOUN"): {1820: (1, 2), 1830: (2, 1), 1850: (2, 2)},
            ("car_NOUN", "red_NOUN"): {1820: (2, 1), 1830: (3, 1), 1840: (1, 3), 1850: (2, 2)},
            ("blue_ADJ", "car_NOUN"): {time: (6, 2) for time in (1820, 1830, 1840, 1850)},
            ("red_ADJ", "bus_NOUN"): {time: (4, 3) for time in (1820, 1830, 1840, 1850)},
            ("ghost_ADJ", "ship_NOUN"): {1820: (2, 2), 1830: (2, 2)},
            ("jump_ADJ", "rope_NOUN"): {time: (1, 4) for time in (1820, 1830, 1840, 1850)},
        }
        for time in (1820, 1830, 1840, 1850):
            totals = [0, 0]
            for (modifier, head), series in patterns.items():
                if time not in series:
                    continue
                for index, context in enumerate(("alpha_NOUN", "beta_VERB")):
                    count = series[time][index]
                    totals[index] += count
                    compounds.append({"modifier": modifier, "head": head, "time": time,
                                      "context": context, "count": count})
            # An unrated eligible pair balances the background to powers of two.
            # The rated zero/vector row is exactly proportional to the full
            # population, so both PPMI weights are genuinely zero (not missing).
            for modifier, head, values, occurrence in (
                ("fill_NOUN", "bank_NOUN", (64 - totals[0], 64 - totals[1]), 50),
                ("zero_ADJ", "vector_NOUN", (64, 64), 64),
            ):
                counts.append({"modifier": modifier, "head": head, "time": time,
                               "count": occurrence})
                compounds.extend({"modifier": modifier, "head": head, "time": time,
                                  "context": context, "count": count}
                                 for context, count in zip(("alpha_NOUN", "beta_VERB"), values))
            # Tagged compounds exclude this VERB target. UnTagged compounds
            # include it and retain the ADJ tag on this context dimension.
            compounds.append({"modifier": "red_VERB", "head": "car_NOUN", "time": time,
                              "context": "extra_ADJ", "count": 200})
        compounds.append({"modifier": "red_ADJ", "head": "car_NOUN", "time": 1820,
                          "context": "thin_NUM", "count": 0})
        general = {
            "red_ADJ": (4, 3, 9), "red_NOUN": (2, 6, 2), "car_NOUN": (3, 5, 8),
            "blue_ADJ": (9, 2, 1), "bus_NOUN": (3, 4, 2), "ghost_ADJ": (2, 1, 3),
            "ship_NOUN": (2, 5, 2), "jump_ADJ": (1, 4, 2), "rope_NOUN": (1, 5, 3),
            "fill_NOUN": (5, 2, 1), "bank_NOUN": (2, 8, 1), "zero_ADJ": (4, 4, 4),
            "vector_NOUN": (8, 8, 8), "red_VERB": (20, 4, 5), "the_DET": (100, 40, 2),
            "move_VERB": (5, 10, 1),
        }
        frequencies = {"red_ADJ": 60, "red_NOUN": 30, "car_NOUN": 25,
                       "blue_ADJ": 80, "bus_NOUN": 80, "ship_NOUN": 60,
                       "jump_ADJ": 60, "rope_NOUN": 60, "fill_NOUN": 100,
                       "bank_NOUN": 100, "zero_ADJ": 100, "vector_NOUN": 100,
                       "red_VERB": 600, "the_DET": 2000, "move_VERB": 200,
                       "alpha_NOUN": 80, "beta_VERB": 40, "extra_ADJ": 30,
                       "delta_ADV": 20, "common_NOUN": 10, "thin_NUM": 5}
        for time in (1820, 1830, 1840, 1850):
            for word, values in general.items():
                for context, count in zip(("alpha_NOUN", "beta_VERB", "extra_ADJ"), values):
                    words.append({"word": word, "time": time, "context": context, "count": count})
            words.append({"word": "red_ADJ", "time": time, "context": "thin_NUM", "count": 0})
            word_counts.extend({"word": word, "time": time, "count": count}
                               for word, count in frequencies.items())
        for name, rows in (("bigrams", compounds), ("bigram_counts", counts),
                           ("unigrams", words), ("unigram_counts", word_counts)):
            self.write_table(name, rows)
        self.vocabulary.write_text(
            "norm\tpos\tlemma\trank\tscore\tdecades_observed\tsurface_match_count\tlemma_match_count\n"
            "alphas\tNOUN\talpha\t2\t0.2\t4\t100\t320\n"
            "beta\tVERB\tbeta\t1\t0.3\t4\t160\t160\n"
            "alpha\tNOUN\talpha\t2\t0.2\t4\t220\t320\n"
            "extra\tADJ\textra\t3\t0.1\t4\t120\t120\n"
            "delta\tADV\tdelta\t4\t0.09\t4\t80\t80\n"
            "common\tNOUN\tcommon\t5\t0.08\t4\t40\t40\n"
            "thin\tNUM\tthin\t6\t0.07\t4\t20\t20\n")
        ratings_rows = [{"modifier": modifier, "head": head, "is_adj": adjective,
                         "rating": index + .5}
                        for index, (modifier, head, adjective) in enumerate([
                            ("red", "car", True), ("car", "red", False),
                            ("ghost", "ship", True), ("zero", "vector", True),
                            ("jump", "rope", True)])]
        for name in ("reddy90", "cordeiro90", "cordeiro100"):
            path = self.root / f"{name}.tsv"
            self.pd.DataFrame(ratings_rows).to_csv(path, sep="\t", index=False)
            self.ratings[name] = path
        table_paths = {name: self.input / f"{name}.parquet"
                       for name in self.shared.FEATURE_TABLES}
        manifest = {"schema_version": 3, "format": "parquet", "raw_counts": True,
                    "context_mode": "local5", "word_target_pos": "all_lexical",
                    "bigram_target_pos": "all_lexical",
                    "table_files": {name: path.name for name, path in table_paths.items()},
                    "tables": {name: {"path": str(path), "bytes": path.stat().st_size,
                                      "sha256": self.digest(path)}
                               for name, path in table_paths.items()},
                    "source_info": {"status": "complete", "stage": "feature_inputs",
                                   "corpus": {"name": "Google Books Ngrams V3"},
                                   "parameters": {"context_vocabulary_sha256": self.digest(self.vocabulary)},
                                   "time": {"column": "time", "unit": "calendar_year"}}}
        (self.input / "corpus_manifest.json").write_text(json.dumps(manifest))
        self.input_bytes = {path: path.read_bytes() for path in table_paths.values()}

    def args(self, stage, *extra):
        arguments = [stage, "--inputdir", str(self.input), "--outputdir", str(self.output),
                     "--cache-directory", str(self.cache), "--temp-directory", str(self.spill),
                     "--context-vocabulary", str(self.vocabulary),
                     "--context-sizes", "2", "3", "--temporal-bins", "10000", "10", "20",
                     "--threads", "1", "--memory-limit", "256MB", "--partitions", "2",
                     "--partition-batch-rows", "28", "--arrow-batch-rows", "7"]
        for name, path in self.ratings.items():
            arguments.extend([f"--{name}", str(path)])
        return self.google.parse_args([*arguments, *extra])

    def assert_inputs_unchanged(self):
        self.assertEqual({path: path.read_bytes() for path in self.input_bytes}, self.input_bytes)

    def oracle(self, destination, *, tagged, context_size=3, temporal=10,
               weights=("RAW", "PPMI"), families=("features", "temporal"), cutoffs=(0, 10, 12)):
        destination.mkdir(parents=True, exist_ok=True)
        if tagged:
            # Tagged retains the established COHA ADJ/NOUN -> NOUN population.
            tables = self.shared.load_inputs(self.input, tagged=True)
        else:
            # Construct the UnTagged population independently of its loader,
            # then reuse the pure calculations as a numerical reference:
            # every lexical bigram enters before pooling modifier/head POS.
            tables = {name: self.pd.read_parquet(self.input / f"{name}.parquet")
                      for name in self.shared.FEATURE_TABLES}
            tables["compounds"] = tables.pop("bigrams")
            tables["compound_counts"] = tables["bigram_counts"].copy()
            for name, frame in tables.items():
                for column in ("modifier", "head", "word"):
                    if column in frame:
                        frame[column] = frame[column].str.replace(r"_[A-Z]+$", "", regex=True)
                tables[name] = self.shared.aggregate(frame, [c for c in frame if c != "count"])
        vocabulary = self.google.read_context_vocabulary(self.vocabulary).head(context_size)
        for name in ("compounds", "unigrams"):
            tables[name] = tables[name].loc[tables[name].context.isin(vocabulary.context)].copy()
        unused_ranked, decade_counts, decades = self.shared.context_vocabulary(
            self.pd.read_parquet(self.input / "unigram_counts.parquet"))
        coverage = self.shared.context_coverage(vocabulary.context, decade_counts, decades)
        for weight in weights:
            args = SimpleNamespace(inputdir=str(self.input), outputdir=str(destination),
                                   tag=tagged, ppmi=weight == "PPMI", cutoffs=list(cutoffs),
                                   temporal=[temporal], association_population="by-setting", plot=False,
                                   **{name: str(path) for name, path in self.ratings.items()})
            for family in families:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.shared_driver.run_configuration(args, family == "temporal", tables,
                                                         context_size, vocabulary, coverage, decades)

    def assert_numeric_output_equal(self, actual_path, expected_path):
        actual, expected = self.pd.read_parquet(actual_path), self.pd.read_parquet(expected_path)
        keys = [column for column in ("modifier", "head", "word", "time", "source", "is_adj", "rating")
                if column in actual and column in expected]
        self.pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-8, atol=1e-10, check_like=True)


class GoogleFeatureParityTests(FeatureTestData):
    def test_both_input_readers_require_completed_feature_input_source_info(self):
        manifest_path = self.input / "corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        readers = {"bounded": lambda: self.google._inputs(self.args("prepare")),
                   "shared": lambda: self.shared.load_inputs(self.input, tagged=True)}
        for block in [None, {"status": "partial", "stage": "feature_inputs"},
                      {"status": "complete", "stage": "reduce"}]:
            invalid = dict(manifest)
            if block is None:
                invalid.pop("source_info")
            else:
                invalid["source_info"] = block
            manifest_path.write_text(json.dumps(invalid))
            for name, reader in readers.items():
                with self.subTest(reader=name, source_info=block):
                    with self.assertRaisesRegex(ValueError, "Feature-input source information"):
                        reader()

    def compare_configuration(self, *, tagged=True, context_size=3, temporal=10,
                              cutoffs=(0, 10, 12), weights=("RAW", "PPMI"),
                              families=("features", "temporal")):
        tag_options = [] if tagged else ["--untagged"]
        self.google.prepare(self.args("prepare", *tag_options))
        result = self.google.run_features(self.args(
            "features", *tag_options, "--context-size", str(context_size),
            "--temporal", str(temporal), "--cutoffs", *map(str, cutoffs),
            "--weights", *weights, "--families", *families))
        expected_root = self.root / ("oracle-tagged" if tagged else "oracle-untagged")
        self.oracle(expected_root, tagged=tagged, context_size=context_size,
                    temporal=temporal, cutoffs=cutoffs, weights=weights, families=families)
        expected_paths = list(expected_root.glob("*_na*.parquet"))
        expected_paths += list(expected_root.glob("setting_constituents_*.parquet"))
        self.assertTrue(expected_paths)
        for expected in expected_paths:
            actual = self.output / expected.name
            with self.subTest(file=expected.name):
                self.assertTrue(actual.is_file(), actual)
                self.assert_numeric_output_equal(actual, expected)
        self.assert_inputs_unchanged()
        return result

    def long(self, family, condition, weight="RAW", *, tagged=True, size=3,
             temporal=10, cutoff=0):
        tag = "Tagged" if tagged else "UnTagged"
        path = self.output / (f"{family}_{condition}_withSetting_{weight}_"
                              f"{tag}_ctx{size}_{temporal}_{cutoff}_na_long.parquet")
        return self.pd.read_parquet(path)

    def test_default_grid_and_all_feature_families_are_explicit(self):
        self.assertEqual(tuple(self.google.CONTEXT_SIZES),
                         (1000, 3000, 5000, 10000, 30000, 50000))
        self.assertEqual(tuple(self.google.TEMPORAL_BINS), (10000, 10, 20, 50, 100))
        args = self.args("features", "--grid-index", "0")
        self.assertFalse(args.untagged)
        self.assertEqual(args.weights, ["RAW", "PPMI"])
        self.assertEqual(args.families, ["features", "temporal"])
        self.assertEqual(args.conditions,
                         ["CompoundAware", "CompoundPooled", "CompoundAgnostic"])
        self.assertEqual(args.cutoffs, [0, 10, 50, 100, 500, 1000])

    def test_vocabulary_deduplicates_inflections_and_uses_frozen_rank(self):
        ranked = self.google.read_context_vocabulary(self.vocabulary)
        self.assertEqual(ranked.context.tolist(), ["beta_VERB", "alpha_NOUN", "extra_ADJ",
                                                 "delta_ADV", "common_NOUN", "thin_NUM"])
        self.assertEqual(ranked["rank"].tolist(), list(range(1, 7)))
        self.assertEqual(len(ranked), 6)

    def test_tagged_full_numerical_parity_with_shared_coha(self):
        self.compare_configuration(tagged=True)
        manifest = json.loads((self.output / "features_RAW_Tagged_ctx3_10_manifest.json").read_text())
        self.assertEqual(manifest["compound_population"], "adj_noun_and_noun_noun")
        self.assertEqual(manifest["association_population"],
                         {"CompoundAware": "eligible", "CompoundPooled": "eligible",
                          "CompoundAgnostic": "bigram_counts"})
        aware = self.long("features", "CompoundAware")
        selected = aware.loc[(aware.modifier == "red_ADJ") &
                             (aware["head"] == "car_NOUN") & (aware.time == 1820)].iloc[0]
        self.assertEqual(selected.comp_freq, 12)
        self.assertEqual(selected.assoc_pair_freq, 12)
        noun_counterpart = aware.loc[(aware.modifier == "red_NOUN") &
                                     (aware["head"] == "car_NOUN") & (aware.time == 1820)].iloc[0]
        self.assertEqual(noun_counterpart.comp_freq, 4)
        self.assertEqual(noun_counterpart.assoc_pair_freq, 4)
        strict = self.long("features", "CompoundAware", cutoff=12)
        self.assertFalse(((strict.modifier == "red_ADJ") &
                          (strict["head"] == "car_NOUN") & (strict.time == 1820)).any())
        self.assertFalse(aware.modifier.str.startswith("ghost").any())
        ppmi = self.long("features", "CompoundAware", "PPMI")
        zero = ppmi.loc[(ppmi.modifier == "zero_ADJ") & (ppmi.time == 1820)].iloc[0]
        self.assertEqual(zero.comp_freq, 64)
        self.assertTrue(self.pd.isna(zero.sim_with_modifier))
        self.assertAlmostEqual(zero.perc_type_mod, 2 / 3)
        changes = self.long("temporal", "CompoundAware", cutoff=10)
        first_eligible = changes.loc[(changes.modifier == "jump_ADJ") &
                                     (changes.time == 1830)].iloc[0]
        self.assertTrue(self.pd.isna(first_eligible.change_comp))
        gap = changes.loc[(changes.modifier == "red_ADJ") & (changes.time == 1850)].iloc[0]
        self.assertTrue(self.pd.isna(gap.change_comp))

    def test_untagged_pools_all_lexical_pair_pos_and_role_populations(self):
        self.compare_configuration(tagged=False)
        prepared = self.google._specification(self.args("prepare", "--untagged"))["root"]
        preparation = json.loads((prepared / "complete.json").read_text())
        self.assertEqual(preparation["source_info"]["parameters"]["compound_population"],
                         "all_lexical_bigrams")
        for weight in ("RAW", "PPMI"):
            for family in ("features", "temporal"):
                manifest = json.loads((self.output /
                    f"{family}_{weight}_UnTagged_ctx3_10_manifest.json").read_text())
                self.assertEqual(manifest["compound_population"], "all_lexical_bigrams")
                self.assertEqual(manifest["feature_definition_version"], 3)
                self.assertEqual(manifest["beta_policy"], "clipped_unit_interval")
                self.assertEqual(manifest["source_info"]["parameters"]["beta_policy"],
                                 "clipped_unit_interval")
                self.assertEqual(manifest["association_population"],
                                 dict.fromkeys(self.shared.CONDITIONS, "bigram_counts"))
        aware, pooled, agnostic = (self.long("features", name, tagged=False)
                                  for name in self.shared.CONDITIONS)
        select = lambda frame: frame.loc[(frame.modifier == "red") &
                                        (frame["head"] == "car") & (frame.time == 1820)].iloc[0]
        aware_row, pooled_row, agnostic_row = map(select, (aware, pooled, agnostic))
        for row in (aware_row, pooled_row, agnostic_row):
            self.assertEqual(row.comp_freq, 116)  # ADJ 12 + NOUN 4 + VERB 100.
            self.assertEqual(row.assoc_pair_freq, 116)
            self.assertEqual(row.mod_prod, 2)  # Collapsed POS variants are not extra types.
            self.assertEqual(row.head_prod, 3)  # red, blue and the (DET) modify car.
        self.assertEqual(aware_row.mod_freq, 131)
        self.assertEqual(aware_row.head_freq, 1136)
        self.assertEqual(pooled_row.mod_freq, 137)
        self.assertEqual(pooled_row.head_freq, 1142)
        self.assertEqual(agnostic_row.mod_freq, 690)
        self.assertEqual(agnostic_row.head_freq, 25)

    def test_untagged_compound_vectors_and_ppmi_use_full_all_pos_context_population(self):
        args = self.args("prepare", "--untagged")
        self.google.prepare(args)
        specification = self.google._specification(args)
        raw, backgrounds = self.google._load_vectors(specification, 1820, 10, 3, args)
        # Frozen rank order is beta_VERB, alpha_NOUN, extra_ADJ. The VERB-NOUN
        # target contributes 200 extra_ADJ events to both pair and background.
        self.assertEqual(backgrounds[0].tolist(), [128, 128, 200])
        pair = specification["pairs"].loc[(specification["pairs"].modifier == "red") &
                                     (specification["pairs"]["head"] == "car")].iloc[0]
        self.assertEqual(raw[2].getrow(int(pair.target)).toarray().ravel().tolist(), [6, 4, 200])
        full = self.pd.read_parquet(self.input / "bigrams.parquet")
        full = full.loc[(full.time == 1820) & full.context.isin(
            specification["vocabulary"].head(3).context)].copy()
        for column in ("modifier", "head"):
            full[column] = full[column].str.replace(r"_[A-Z]+$", "", regex=True)
        expected = self.shared.ppmi(full).loc[lambda frame: (frame.modifier == "red") &
                                              (frame["head"] == "car")]
        vector = self.google.weight_matrix(raw[2], backgrounds[0], ppmi=True)
        observed = vector.getrow(int(pair.target)).toarray().ravel()
        expected_by_context = expected.set_index("context")["count"].to_dict()
        for index, context in enumerate(specification["vocabulary"].head(3).context):
            self.assertAlmostEqual(observed[index], expected_by_context.get(context, 0))
        self.assertGreater(observed[2], 0)
        self.assertEqual(specification["vocabulary"].head(3).context.tolist(),
                         ["beta_VERB", "alpha_NOUN", "extra_ADJ"])

    def test_untagged_all_conditions_receive_identical_compound_vectors(self):
        self.google.prepare(self.args("prepare", "--untagged"))
        observed = []
        original = self.google.static_features

        def capture(compounds, modifiers, heads):
            observed.append((compounds.shape, compounds.data.tobytes(),
                             compounds.indices.tobytes(), compounds.indptr.tobytes()))
            return original(compounds, modifiers, heads)

        with mock.patch.object(self.google, "static_features", side_effect=capture):
            self.google.run_features(self.args("features", "--untagged", "--context-size", "3",
                "--temporal", "10", "--cutoffs", "0", "--weights", "RAW", "--families", "features"))
        self.assertEqual(len(observed), 12)  # Four periods x three conditions.
        for start in range(0, len(observed), 3):
            self.assertEqual(observed[start], observed[start + 1])
            self.assertEqual(observed[start], observed[start + 2])
        self.assert_inputs_unchanged()

    def test_untagged_productivity_deduplicates_pos_variants_and_decades(self):
        args = self.args("prepare", "--untagged")
        prepared = self.google.prepare(args)
        frequencies = self.pd.read_parquet(prepared / "frequencies/width-20/frequencies.parquet")
        values = self.google._frequency_maps(frequencies, 1820)
        self.assertEqual(values[(103, "")], 9)   # Distinct modifier lemmas.
        self.assertEqual(values[(104, "")], 8)   # Distinct head lemmas.
        self.assertEqual(values[(112, "")], 10)  # Not twenty decade-specific types.
        self.assertEqual(values[(113, "red")], 2)
        self.assertEqual(values[(114, "car")], 3)
        self.assertEqual(values[(3, "red")], 207)
        self.assertEqual(values[(4, "car")], 2229)
        self.assert_inputs_unchanged()

    def test_wider_bins_reaggregate_raw_populations_before_weighting(self):
        self.compare_configuration(tagged=False, context_size=2, temporal=20,
                                   cutoffs=(0, 10))

    def test_pooled_bin_occurrence_counts_remain_independent_of_context_events(self):
        self.compare_configuration(tagged=False, temporal=10000, cutoffs=(0,),
                                   weights=("RAW",), families=("features",))
        row = self.long("features", "CompoundAgnostic", tagged=False,
                        temporal=10000).loc[lambda frame: (frame.modifier == "red") &
                                            (frame["head"] == "car")].iloc[0]
        self.assertEqual(row.time, 0)
        self.assertEqual(row.comp_freq, 303)
        self.assertEqual(row.assoc_pair_freq, 303)

    def test_annotation_only_manifest_changes_do_not_repeat_preparation(self):
        args = self.args("prepare")
        prepared = self.google.prepare(args)
        before = {path: (self.digest(path), path.stat().st_mtime_ns)
                  for path in self.cache.rglob("*.parquet")}
        manifest_path = self.input / "corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["annotation_enrichment"] = {"ner": "annotation-only change"}
        manifest["source_info"]["parameters"]["annotation_availability"] = {"ner": "changed"}
        manifest_path.write_text(json.dumps(manifest))
        with mock.patch("duckdb.connect", side_effect=AssertionError("completed preparation reran SQL")):
            reused = self.google.prepare(args)
        self.assertEqual(reused, prepared)
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns) for path in before}, before)
        self.assert_inputs_unchanged()

    def test_frozen_vocabulary_hash_mismatch_is_refused(self):
        self.vocabulary.write_text(self.vocabulary.read_text().replace("\t0.3\t", "\t0.4\t"))
        with self.assertRaisesRegex(ValueError, "(?i)vocabulary hash"):
            self.google.prepare(self.args("prepare"))

    def test_feature_configuration_requires_completed_preparation(self):
        with self.assertRaisesRegex(RuntimeError, "(?i)preparation.*incomplete"):
            self.google.run_features(self.args("features", "--context-size", "3", "--temporal", "10"))

    def test_preparation_resumes_completed_count_batches_without_rewriting_them(self):
        args = self.args("prepare")
        original_write = self.google.write_json

        def interrupt_after_checkpoint(path, value):
            original_write(path, value)
            path = Path(path)
            if path.name == "complete.json" and path.parent.name == "batch-000000" \
                    and path.parent.parent.name == "bigram_counts":
                raise InterruptedError("synthetic interruption after a durable count checkpoint")

        with mock.patch.object(self.google, "write_json", side_effect=interrupt_after_checkpoint):
            with self.assertRaisesRegex(InterruptedError, "durable count checkpoint"):
                self.google.prepare(args)
        checkpoints = list(self.cache.glob("prepare-*/counts/bigram_counts/batch-000000/complete.json"))
        self.assertEqual(len(checkpoints), 1)
        completed_directory = checkpoints[0].parent
        paths = list(completed_directory.rglob("*.parquet"))
        paths += list(self.cache.glob("prepare-*/*_vectors.parquet"))
        self.assertTrue(paths)
        before = {path: (self.digest(path), path.stat().st_mtime_ns) for path in paths}
        prepared = self.google.prepare(args)
        self.assertTrue((prepared / "complete.json").is_file())
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns) for path in paths}, before)
        self.assert_inputs_unchanged()

    def test_preparation_resumes_completed_frequency_buckets_without_rewriting_them(self):
        args = self.args("prepare")
        original_write = self.google.write_json

        def interrupt_after_checkpoint(path, value):
            original_write(path, value)
            path = Path(path)
            if path.name == "bucket-00000.json" and path.parent.name == "width-10":
                raise InterruptedError("synthetic interruption after a durable frequency checkpoint")

        with mock.patch.object(self.google, "write_json", side_effect=interrupt_after_checkpoint):
            with self.assertRaisesRegex(InterruptedError, "durable frequency checkpoint"):
                self.google.prepare(args)
        paths = list(self.cache.glob("prepare-*/frequencies/width-10/bucket-00000.parquet"))
        paths += [path for directory in self.cache.glob("prepare-*/counts")
                  for path in directory.rglob("*.parquet")]
        self.assertTrue(paths)
        before = {path: (self.digest(path), path.stat().st_mtime_ns) for path in paths}
        prepared = self.google.prepare(args)
        self.assertTrue((prepared / "complete.json").is_file())
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns) for path in paths}, before)
        self.assert_inputs_unchanged()

    def test_completed_feature_configuration_is_reused_without_recomputation(self):
        self.google.prepare(self.args("prepare"))
        args = self.args("features", "--grid-index", "1", "--cutoffs", "0",
                         "--weights", "RAW", "--families", "features")
        completed = self.google.run_features(args)
        paths = [Path(record["path"]) for record in completed["outputs"]]
        before = {path: (self.digest(path), path.stat().st_mtime_ns) for path in paths}
        with mock.patch.object(self.google, "_run_configuration",
                               side_effect=AssertionError("completed features were recomputed")):
            self.assertEqual(self.google.run_features(args), completed)
        self.assertEqual({path: (self.digest(path), path.stat().st_mtime_ns) for path in paths}, before)
        self.assert_inputs_unchanged()

    def test_grid_indices_follow_context_major_temporal_minor_order(self):
        self.assertEqual([self.google._configuration(self.args("features", "--grid-index", str(index)))
                          for index in range(6)],
                         [(2, 10000), (2, 10), (2, 20), (3, 10000), (3, 10), (3, 20)])
        with self.assertRaisesRegex(ValueError, "outside"):
            self.google._configuration(self.args("features", "--grid-index", "6"))

    def test_tag_modes_have_distinct_caches_but_context_prefixes_share_preparation(self):
        tagged = self.google._specification(self.args("prepare"))
        untagged = self.google._specification(self.args("prepare", "--untagged"))
        smaller_prefix = self.google._specification(self.args("prepare", "--context-sizes", "2"))
        self.assertNotEqual(tagged["root"], untagged["root"])
        self.assertEqual(tagged["root"], smaller_prefix["root"])
        self.assertEqual(tagged["identity"]["temporal_bins"], [10, 20, 10000])
        self.assertEqual(tagged["identity"]["compound_population"], "adj_noun_and_noun_noun")
        self.assertEqual(untagged["identity"]["compound_population"], "all_lexical_bigrams")
        self.assertEqual(tagged["identity"]["version"], 3)

    def test_old_population_preparation_cannot_supply_untagged_features(self):
        args = self.args("prepare", "--untagged")
        new = self.google._specification(args)
        old = dict(new["identity"])
        old.pop("compound_population")
        old["version"] = 1
        incompatible = self.cache / f"prepare-{self.google._digest(old)[:24]}"
        incompatible.mkdir(parents=True)
        (incompatible / "complete.json").write_text(json.dumps(
            {"fingerprint": self.google._digest(old), "files": []}))
        self.assertNotEqual(incompatible, new["root"])
        with self.assertRaisesRegex(RuntimeError, "(?i)preparation.*incomplete"):
            self.google.run_features(self.args("features", "--untagged", "--context-size", "3",
                                              "--temporal", "10"))

    def test_tagged_ratings_expand_adjectives_without_pooling_noun_counterparts(self):
        tagged = self.google._specification(self.args("prepare"))
        untagged = self.google._specification(self.args("prepare", "--untagged"))
        tagged_pairs = set(zip(tagged["pairs"].modifier, tagged["pairs"]["head"]))
        self.assertIn(("red_ADJ", "car_NOUN"), tagged_pairs)
        self.assertIn(("red_NOUN", "car_NOUN"), tagged_pairs)
        self.assertIn(("car_NOUN", "red_NOUN"), tagged_pairs)
        self.assertNotIn(("car_ADJ", "red_NOUN"), tagged_pairs)
        red_ratings = tagged["ratings"].loc[tagged["ratings"]["head"] == "car_NOUN"]
        self.assertEqual(set(red_ratings.modifier), {"red_ADJ", "red_NOUN"})
        self.assertEqual(len(red_ratings), 6)  # Both POS variants retain all three rating sources.
        self.assertEqual(set(zip(untagged["pairs"].modifier, untagged["pairs"]["head"])),
                         {("red", "car"), ("car", "red"), ("ghost", "ship"),
                          ("zero", "vector"), ("jump", "rope")})
        self.pd.testing.assert_frame_equal(tagged["vocabulary"], untagged["vocabulary"])

    def test_configuration_refuses_unprepared_bin_or_vocabulary_above_extraction_ceiling(self):
        self.google.prepare(self.args("prepare"))
        with self.assertRaisesRegex(ValueError, "(?i)frozen extraction ceiling"):
            self.google.run_features(self.args("features", "--context-size", "7", "--temporal", "10"))
        with self.assertRaisesRegex(ValueError, "(?i)not included.*preparation"):
            self.google.run_features(self.args("features", "--context-size", "3", "--temporal", "50"))

    def test_preparation_rejects_context_grid_above_ceiling_before_any_sql(self):
        with mock.patch("duckdb.connect", side_effect=AssertionError("invalid grid started SQL")), \
                mock.patch.object(self.google, "partitioned_sum",
                                  side_effect=AssertionError("invalid grid started scans")):
            with self.assertRaisesRegex(ValueError, "(?i)extraction ceiling"):
                self.google.prepare(self.args("prepare", "--context-sizes", "2", "7"))
        self.assertFalse(self.cache.exists())
        self.assert_inputs_unchanged()

    def test_features_cli_requires_grid_or_complete_explicit_setting(self):
        for extra in ((), ("--context-size", "3"), ("--temporal", "10")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    self.args("features", *extra)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--grid-index", errors.getvalue())

    def test_empty_eligible_corpus_keeps_static_schema_and_wide_ratings(self):
        for name in ("bigrams", "bigram_counts"):
            frame = self.pd.read_parquet(self.input / f"{name}.parquet")
            eligible = frame.modifier.str.endswith(("_ADJ", "_NOUN")) & \
                frame["head"].str.endswith("_NOUN")
            self.write_table(name, frame.loc[~eligible].to_dict("records"))
        manifest_path = self.input / "corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name in ("bigrams", "bigram_counts"):
            path = self.input / f"{name}.parquet"
            manifest["tables"][name].update(bytes=path.stat().st_size, sha256=self.digest(path))
        manifest_path.write_text(json.dumps(manifest))
        self.input_bytes = {path: path.read_bytes() for path in self.input_bytes}
        self.google.prepare(self.args("prepare"))
        args = self.args("features", "--context-size", "3", "--temporal", "10",
                         "--cutoffs", "0")
        self.google.run_features(args)
        ratings = self.shared_driver.read_ratings(SimpleNamespace(
            tag=True, **{name: str(path) for name, path in self.ratings.items()}))
        rating_keys = ["modifier", "head", "source", "is_adj", "rating"]
        expected_columns = {"modifier", "head", "time", "sim_with_modifier", "sim_with_head",
                            "sim_bw_constituents", "beta", "geom_mean_sim", "arith_mean_sim",
                            "sim_cpf_0", "sim_cpf_25", "sim_cpf_50", "sim_cpf_75",
                            "sim_cpf_100", "sim_cpf_beta", "comp_freq", "mod_freq", "head_freq",
                            "mod_prod", "head_prod", "assoc_pair_freq", "assoc_left_freq",
                            "assoc_right_freq", "assoc_population_size", "ppmi", "local_mi",
                            "log_ratio"}
        for weight in ("RAW", "PPMI"):
            for condition in ("CompoundAware", "CompoundPooled", "CompoundAgnostic"):
                with self.subTest(weight=weight, condition=condition):
                    static = self.long("features", condition, weight)
                    self.assertTrue(static.empty)
                    self.assertTrue(expected_columns.issubset(static.columns), static.columns)
                    if condition == "CompoundAware":
                        self.assertTrue({"sim_bw_settings_modifier", "perc_token_mod", "perc_type_mod",
                                         "sim_bw_settings_head", "perc_token_head", "perc_type_head"}
                                        .issubset(static.columns))
                    elif condition == "CompoundPooled":
                        for suffix in ("mod", "head"):
                            self.assertTrue({f"sim_bw_settings_constituent_{suffix}",
                                             f"perc_token_constituent_{suffix}",
                                             f"perc_type_constituent_{suffix}"}.issubset(static.columns))
                    temporal = self.long("temporal", condition, weight)
                    self.assertTrue(temporal.empty)
                    self.assertEqual(set(temporal.columns),
                                     {"modifier", "head", "time", "change_comp", "change_mod", "change_head"})
                    for family in ("features", "temporal"):
                        path = self.output / (f"{family}_{condition}_withSetting_{weight}_"
                                              "Tagged_ctx3_10_0_na.parquet")
                        wide = self.pd.read_parquet(path)
                        self.pd.testing.assert_frame_equal(
                            wide.sort_values(rating_keys).reset_index(drop=True),
                            ratings.sort_values(rating_keys).reset_index(drop=True),
                            check_dtype=False, check_like=True)
        self.assert_inputs_unchanged()


@unittest.skipUnless(
    all((GOOGLE_DIRECTORY / "slurm" / name).is_file() for name in ("run_features.sh", "run_features.slurm")),
    "Optional local batch helpers are not distributed")
class GoogleFeatureSlurmTests(unittest.TestCase):
    """Mock every job submission, environment change, and Python invocation."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "conda/bin").mkdir(parents=True)
        (self.root / "conda/bin/activate").write_text("")
        (self.root / "project").mkdir(parents=True)
        (self.root / "project/features.py").write_text("")

    def environment(self, overrides):
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith(("SLURM_", "FEATURE_")) or name in {
                    "MODE", "SUBMIT_MODE", "PROJECT_ROOT", "CONDA_ROOT", "CONDA_ENV",
                    "CONTEXT_SIZES", "TEMPORAL_BINS", "CUTOFFS", "TARGET_TAGGED", "TAG_MODES",
                    "RATINGS_DIR", "CONTEXT_VOCABULARY", "CACHE_DIRECTORY", "TEMP_DIRECTORY",
                    "DUCKDB_MEMORY_LIMIT", "REDDY90", "CORDEIRO90", "CORDEIRO100",
                    "PREPARE_JOB_ID", "PARTITIONS", "PARTITION_BATCH_ROWS", "ARROW_BATCH_ROWS"}:
                environment.pop(name, None)
        environment.update({"PROJECT_ROOT": str(self.root / "project"), "PYTHON_BIN": "python",
                            "CONDA_ROOT": str(self.root / "conda"),
                            "FEATURE_INPUT": str(self.root / "inputs"),
                            "FEATURE_OUTPUT": str(self.root / "output"),
                            "FEATURE_LOGS": str(self.root / "logs"),
                            "FEATURE_SLURM_SCRIPT": str(GOOGLE_DIRECTORY / "slurm" / "run_features.slurm"),
                            "RATINGS_DIR": str(self.root / "ratings"),
                            "CONTEXT_VOCABULARY": str(self.root / "contexts.tsv"),
                            "CACHE_DIRECTORY": str(self.root / "cache"),
                            "TEMP_DIRECTORY": str(self.root / "spill"),
                            "MOCK_SBATCH_CALLS": str(self.root / "sbatch-calls"),
                            **{name: str(value) for name, value in overrides.items()}})
        return environment

    @staticmethod
    def decode_calls(text):
        return [line.rstrip("\0").split("\0") for line in text.splitlines() if line]

    @staticmethod
    def option_values(arguments, option):
        start = arguments.index(option) + 1
        stop = next((index for index in range(start, len(arguments))
                     if arguments[index].startswith("--")), len(arguments))
        return arguments[start:stop]

    def run_wrapper(self, **overrides):
        script = """
module() { return 0; }
conda() { return 0; }
python() { printf '%s\\0' "$@"; printf '\\n'; }
source "$1"
"""
        result = subprocess.run(["bash", "-c", script, "mock-feature-wrapper",
                                 str(GOOGLE_DIRECTORY / "slurm" / "run_features.slurm")],
                                env=self.environment(overrides), text=True,
                                capture_output=True, timeout=30, check=False)
        return result, self.decode_calls(result.stdout)

    def run_helper(self, arguments=(), **overrides):
        script = """
sbatch() {
    [[ -d "$FEATURE_LOGS" && -d "$FEATURE_OUTPUT" ]] || return 91
    printf 'mock-grid:%s|%s\\n' "$CONTEXT_SIZES" "$TEMPORAL_BINS" >&2
    printf '%s\\0' "$@" >> "$MOCK_SBATCH_CALLS"
    printf '\\n' >> "$MOCK_SBATCH_CALLS"
    if [[ "$*" == *'MODE=prepare'* && "$*" == *'TARGET_TAGGED=0'* ]]; then printf '12346;cluster\\n';
    elif [[ "$*" == *'MODE=prepare'* ]]; then printf '12345;cluster\\n';
    else printf '67890\\n'; fi
}
source "$1" "${@:2}"
"""
        result = subprocess.run(["bash", "-c", script, "mock-feature-submission",
                                 str(GOOGLE_DIRECTORY / "slurm" / "run_features.sh"), *arguments],
                                env=self.environment(overrides), text=True,
                                capture_output=True, timeout=30, check=False)
        path = self.root / "sbatch-calls"
        return result, self.decode_calls(path.read_text()) if path.exists() else []

    def test_default_wrapper_prepares_once_with_full_grid_and_resource_options(self):
        result, calls = self.run_wrapper(SLURM_CPUS_PER_TASK=2, DUCKDB_MEMORY_LIMIT="8GB")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "prepare")
        self.assertEqual(self.option_values(calls[0], "--context-sizes"),
                         ["1000", "3000", "5000", "10000", "30000", "50000"])
        self.assertEqual(self.option_values(calls[0], "--temporal-bins"),
                         ["10000", "10", "20", "50", "100"])
        self.assertEqual(self.option_values(calls[0], "--threads"), ["2"])
        self.assertEqual(self.option_values(calls[0], "--memory-limit"), ["8GB"])
        self.assertNotIn("--untagged", calls[0])

    def test_feature_array_prioritizes_slurm_index_and_emits_complete_science_grid(self):
        result, calls = self.run_wrapper(MODE="features", SLURM_ARRAY_TASK_ID=7, FEATURE_INDEX=2,
                                         TARGET_TAGGED=0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0][2], "features")
        self.assertEqual(self.option_values(calls[0], "--grid-index"), ["7"])
        self.assertIn("--untagged", calls[0])
        self.assertEqual(self.option_values(calls[0], "--weights"), ["RAW", "PPMI"])
        self.assertEqual(self.option_values(calls[0], "--families"), ["features", "temporal"])
        self.assertEqual(self.option_values(calls[0], "--conditions"),
                         ["CompoundAware", "CompoundPooled", "CompoundAgnostic"])
        self.assertEqual(self.option_values(calls[0], "--cutoffs"), ["0", "10", "50", "100", "500", "1000"])

    def test_explicit_wrapper_setting_and_incompatible_array_are_checked(self):
        result, calls = self.run_wrapper(MODE="features", FEATURE_CONTEXT_SIZE=3000,
                                         FEATURE_TEMPORAL_BIN=20, TARGET_TAGGED=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.option_values(calls[0], "--context-size"), ["3000"])
        self.assertEqual(self.option_values(calls[0], "--temporal"), ["20"])
        self.assertNotIn("--grid-index", calls[0])
        result, calls = self.run_wrapper(MODE="features", SLURM_ARRAY_TASK_ID=0,
                                         FEATURE_CONTEXT_SIZE=3000, FEATURE_TEMPORAL_BIN=20,
                                         TARGET_TAGGED=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit settings", result.stderr)
        self.assertEqual(calls, [])

    def test_preparation_array_is_rejected_without_running_python(self):
        result, calls = self.run_wrapper(SLURM_ARRAY_TASK_ID=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preparation must run once", result.stderr)
        self.assertEqual(calls, [])

    def test_submission_serializes_two_preparations_before_sixty_task_array(self):
        result, calls = self.run_helper(("submit", "--mem=32G"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        for call, tagged in zip(calls[:2], ("1", "0")):
            exports = next(option for option in call if option.startswith("--export="))
            self.assertIn("MODE=prepare", exports)
            self.assertIn(f"TARGET_TAGGED={tagged}", exports)
            self.assertNotIn("--array", call)
        self.assertNotIn("--dependency", calls[0])
        self.assertEqual(self.option_values(calls[1], "--dependency"), ["afterok:12345"])
        exports = next(option for option in calls[2] if option.startswith("--export="))
        self.assertIn("MODE=features", exports)
        self.assertNotIn("TARGET_TAGGED=", exports)
        self.assertEqual(self.option_values(calls[2], "--dependency"), ["afterok:12345:12346"])
        self.assertEqual(self.option_values(calls[2], "--array"), ["0-59%2"])
        self.assertTrue((self.root / "logs").is_dir())
        for call in calls:
            self.assertIn("--mem=32G", call)
            self.assertEqual(self.option_values(call, "--output"),
                             [str(self.root / "logs/features_%A_%a.out")])

    def test_custom_submission_grid_and_existing_preparation_dependency(self):
        result, calls = self.run_helper(("features",), CONTEXT_SIZES="1000 3000",
                                        TEMPORAL_BINS="10 20", FEATURE_ARRAY_CONCURRENCY=1,
                                        PREPARE_JOB_ID=112233)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.option_values(calls[0], "--array"), ["0-7%1"])
        self.assertEqual(self.option_values(calls[0], "--dependency"), ["afterok:112233"])

    def test_invalid_submission_grid_never_reaches_sbatch(self):
        result, calls = self.run_helper(("submit",), CONTEXT_SIZES="1000 nope")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive integers", result.stderr)
        self.assertEqual(calls, [])

    def test_duplicate_submission_grid_is_deduplicated_in_order_for_both_stages(self):
        result, calls = self.run_helper(("submit",), CONTEXT_SIZES="1000 1000 5000",
                                        TEMPORAL_BINS="10 20 10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.option_values(calls[2], "--array"), ["0-7%2"])
        self.assertEqual(result.stderr.count("mock-grid:1000 5000|10 20\n"), 3)

    def test_default_feature_array_maps_tagged_major_index_boundaries(self):
        for combined_index, tagged, local_index in ((0, True, 0), (29, True, 29),
                                                    (30, False, 0), (59, False, 29)):
            with self.subTest(combined_index=combined_index):
                result, calls = self.run_wrapper(MODE="features",
                                                 SLURM_ARRAY_TASK_ID=combined_index)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 1)
                self.assertEqual(self.option_values(calls[0], "--grid-index"), [str(local_index)])
                self.assertEqual("--untagged" not in calls[0], tagged)
        for combined_index in (60, -1, "not-an-index"):
            with self.subTest(combined_index=combined_index):
                result, calls = self.run_wrapper(MODE="features",
                                                 SLURM_ARRAY_TASK_ID=combined_index)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])

    def test_single_tag_mode_overrides_retain_thirty_task_grid(self):
        for overrides, tagged in (({"TARGET_TAGGED": 1}, True), ({"TARGET_TAGGED": 0}, False),
                                  ({"TAG_MODES": "Tagged"}, True),
                                  ({"TAG_MODES": "UnTagged"}, False)):
            with self.subTest(overrides=overrides):
                path = self.root / "sbatch-calls"
                path.unlink(missing_ok=True)
                result, calls = self.run_helper(("submit",), **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 2)
                exports = next(option for option in calls[0] if option.startswith("--export="))
                self.assertIn(f"TARGET_TAGGED={int(tagged)}", exports)
                self.assertEqual(self.option_values(calls[1], "--array"), ["0-29%2"])
                result, calls = self.run_wrapper(MODE="features", SLURM_ARRAY_TASK_ID=29,
                                                 **overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.option_values(calls[0], "--grid-index"), ["29"])
                self.assertEqual("--untagged" not in calls[0], tagged)

    def test_prepare_only_submits_both_modes_serially(self):
        result, calls = self.run_helper(("prepare",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--dependency", calls[0])
        self.assertEqual(self.option_values(calls[1], "--dependency"), ["afterok:12345"])
        self.assertTrue(all("--array" not in call for call in calls))

    def test_tag_modes_are_ordered_deduplicated_for_submission_and_wrapper(self):
        result, calls = self.run_helper(("features",), TAG_MODES="UnTagged Tagged UnTagged",
                                        CONTEXT_SIZES="1000", TEMPORAL_BINS="10 20")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.option_values(calls[0], "--array"), ["0-3%2"])
        for index, tagged, local_index in ((0, False, 0), (1, False, 1),
                                            (2, True, 0), (3, True, 1)):
            result, calls = self.run_wrapper(MODE="features", SLURM_ARRAY_TASK_ID=index,
                                             TAG_MODES="UnTagged Tagged UnTagged",
                                             CONTEXT_SIZES="1000", TEMPORAL_BINS="10 20")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.option_values(calls[0], "--grid-index"), [str(local_index)])
            self.assertEqual("--untagged" not in calls[0], tagged)

    def test_invalid_tag_modes_do_not_run_python_or_submit(self):
        for overrides in ({"TAG_MODES": "Tagged Unknown"}, {"TAG_MODES": ""},
                          {"TARGET_TAGGED": 2}, {"TARGET_TAGGED": ""}):
            with self.subTest(overrides=overrides):
                path = self.root / "sbatch-calls"
                path.unlink(missing_ok=True)
                result, calls = self.run_helper(("submit",), **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])
                result, calls = self.run_wrapper(MODE="features", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])

    def test_conflicting_helper_tag_settings_are_rejected_before_submission(self):
        result, calls = self.run_helper(("submit",), TARGET_TAGGED=1, TAG_MODES="Tagged")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_explicit_wrapper_configuration_requires_a_single_tag_mode(self):
        result, calls = self.run_wrapper(MODE="features", FEATURE_CONTEXT_SIZE=1000,
                                         FEATURE_TEMPORAL_BIN=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])
        result, calls = self.run_wrapper(MODE="features", FEATURE_CONTEXT_SIZE=1000,
                                         FEATURE_TEMPORAL_BIN=10, TAG_MODES="UnTagged")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--untagged", calls[0])


if __name__ == "__main__":
    unittest.main()

