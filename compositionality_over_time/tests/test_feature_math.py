"""Sparse Google calculations agree with the corrected shared COHA layer."""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

GOOGLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOOGLE))
sys.path.insert(0, str(GOOGLE / "feature_helpers"))

import feature_math as math
from coha_features import (PAIR, cosine_features, ppmi, setting_comparison,
                           condition_matrices, condition_counts)


def frame_for(matrix, keys):
    """Row i maps to modifier m_i/head h_i/time 2000 or word w_i."""
    matrix = sparse.coo_matrix(matrix)
    rows = []
    for row, column, count in zip(matrix.row, matrix.col, matrix.data):
        values = {"modifier": f"m{row}", "head": f"h{row}",
                  "word": f"w{row}", "time": 2000}
        rows.append([*[values[key] for key in keys], f"c{column}", float(count)])
    return pd.DataFrame(rows, columns=[*keys, "context", "count"])


def reference_static(compounds, modifiers, heads):
    evaluation = pd.DataFrame([[f"m{row}", f"h{row}", 2000]
                               for row in range(compounds.shape[0])], columns=PAIR)
    return cosine_features(frame_for(compounds, PAIR),
                           frame_for(modifiers, ["modifier", "time"]),
                           frame_for(heads, ["head", "time"]),
                           evaluation).drop(columns=PAIR)


class SparseWeightingTests(unittest.TestCase):
    def test_ppmi_complete_background_before_selected_targets(self):
        population = sparse.csr_matrix([[30, 10, 0], [10, 30, 0], [0, 0, 100]])
        selected = population[:1]
        before = selected.copy()
        result = math.weight_matrix(selected, np.asarray(population.sum(axis=0)).ravel(), ppmi=True)
        expected = ppmi(frame_for(population, ["word", "time"]))
        expected = expected.loc[expected.word == "w0"]
        values = np.zeros((1, population.shape[1]))
        for row in expected.itertuples(index=False):
            values[0, int(row.context[1:])] = row.count
        np.testing.assert_allclose(result.toarray(), values, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(selected.toarray(), before.toarray())
        self.assertGreater(result[0, 1], 0)  # Different from restricted-population PPMI.

    def test_weighting_context_selection_changes_population(self):
        population = sparse.csr_matrix([[30, 10, 1000], [10, 30, 0]])
        limited = population[:, :2]
        background = np.asarray(limited.sum(axis=0)).ravel()
        result = math.weight_matrix(limited, background, ppmi=True)
        np.testing.assert_allclose(result.toarray(), [[np.log2(1.5), 0], [0, np.log2(1.5)]])

    def test_weighting_empty_zero_duplicates_and_raw_copy(self):
        empty = sparse.csr_matrix((3, 0))
        self.assertEqual(math.weight_matrix(empty, [], ppmi=True).shape, (3, 0))
        zeros = sparse.csr_matrix([[0, 0], [0, 0]])
        self.assertEqual(math.weight_matrix(zeros, [0, 0], ppmi=True).nnz, 0)
        duplicated = sparse.csr_matrix((np.array([2., 3., 0.]),
                                        np.array([0, 0, 1]), np.array([0, 3])), shape=(1, 2))
        self.assertFalse(duplicated.has_canonical_format)
        result = math.weight_matrix(duplicated, [5, 0])
        np.testing.assert_array_equal(result.toarray(), [[5, 0]])
        self.assertEqual(duplicated.nnz, 3)
        result.data[:] = 99
        np.testing.assert_array_equal(duplicated.data, [2, 3, 0])

    def test_background_validation(self):
        matrix = sparse.csr_matrix([[1, 0]])
        for background in ([1], [[1, 0]], [-1, 0], [np.nan, 1]):
            with self.assertRaises(ValueError):
                math.weight_matrix(matrix, background, ppmi=True)
        with self.assertRaisesRegex(ValueError, "positive context"):
            math.weight_matrix(matrix, [0, 1], ppmi=True)
        with self.assertRaisesRegex(ValueError, "positive finite"):
            math.weight_matrix(matrix, [0, 0], ppmi=True)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            math.weight_matrix(sparse.csr_matrix([[-1, 0]]), [1, 1], ppmi=True)


class SparseCosineTests(unittest.TestCase):
    def test_full_support_zero_missing_disjoint(self):
        left = sparse.csr_matrix([[1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])
        right = sparse.csr_matrix([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(math.cosine_rows(left, right),
                                   [1 / np.sqrt(2), 0, np.nan, np.nan], equal_nan=True)
        with self.assertRaisesRegex(ValueError, "identical"):
            math.cosine_rows(left, right[:, :2])

    def test_static_reference_missing_and_analytic_head_weight(self):
        compounds = sparse.csr_matrix([[1, 0], [0, 0], [1, 0], [1, 0], [1, 0]])
        modifiers = sparse.csr_matrix([[1, 0], [1, 0], [0, 0], [1, 1], [1, 0]])
        heads = sparse.csr_matrix([[0, 1], [1, 0], [1, 0], [1, 1], [1, 0]])
        result = math.static_features(compounds, modifiers, heads)
        reference = reference_static(compounds, modifiers, heads)
        pd.testing.assert_frame_equal(result, reference, rtol=1e-12, atol=1e-12)
        self.assertEqual(result.iloc[0].beta, 0)
        self.assertEqual(result.iloc[0].sim_cpf_beta, 1)
        self.assertEqual(result.iloc[4].beta, .5)
        self.assertTrue(result.iloc[2].filter(like="sim_cpf_").isna().all())

    def test_random_static_matches_coha_and_preserves_inputs(self):
        rng = np.random.default_rng(5842)
        for _ in range(8):
            matrices = []
            for _ in range(3):
                values = rng.integers(0, 100, size=(40, 24)).astype(float)
                values[rng.random(values.shape) < .85] = 0
                matrices.append(sparse.csr_matrix(values))
            before = [matrix.copy() for matrix in matrices]
            result = math.static_features(*matrices)
            reference = reference_static(*matrices)
            pd.testing.assert_frame_equal(result, reference, rtol=1e-11, atol=1e-12)
            for original, saved in zip(matrices, before):
                np.testing.assert_array_equal(original.data, saved.data)
                np.testing.assert_array_equal(original.indices, saved.indices)
                np.testing.assert_array_equal(original.indptr, saved.indptr)

    def test_analytic_beta_is_clamped_at_both_endpoints(self):
        compounds = sparse.csr_matrix([[0, 1], [0, 1]])
        modifiers = sparse.csr_matrix([[1, 0], [1, 1]])
        heads = sparse.csr_matrix([[1, 1], [1, 0]])
        result = math.static_features(compounds, modifiers, heads)
        np.testing.assert_array_equal(result.beta, [1, 0])
        pd.testing.assert_frame_equal(result, reference_static(compounds, modifiers, heads),
                                      rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result.sim_cpf_beta, [1 / np.sqrt(2)] * 2)
        self.assertEqual(result.iloc[0].sim_cpf_beta, result.iloc[0].sim_cpf_100)
        self.assertEqual(result.iloc[1].sim_cpf_beta, result.iloc[1].sim_cpf_0)

    def test_bounded_beta_matches_grid_optimum_for_nonnegative_vectors(self):
        rng = np.random.default_rng(17032)
        # Include sparse/disjoint support along with dense vectors.
        matrices = []
        for _ in range(3):
            values = rng.random((70, 8))
            values[rng.random(values.shape) < .6] = 0
            values[:, 0] += .01  # Guarantee nonzero vectors.
            matrices.append(sparse.csr_matrix(values))
        result = math.static_features(*matrices)
        self.assertTrue(result.beta.between(0, 1).all())
        weights = np.linspace(0, 1, 8193)
        for row in result.itertuples(index=False):
            grid_similarities = math.composition_cosine_from_cosines(
                np.full(weights.shape, row.sim_with_modifier),
                np.full(weights.shape, row.sim_with_head),
                np.full(weights.shape, row.sim_bw_constituents), weights)
            self.assertGreaterEqual(row.sim_cpf_beta + 1e-12, grid_similarities.max())
            self.assertLess(row.sim_cpf_beta - grid_similarities.max(), 1e-7)

    def test_beta_helpers_preserve_degeneracy_missingness_and_input_arrays(self):
        cm = np.array([1., 0., np.nan, .3, .4])
        ch = np.array([1., 0., .7, .3, .2])
        mh = np.array([1., .4, np.nan, 1. - 1e-14, .2])
        saved = [values.copy() for values in (cm, ch, mh)]
        beta = math.bounded_beta_from_cosines(cm, ch, mh)
        np.testing.assert_array_equal(beta[:4], [.5] * 4)
        self.assertAlmostEqual(beta[4], .25)
        composed = math.composition_cosine_from_cosines(cm, ch, mh, beta)
        self.assertEqual(composed[0], 1)
        self.assertEqual(composed[1], 0)
        self.assertTrue(np.isnan(composed[2]))
        # Endpoint composition still requires both constituent vectors.
        self.assertTrue(np.isnan(math.composition_cosine_from_cosines(
            [np.nan], [.7], [np.nan], 1)[0]))
        for values, before in zip((cm, ch, mh), saved):
            np.testing.assert_array_equal(values, before)
        self.assertEqual(math.bounded_beta_from_cosines(.2, .4, .2).shape, ())
        self.assertEqual(math.bounded_beta_from_cosines([], [], []).shape, (0,))

    def test_beta_helpers_validate_shapes_and_allow_nans(self):
        for helper in (math.bounded_beta_from_cosines,
                       lambda cm, ch, mh: math.composition_cosine_from_cosines(cm, ch, mh, .5)):
            with self.assertRaisesRegex(ValueError, "identical shapes"):
                helper([.2], [[.3]], [.4])
            with self.assertRaisesRegex(ValueError, "finite values or NaN"):
                helper([np.inf], [.3], [.4])
        with self.assertRaisesRegex(ValueError, "scalar or match"):
            math.composition_cosine_from_cosines([.2], [.3], [.4], [.5, .5])
        with self.assertRaisesRegex(ValueError, "finite values or NaN"):
            math.composition_cosine_from_cosines([.2], [.3], [.4], np.inf)
        self.assertTrue(np.isnan(math.composition_cosine_from_cosines(
            [.2], [.3], [.4], np.nan)[0]))

    def test_clipping_changes_only_beta_dependent_features(self):
        compounds = sparse.csr_matrix([[0, 1], [1, 0], [0, 0]])
        modifiers = sparse.csr_matrix([[1, 0], [1, 1], [1, 0]])
        heads = sparse.csr_matrix([[1, 1], [1, 0], [1, 1]])
        result = math.static_features(compounds, modifiers, heads)
        cm = math.cosine_rows(compounds, modifiers)
        ch = math.cosine_rows(compounds, heads)
        mh = math.cosine_rows(modifiers, heads)
        expected = {
            "sim_with_modifier": cm, "sim_with_head": ch,
            "sim_bw_constituents": mh, "geom_mean_sim": np.sqrt(cm * ch),
            "arith_mean_sim": (cm + ch) / 2.0,
        }
        # Independent normalized sparse composition verifies all fixed weights.
        for suffix, weight in zip(("0", "25", "50", "75", "100"),
                                  (0., .25, .5, .75, 1.)):
            mod_unit = modifiers.multiply(1 / np.sqrt(np.asarray(modifiers.power(2).sum(axis=1))))
            head_unit = heads.multiply(1 / np.sqrt(np.asarray(heads.power(2).sum(axis=1))))
            expected[f"sim_cpf_{suffix}"] = math.cosine_rows(
                compounds, weight * head_unit + (1 - weight) * mod_unit)
        expected = pd.DataFrame(expected)
        pd.testing.assert_frame_equal(result[expected.columns], expected,
                                      rtol=1e-12, atol=1e-12)
        self.assertEqual(result.iloc[2].beta, .5)
        self.assertTrue(np.isnan(result.iloc[2].sim_cpf_beta))

    def test_large_sparse_dimensions_and_empty_rows_do_not_densify(self):
        matrix = sparse.csr_matrix((np.array([2., 3., 4.]),
                                    (np.array([1, 10000, 19999]), np.array([99999, 1, 2]))),
                                   shape=(20000, 100000))
        result = math.cosine_rows(matrix, matrix)
        self.assertEqual(np.isfinite(result).sum(), 3)
        np.testing.assert_array_equal(result[[1, 10000, 19999]], [1, 1, 1])

    def test_chunk_boundary_inside_row_keeps_full_norm(self):
        columns = math._ENTRY_BATCH + 37
        matrix = sparse.csr_matrix(np.ones((1, columns)))
        other = sparse.csr_matrix(([1.], ([0], [0])), shape=matrix.shape)
        np.testing.assert_allclose(math.cosine_rows(matrix, other), [1 / np.sqrt(columns)])

    def test_temporal_passthrough_missing_previous_stays_nan(self):
        current = sparse.csr_matrix([[1, 0], [1, 1]])
        previous = sparse.csr_matrix([[0, 0], [1, 0]])
        np.testing.assert_allclose(math.temporal_cosines(current, previous),
                                   [np.nan, 1 / np.sqrt(2)], equal_nan=True)

    def test_empty_static_schema(self):
        matrix = sparse.csr_matrix((0, 10))
        result = math.static_features(matrix, matrix, matrix)
        self.assertEqual(list(result), list(math.STATIC_COLUMNS))
        self.assertTrue(result.empty)


class SparseSettingTests(unittest.TestCase):
    def test_raw_types_before_ppmi_and_raw_occurrences_not_event_sums(self):
        raw = sparse.csr_matrix([[10, 10], [0, 0], [10, 0]])
        weighted = sparse.csr_matrix([[1, 0], [0, 0], [0, 0]])
        general = sparse.csr_matrix([[1, 1], [1, 0], [0, 0]])
        result = math.setting_features(weighted, general, raw, general,
                                       [2, 0, 1], [1, 4, 0], "modifier")
        self.assertEqual(result.iloc[0].perc_token_mod, 2)
        self.assertEqual(result.iloc[0].perc_type_mod, 1)
        self.assertAlmostEqual(result.iloc[0].sim_bw_settings_modifier, 1 / np.sqrt(2))
        self.assertEqual(result.iloc[1].perc_token_mod, 0)
        self.assertEqual(result.iloc[1].perc_type_mod, 0)
        self.assertTrue(result.iloc[2].isna().all())
        unweighted = math.setting_features(raw, general, raw, general,
                                           [2, 0, 1], [1, 4, 0], "modifier")
        pd.testing.assert_frame_equal(result.drop(columns="sim_bw_settings_modifier"),
                                      unweighted.drop(columns="sim_bw_settings_modifier"))

    def test_setting_comparison_matches_coha(self):
        role = sparse.csr_matrix([[3, 0, 0], [0, 0, 0], [1, 0, 1]])
        general = sparse.csr_matrix([[3, 2, 0], [0, 2, 0], [0, 0, 0]])
        raw_role = sparse.csr_matrix([[7, 0, 8], [0, 0, 0], [2, 0, 1]])
        raw_general = sparse.csr_matrix([[9, 9, 9], [0, 10, 0], [0, 0, 0]])
        role_counts = np.array([4, 0, 1])
        general_counts = np.array([2, 7, np.nan])
        for label in ["modifier", "head", "constituent"]:
            result = math.setting_features(role, general, raw_role, raw_general,
                                           role_counts, general_counts, label)
            targets = pd.DataFrame({"word": ["w0", "w1", "w2"], "time": [2000] * 3})
            counts = targets.assign(count=role_counts)
            counts = counts.loc[counts["count"] > 0]
            general_occurrences = targets.assign(count=general_counts)
            expected = setting_comparison(frame_for(role, ["word", "time"]),
                                          frame_for(general, ["word", "time"]),
                                          frame_for(raw_role, ["word", "time"]),
                                          frame_for(raw_general, ["word", "time"]),
                                          counts, general_occurrences, "word", targets, label)
            pd.testing.assert_frame_equal(result, expected.drop(columns=["word", "time"]),
                                          rtol=1e-12, atol=1e-12)

    def test_pooled_self_pairs_count_both_roles(self):
        compounds = pd.DataFrame([["a", "a", 2000, "x", 3],
                                  ["a", "b", 2000, "y", 4]], columns=PAIR + ["context", "count"])
        counts = pd.DataFrame([["a", "a", 2000, 2], ["a", "b", 2000, 5]],
                              columns=PAIR + ["count"])
        words = pd.DataFrame([["a", 2000, "x", 1], ["a", 2000, "y", 1]],
                             columns=["word", "time", "context", "count"])
        word_counts = pd.DataFrame([["a", 2000, 3], ["b", 2000, 5]],
                                   columns=["word", "time", "count"])
        pooled, _ = condition_matrices(compounds, words)["CompoundPooled"]
        selected = pooled.loc[pooled.modifier == "a"].set_index("context")["count"]
        np.testing.assert_array_equal(selected.reindex(["x", "y"]), [6, 4])
        occurrence, _ = condition_counts(counts, word_counts)["CompoundPooled"]
        self.assertEqual(occurrence.set_index("modifier").loc["a", "count"], 9)
        raw = sparse.csr_matrix([selected.reindex(["x", "y"]).to_numpy()])
        general = sparse.csr_matrix([[1, 1]])
        result = math.setting_features(raw, general, raw, general, [9], [3], "constituent")
        self.assertEqual(result.iloc[0].perc_token_constituent, 3)
        self.assertEqual(result.iloc[0].perc_type_constituent, 1)

    def test_count_and_type_ratio_unknowns_and_validation(self):
        role = sparse.csr_matrix([[1, 2, 3], [0, 0, 0]])
        general = sparse.csr_matrix([[1, 0, 0], [1, 0, 0]])
        result = math.setting_features(role, general, role, general,
                                       [np.nan, 0], [4, np.nan], "head")
        self.assertEqual(result.iloc[0].perc_token_head, 0)
        self.assertEqual(result.iloc[0].perc_type_head, 3)
        self.assertTrue(np.isnan(result.iloc[1].perc_token_head))
        self.assertEqual(result.iloc[1].perc_type_head, 0)
        with self.assertRaisesRegex(ValueError, "one occurrence"):
            math.setting_features(role, general, role, general, [1], [1, 1], "head")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            math.setting_features(role, general, role, general, [-1, 0], [1, 1], "head")


if __name__ == "__main__":
    unittest.main()
