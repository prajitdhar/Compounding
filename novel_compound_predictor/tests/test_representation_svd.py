from contextlib import contextmanager
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

MODULE_PATH = Path(__file__).resolve().parents[1] / "representation_svd.py"
spec = importlib.util.spec_from_file_location("representation_svd", MODULE_PATH)
svd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svd)


class RecordingReporter:
    def __init__(self):
        self.events = []
        self.stages = []
        self.updates = []
        self.active = 0

    def log(self, event, **metadata):
        self.events.append((event, metadata))

    @contextmanager
    def stage(self, name, total=None):
        self.stages.append((name, total))
        self.active += 1
        try:
            yield self
        finally:
            self.active -= 1

    def update(self, detail, completed=None, total=None):
        if not self.active:
            raise AssertionError("Progress update outside stage")
        self.updates.append((detail, completed, total))


class StreamingSVDTests(unittest.TestCase):
    def well_conditioned_low_rank(self):
        rng = np.random.default_rng(4)
        left, _ = np.linalg.qr(rng.normal(size=(401, 7)))
        right, _ = np.linalg.qr(rng.normal(size=(61, 7)))
        singular_values = np.arange(9, 2, -1, dtype=float)
        dense = (left * singular_values) @ right.T
        return dense, singular_values

    def test_matches_direct_svd_projection_on_well_conditioned_low_rank(self):
        dense, expected_values = self.well_conditioned_low_rank()
        model = svd.fit_streaming_svd(sparse.csr_matrix(dense), 7,
                                      n_iter=2, seed=1991, batch_size=29)
        np.testing.assert_allclose(model.singular_values_, expected_values, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(7), atol=1e-12)
        projected = model.transform(sparse.csr_matrix(dense))
        np.testing.assert_allclose(projected @ model.components_, dense, atol=1e-12)
        np.testing.assert_allclose(model.explained_variance_, projected.var(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.explained_variance_ratio_,
                                   projected.var(axis=0) / dense.var(axis=0).sum(), rtol=1e-12, atol=1e-12)

    def test_every_training_product_and_slice_is_row_bounded(self):
        batch_size = 13
        slices = []

        class GuardedCSR(sparse.csr_matrix):
            def __getitem__(self, key):
                if isinstance(key, slice):
                    length = min(key.stop, self.shape[0]) - key.start
                    if length > batch_size:
                        raise AssertionError("Full-corpus row slice")
                    slices.append(length)
                return super().__getitem__(key)

            def __matmul__(self, other):
                if self.shape[0] > batch_size:
                    raise AssertionError("Full-corpus dense product")
                return super().__matmul__(other)

        dense, _ = self.well_conditioned_low_rank()
        reporter = RecordingReporter()
        model = svd.fit_streaming_svd(GuardedCSR(dense.astype(np.float32)), 7,
                                      n_iter=2, batch_size=batch_size, reporter=reporter)
        self.assertEqual(model.components_.shape, (7, dense.shape[1]))
        self.assertTrue(slices)
        self.assertLessEqual(max(slices), batch_size)
        self.assertEqual(len(reporter.stages), 4)
        self.assertTrue(reporter.updates)
        self.assertEqual(reporter.updates[-1][1:], (dense.shape[0], dense.shape[0]))

    def test_sparse_variance_counts_implicit_zeros(self):
        dense = np.array([[0, 1, 0, 0], [2, 0, 0, 4], [0, 0, 0, 0],
                          [3, 0, 5, 0], [0, 1, 0, 2]], dtype=np.float32)
        model = svd.fit_streaming_svd(sparse.csr_matrix(dense), 4, n_iter=1, batch_size=2)
        projection = model.transform(sparse.csr_matrix(dense))
        np.testing.assert_allclose(model.singular_values_, np.linalg.svd(dense, compute_uv=False), rtol=1e-6)
        np.testing.assert_allclose(model.explained_variance_ratio_,
                                   projection.var(axis=0) / dense.astype(float).var(axis=0).sum(), rtol=1e-12)

    def test_seed_and_batching_are_reproducible(self):
        rng = np.random.default_rng(3)
        matrix = sparse.random(119, 47, density=0.3, random_state=rng, format="csr", dtype=np.float32)
        first = svd.fit_streaming_svd(matrix, 5, n_iter=3, seed=91, batch_size=17)
        second = svd.fit_streaming_svd(matrix, 5, n_iter=3, seed=91, batch_size=17)
        other_batching = svd.fit_streaming_svd(matrix, 5, n_iter=3, seed=91, batch_size=9)
        np.testing.assert_array_equal(first.components_, second.components_)
        np.testing.assert_allclose(first.components_, other_batching.components_, rtol=1e-10, atol=1e-10)

    def test_memory_plans_cover_64_bit_offsets_without_allocating(self):
        small = svd.estimate_csr_memory(1000, 5000, 20000)
        large = svd.estimate_csr_memory(1000, 5000, 2**31)
        self.assertEqual(small["sparse_index_bits"], 32)
        self.assertEqual(large["sparse_index_bits"], 64)
        self.assertEqual(large["csr_bytes"], 12 * 2**31 + 8 * 1001)
        self.assertGreater(large["estimated_construction_peak_bytes"], large["csr_bytes"])
        self.assertIsInstance(svd.estimate_svd_workspace(5000, 300), int)
        plan = svd.estimate_svd_workspace(5000, 300, details=True)
        self.assertEqual(plan["sketch_columns"], 310)
        self.assertLess(plan["estimated_workspace_bytes"], 2 * 1024**3)
        with self.assertRaisesRegex(ValueError, "64-bit"):
            svd.estimate_csr_memory(1, 2, 2**63)

    def test_invalid_shapes_and_nonfinite_weights_fail(self):
        matrix = sparse.csr_matrix([[1., 2.], [3., 4.]])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            svd.fit_streaming_svd(matrix, 3)
        matrix.data[0] = np.inf
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            svd.fit_streaming_svd(matrix, 1, n_iter=1)


if __name__ == "__main__":
    unittest.main()
