"""Resource controls preserve sparse coordinates and reject invalid budgets."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import duckdb
import numpy as np

SPEC = importlib.util.spec_from_file_location('resource_builder', Path(__file__).resolve().parents[1]/'build_representations.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ResourceTests(unittest.TestCase):
    def connection(self):
        connection = duckdb.connect()
        self.addCleanup(connection.close)
        connection.execute('CREATE TABLE rows(row_id BIGINT, kind VARCHAR, modifier VARCHAR, head VARCHAR, decade INTEGER)')
        connection.execute("INSERT INTO rows VALUES (0,'modifier','oil_NOUN','',-1),(1,'head','','field_NOUN',-1)")
        connection.execute('CREATE TABLE contexts(context_id BIGINT, context VARCHAR)')
        connection.execute("INSERT INTO contexts VALUES (0,'land_NOUN'),(1,'energy_NOUN'),(2,'green_ADJ')")
        connection.execute('CREATE TABLE events(kind VARCHAR, modifier VARCHAR, head VARCHAR, decade INTEGER, context_id BIGINT, count DOUBLE)')
        connection.execute("INSERT INTO events VALUES ('modifier','oil_NOUN','',-1,2,3),('head','','field_NOUN',-1,0,4),('modifier','oil_NOUN','',-1,0,2)")
        return connection

    def test_disk_csr_preserves_values_and_file_backing(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix = builder.build_sparse_matrix(self.connection(), 1, scratch=directory, storage='disk')
            self.assertEqual(matrix.storage_mode, 'disk')
            np.testing.assert_array_equal(matrix.toarray(), [[2,0,3],[4,0,0]])
            self.assertTrue(any(isinstance(value, np.memmap) for value in [matrix.data, matrix.data.base, matrix.data.base.base]))
            self.assertTrue((Path(directory)/'indptr.bin').exists())

    def test_auto_storage_maps_matrix_above_ram_share(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix = builder.build_sparse_matrix(self.connection(), 2, scratch=directory,
                                                 storage='auto', memory_limit=100)
            self.assertEqual(matrix.storage_mode, 'disk')
            np.testing.assert_array_equal(matrix.toarray(), [[2,0,3],[4,0,0]])

    def test_ppmi_does_not_copy_pruned_disk_arrays_back_into_ram(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = self.connection()
            connection.execute('DELETE FROM events')
            connection.execute("""INSERT INTO events SELECT kind,modifier,head,decade,context_id,1
                FROM rows CROSS JOIN contexts""")
            matrix = builder.build_sparse_matrix(connection, 2, scratch=directory, storage='disk')
            backing = matrix.data
            expected = builder.weight_matrix(matrix.copy(), 'ppmi').toarray()
            weighted = builder.weight_matrix(matrix, 'ppmi')
            np.testing.assert_allclose(weighted.toarray(), expected)
            self.assertTrue(np.shares_memory(weighted.data, backing))
            self.assertEqual(weighted.logical_nonzeros, np.count_nonzero(expected))
            self.assertEqual(weighted.logical_nonzeros, 0)
            self.assertEqual(weighted.nnz, 6)  # Explicit zeros retain the mapped buffers.

    def test_memory_budget_is_checked_before_allocating_csr(self):
        with self.assertRaisesRegex(ValueError, 'matrix-storage disk'):
            builder.build_sparse_matrix(self.connection(), 1, storage='memory', memory_limit=1)

    def test_memory_sizes_and_explicit_numerical_limit(self):
        self.assertEqual(builder.memory_bytes('2GiB'), 2*1024**3)
        self.assertEqual(builder.memory_bytes('1.5GB'), 1500000000)
        with self.assertRaises(ValueError):
            builder.memory_bytes('unlimited')
        args = builder.parser().parse_args(['--pair-contexts','contexts.parquet','--pair-counts','counts.parquet',
                                             '--output','unused','--numerical-memory-limit','1MiB'])
        self.assertEqual(builder.numerical_budget(args), 1024**2)

    def test_slurm_allocation_bounds_automatic_numerical_budget(self):
        args = builder.parser().parse_args(['--pair-contexts','contexts.parquet','--pair-counts','counts.parquet','--output','unused'])
        with mock.patch.dict('os.environ', {'SLURM_MEM_PER_NODE':'196608'}, clear=True), \
                mock.patch.object(Path, 'read_text', side_effect=OSError):
            self.assertEqual(builder.numerical_budget(args), int(192*1024**3*.65))


if __name__ == '__main__':
    unittest.main()
