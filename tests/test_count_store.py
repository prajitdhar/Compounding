"""Generic exact counters, explicit export typing, and filtered grouping."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from src.count_store import CountStore


class CountStoreTests(unittest.TestCase):
    def test_export_has_explicit_key_types_filters_and_text_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CountStore(root / 'counts.sqlite')
            try:
                store.create('events', ['target', 'epoch', 'category', 'channel'])
                huge = 2 ** 53 + 1
                store.add('events', [
                    (('x', 2, 'alpha', 'visible'), huge),
                    (('x', 10, 'alpha', 'visible'), 3),
                    (('x', 2, 'beta', 'visible'), 4),
                    (('x', 2, 'alpha', 'hidden'), 5),
                    (('y', 2, 'alpha', 'visible'), 7)])
                self.assertEqual([row[2] for row in store.connection.execute(
                    'PRAGMA table_info(events)')], ['TEXT'] * 4 + ['INTEGER'])
                selected = root / 'selected.parquet'
                result = store.export_parquet('events', selected,
                    columns=['target', 'epoch'], integer_columns={'epoch'},
                    filters={'category': ['alpha'], 'channel': ['visible']}, batch_size=2)
                actual = pq.read_table(selected)
                self.assertEqual(actual.to_pylist(), [
                    {'target': 'x', 'epoch': 10, 'count': 3},
                    {'target': 'x', 'epoch': 2, 'count': huge},
                    {'target': 'y', 'epoch': 2, 'count': 7}])
                self.assertEqual(actual.schema.field('epoch').type, pa.int64())
                self.assertEqual(actual.schema.field('count').type, pa.int64())
                self.assertEqual(result, {'rows': 3, 'count_sum': huge + 10})
                self.assertEqual(pq.ParquetFile(selected).num_row_groups, 2)
                untyped = root / 'untyped.parquet'
                store.export_parquet('events', untyped, columns=['target', 'epoch'])
                self.assertEqual(pq.read_schema(untyped).field('epoch').type, pa.string())
                empty = root / 'empty.parquet'
                store.export_parquet('events', empty, columns=['target', 'epoch'],
                    integer_columns={'epoch'}, filters={'category': []})
                self.assertEqual(pq.read_table(empty).num_rows, 0)
                self.assertEqual(pq.read_schema(empty), actual.schema)
                with self.assertRaisesRegex(ValueError, 'Filter columns'):
                    store.export_parquet('events', selected, filters={'unknown': ['x']})
            finally:
                store.close()

    def test_missing_merge_input_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CountStore(root / 'counts.sqlite')
            try:
                store.create('events', ['target'])
                missing = root / 'missing.sqlite'
                with self.assertRaises(sqlite3.OperationalError):
                    store.merge_from(missing)
                self.assertFalse(missing.exists())
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
