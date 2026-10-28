"""Exercise actual SQLite disk spills, not just the configured path string."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coha_pipeline import sqlite_scratch
from src.count_store import CountStore


class SQLiteScratchTests(unittest.TestCase):
    def test_environment_restored_after_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            for original in [None, '/previous/sqlite-temp']:
                with patch.dict(os.environ):
                    if original is None:
                        os.environ.pop('SQLITE_TMPDIR', None)
                    else:
                        os.environ['SQLITE_TMPDIR'] = original
                    with sqlite_scratch(directory):
                        self.assertEqual(os.environ['SQLITE_TMPDIR'], str(Path(directory).resolve()))
                    self.assertEqual(os.environ.get('SQLITE_TMPDIR'), original)
                    with self.assertRaisesRegex(RuntimeError, 'failure'):
                        with sqlite_scratch(directory):
                            raise RuntimeError('failure')
                    self.assertEqual(os.environ.get('SQLITE_TMPDIR'), original)

    def test_spawned_workers_inherit_sqlite_scratch(self):
        with tempfile.TemporaryDirectory() as directory, sqlite_scratch(directory):
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
                observed = pool.submit(os.getenv, 'SQLITE_TMPDIR').result()
            self.assertEqual(observed, str(Path(directory).resolve()))

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Uses Linux /proc to observe open spill files')
    def test_real_group_by_spills_to_requested_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch = root / 'sql-scratch'
            scratch.mkdir()
            observed = set()
            with sqlite_scratch(scratch):
                store = CountStore(root / 'counts.sqlite', disk_temp=True)
                try:
                    db = store.connection
                    self.assertEqual(db.execute('PRAGMA temp_store').fetchone()[0], 1)
                    if ('TEMP_STORE=3',) in db.execute('PRAGMA compile_options').fetchall():
                        self.skipTest('SQLite compiled to force temporary storage into RAM')
                    db.execute('PRAGMA cache_size=-64')
                    db.execute('PRAGMA threads=0')
                    # No index can satisfy this grouping; ~15 MB exceeds the
                    # small SQLite sort buffer and exercises real temp I/O.
                    db.execute('CREATE TABLE words_sentence(word TEXT, time INTEGER, context TEXT, count INTEGER)')
                    db.executemany('INSERT INTO words_sentence VALUES (?, ?, ?, ?)',
                        ((f'w{i % 1000:04}', 2000, 'x' * 300 + str(50000 - i), 1) for i in range(50000)))
                    db.commit()

                    def observe():
                        for fd in Path('/proc/self/fd').iterdir():
                            try:
                                target = os.readlink(fd)
                            except FileNotFoundError:
                                continue
                            if 'etilqs_' in target:
                                observed.add(target)
                        return 0

                    db.set_progress_handler(observe, 10000)
                    total = sum(row[-1] for row in db.execute(
                        'SELECT word, time, context, SUM(count) FROM words_sentence '
                        'GROUP BY word, time, context ORDER BY word, time, context'))
                    self.assertEqual(total, 50000)
                    self.assertTrue(observed, 'The regression must exercise disk spills')
                    self.assertTrue(all(Path(p.removesuffix(' (deleted)')).parent == scratch for p in observed), observed)
                finally:
                    store.close()
            self.assertEqual(list(scratch.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
