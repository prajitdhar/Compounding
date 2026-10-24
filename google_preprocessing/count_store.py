"""Disk-backed additive counters used by the COHA extraction pipeline."""
from collections import Counter
from pathlib import Path
import sqlite3


INTEGER_COLUMNS = {
    'time', 'distance', 'direct_dependency', 'sentence_complete', 'count'
}


class CountStore:
    """Store tuple counters in SQLite and merge worker databases exactly."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=NORMAL')
        self.schemas = {}
        for (name,) in self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            columns = [row[1] for row in self.connection.execute(f'PRAGMA table_info("{name}")')]
            if columns and columns[-1] == 'count':
                self.schemas[name] = columns[:-1]

    def create(self, name, columns):
        columns = list(columns)
        if name in self.schemas:
            if self.schemas[name] != columns:
                raise ValueError(f'Conflicting schema for {name!r}')
            return
        definitions = []
        for column in columns:
            kind = 'INTEGER' if column in INTEGER_COLUMNS else 'TEXT'
            definitions.append(f'"{column}" {kind} NOT NULL')
        key = ', '.join(f'"{column}"' for column in columns)
        self.connection.execute(
            f'CREATE TABLE "{name}" ({", ".join(definitions)}, '
            f'"count" INTEGER NOT NULL, PRIMARY KEY ({key})) WITHOUT ROWID'
        )
        self.schemas[name] = columns

    def add(self, name, rows):
        if name not in self.schemas:
            raise KeyError(f'Unknown count table {name!r}')
        combined = Counter()
        for key, value in rows:
            combined[tuple(key)] += int(value)
        if not combined:
            return
        columns = self.schemas[name]
        quoted = ', '.join([*(f'"{column}"' for column in columns), '"count"'])
        placeholders = ', '.join('?' for _ in range(len(columns) + 1))
        key = ', '.join(f'"{column}"' for column in columns)
        self.connection.executemany(
            f'INSERT INTO "{name}" ({quoted}) VALUES ({placeholders}) '
            f'ON CONFLICT ({key}) DO UPDATE SET "count" = "count" + excluded."count"',
            [(*values, count) for values, count in combined.items()],
        )

    def merge_from(self, path):
        path = Path(path)
        alias = 'worker_counts'
        self.connection.execute(f'ATTACH DATABASE ? AS {alias}', [str(path)])
        try:
            for name, columns in self.schemas.items():
                available = self.connection.execute(
                    f"SELECT 1 FROM {alias}.sqlite_master WHERE type='table' AND name=?",
                    [name],
                ).fetchone()
                if not available:
                    continue
                quoted = ', '.join([*(f'"{column}"' for column in columns), '"count"'])
                key = ', '.join(f'"{column}"' for column in columns)
                self.connection.execute(
                    f'INSERT INTO "{name}" ({quoted}) SELECT {quoted} FROM {alias}."{name}" '
                    f'WHERE true ON CONFLICT ({key}) DO UPDATE '
                    f'SET "count" = "count" + excluded."count"'
                )
        finally:
            self.connection.execute(f'DETACH DATABASE {alias}')

    def close(self):
        self.connection.close()
