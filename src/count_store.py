"""Disk-backed aggregation of integer counts; no corpus-sized result list."""
import csv
import sqlite3
from pathlib import Path
from collections import Counter


class CountStore:
    def __init__(self, path, *, disk_temp=False):
        self.connection = sqlite3.connect(path, uri=True)
        if disk_temp:
            # The extraction process configures its SQL scratch before opening
            # stores. Keep large GROUP BY/ORDER BY work disk-backed, not in RAM.
            self.connection.execute('PRAGMA temp_store=FILE')
        self.schemas = {}

    def create(self, name, columns):
        self.schemas[name] = list(columns)
        fields = ', '.join(f'"{c}" TEXT NOT NULL' for c in columns)
        keys = ', '.join(f'"{c}"' for c in columns)
        self.connection.execute(
            f'CREATE TABLE "{name}" ({fields}, count INTEGER NOT NULL, '
            f'PRIMARY KEY ({keys})) WITHOUT ROWID')

    def add(self, name, rows):
        # Combine duplicate events within a batch before writing to SQLite.
        counts = Counter()
        for key, count in rows:
            counts[tuple(str(v) for v in key)] += int(count)
        if not counts:
            return
        placeholders = ','.join('?' for _ in range(len(self.schemas[name]) + 1))
        self.connection.executemany(
            f'INSERT INTO "{name}" VALUES ({placeholders}) '
            'ON CONFLICT DO UPDATE SET count = count + excluded.count',
            [(*key, count) for key, count in counts.items()])

    def export(self, name, path):
        self.connection.commit()
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow([*self.schemas[name], 'count'])
            order = ', '.join(f'"{column}"' for column in self.schemas[name])
            cursor = self.connection.execute(f'SELECT * FROM "{name}" ORDER BY {order}')
            writer.writerows(cursor)
        temporary.replace(path)

    def merge_from(self, path):
        """Add an independently counted shard using SQLite, without Python rows."""
        self.connection.commit()
        uri = Path(path).resolve().as_uri() + '?mode=ro'
        self.connection.execute('ATTACH DATABASE ? AS shard', (uri,))
        try:
            with self.connection:
                for name, columns in self.schemas.items():
                    actual = self.connection.execute(f'PRAGMA shard.table_info("{name}")').fetchall()
                    if [column[1] for column in actual] != [*columns, 'count']:
                        raise ValueError(f'Incompatible shard schema for {name!r} in {path}')
                    fields = ', '.join(f'"{column}"' for column in [*columns, 'count'])
                    # WHERE resolves SQLite's SELECT/UPSERT parsing ambiguity.
                    self.connection.execute(
                        f'INSERT INTO "{name}" ({fields}) '
                        f'SELECT {fields} FROM shard."{name}" WHERE true '
                        'ON CONFLICT DO UPDATE SET count = count + excluded.count')
        finally:
            self.connection.execute('DETACH DATABASE shard')

    def export_parquet(self, name, path, *, columns=None, integer_columns=(),
                       filters=None, batch_size=65536):
        """Stream exact aggregated integers to Parquet without a full DataFrame.

        Output key types and allowed values are supplied by the caller. Filters
        apply before grouping over selected columns; unselected categories are
        pooled. The count column always uses signed 64-bit integers.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if batch_size < 1:
            raise ValueError('Parquet batch size must be positive')
        columns = list(self.schemas[name] if columns is None else columns)
        if not columns or not set(columns) <= set(self.schemas[name]):
            raise ValueError('Export columns must come from the stored schema')
        fields = ', '.join(f'"{c}"' for c in columns)
        clauses, parameters = [], []
        for column, values in (filters or {}).items():
            if column not in self.schemas[name]:
                raise ValueError('Filter columns must come from the stored schema')
            values = list(values)
            if not values:
                clauses.append('0 = 1')
            else:
                clauses.append(f'"{column}" IN (' + ','.join('?' for _ in values) + ')')
                parameters.extend(values)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        sql = f'SELECT {fields}, SUM(count) AS count FROM "{name}"{where} GROUP BY {fields} ORDER BY {fields}'
        cursor = self.connection.execute(sql, parameters)
        names = columns + ['count']
        numeric = {'count', *integer_columns}
        schema = pa.schema([(c, pa.int64() if c in numeric else pa.string()) for c in names])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        row_count, count_sum = 0, 0
        try:
            with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
                while rows := cursor.fetchmany(batch_size):
                    arrays = [pa.array([int(row[i]) if c in numeric else row[i] for row in rows], type=schema.field(c).type)
                              for i, c in enumerate(names)]
                    writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                    row_count += len(rows)
                    count_sum += sum(row[-1] for row in rows)
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            cursor.close()
        return {'rows': row_count, 'count_sum': count_sum}

    def close(self):
        self.connection.close()
