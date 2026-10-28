"""Bounded hash-partitioned aggregation for the representation builder.

A projection is scanned once and routed from bounded Arrow batches into
Parquet buckets. Equal partition keys always reach the same bucket. Reduction
then opens one bucket at a time; an oversized bucket is recursively rehashed
on those same keys. This follows the shared-budget Arrow shuffle pattern in
the shared src/partitioned_parquet.py implementation without importing corpus
preprocessing or compositionality-analysis code.

Only scratch files are created here. These are not cross-run checkpoints:
the caller owns the enclosing work directory and its lifecycle.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import shutil
import time


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def columns(names):
    return ', '.join('"' + name.replace('"', '""') + '"' for name in names)


@dataclass
class PartitionedTable:
    schema: object
    buckets: dict
    empty_file: Path
    partition_keys: tuple
    routing: tuple

    def files(self, bucket=None):
        if bucket is None:
            return [path for key in sorted(self.buckets) for path in self.buckets[key]]
        return self.buckets.get(bucket, [])

    def query(self, bucket=None):
        paths = self.files(bucket) or [self.empty_file]
        return 'SELECT * FROM read_parquet([' + ','.join(quote(path) for path in paths) + \
               '], hive_partitioning=false, union_by_name=true)'

    def register(self, connection, name):
        connection.execute(f'CREATE VIEW "{name}" AS {self.query()}')


class BufferedWriter:
    """One Arrow buffer budget across all buckets, with bounded open writers."""

    def __init__(self, root, schema, buffer_bytes, batch_size, max_open=64):
        self.root = Path(root)
        self.schema = schema
        self.limit = buffer_bytes
        self.batch_size = batch_size
        self.max_open = max_open
        self.buffers = {}
        self.sizes = {}
        self.buffer_rows = {}
        self.buffer_bytes = 0
        self.writers = OrderedDict()
        self.numbers = {}
        self.files = {}

    def append(self, bucket, table):
        if not table.num_rows:
            return
        self.buffers.setdefault(bucket, []).append(table)
        size = table.get_total_buffer_size()
        self.sizes[bucket] = self.sizes.get(bucket, 0) + size
        self.buffer_rows[bucket] = self.buffer_rows.get(bucket, 0) + table.num_rows
        self.buffer_bytes += size
        if self.buffer_rows[bucket] >= self.batch_size:
            self.flush(bucket)
        while self.buffer_bytes > self.limit:
            self.flush(max(self.sizes, key=self.sizes.get))

    def close_writer(self, bucket):
        writer, path, unused = self.writers.pop(bucket)
        writer.close()
        self.files.setdefault(bucket, []).append(path)

    def writer(self, bucket):
        import pyarrow.parquet as pq
        if bucket in self.writers:
            self.writers.move_to_end(bucket)
            return self.writers[bucket]
        if len(self.writers) >= self.max_open:
            self.close_writer(next(iter(self.writers)))
        directory = self.root / f'bucket-{bucket:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        number = self.numbers.get(bucket, 0)
        self.numbers[bucket] = number + 1
        path = directory / f'fragment-{number:05d}.parquet'
        state = (pq.ParquetWriter(path, self.schema, compression='zstd'), path, 0)
        self.writers[bucket] = state
        return state

    def flush(self, bucket):
        import pyarrow as pa
        table = pa.concat_tables(self.buffers.pop(bucket)).combine_chunks()
        self.buffer_bytes -= self.sizes.pop(bucket)
        self.buffer_rows.pop(bucket)
        offset = 0
        while offset < table.num_rows:
            writer, path, written = self.writer(bucket)
            size = min(table.num_rows-offset, 2_000_000-written)
            writer.write_table(table.slice(offset, size), row_group_size=self.batch_size)
            self.writers[bucket] = (writer, path, written+size)
            offset += size
            if written+size >= 2_000_000:
                self.close_writer(bucket)

    def finish(self):
        for bucket in list(self.buffers):
            self.flush(bucket)
        for bucket in list(self.writers):
            self.close_writer(bucket)
        return self.files

    def close(self):
        for bucket in list(self.writers):
            self.close_writer(bucket)
        self.buffers.clear()
        self.sizes.clear()
        self.buffer_rows.clear()
        self.buffer_bytes = 0


class BoundedAggregation:
    def __init__(self, connection, root, *, partitions=256, batch_size=65536,
                 buffer_mb=256, max_split_depth=4, reporter=None):
        self.connection = connection
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.partitions = partitions
        self.batch_size = batch_size
        self.buffer_bytes = buffer_mb*1024**2
        self.max_split_depth = max_split_depth
        self.reporter = reporter
        if min(partitions, batch_size, buffer_mb) < 1 or max_split_depth < 0:
            raise ValueError('Aggregation partition/batch/buffer limits must be positive; split depth nonnegative')

    def log(self, message):
        if self.reporter is not None:
            self.reporter.log(message)

    def update(self, message, completed=None, total=None):
        if self.reporter is not None:
            self.reporter.update(detail=message, completed=completed, total=total)

    def empty(self, root, schema):
        import pyarrow as pa
        import pyarrow.parquet as pq
        root.mkdir(parents=True, exist_ok=True)
        path = root/'empty.parquet'
        pq.write_table(pa.Table.from_batches([], schema=schema), path)
        return path

    def shuffle(self, query, partition_keys, name, *, partitions=None, seed=0):
        """Read the projection once, never once per hash bucket."""
        import numpy as np
        import pyarrow as pa

        root = self.root/name
        root.mkdir(parents=True, exist_ok=True)
        schema = self.connection.execute(f'SELECT * FROM ({query}) LIMIT 0').to_arrow_table().schema
        if '__bucket' in schema.names:
            raise ValueError('The source projection uses reserved column __bucket')
        count = partitions or self.partitions
        project = (f'SELECT *, hash({seed}, {columns(partition_keys)}) % {count} AS __bucket '
                   f'FROM ({query}) p')
        writer = BufferedWriter(root, schema, self.buffer_bytes, self.batch_size)
        reader = None
        seen = 0
        updated = time.monotonic()
        self.log(f'{name}: shuffle projection once into {count} hash buckets')
        try:
            reader = self.connection.execute(project).to_arrow_reader(self.batch_size)
            for batch in reader:
                table = pa.Table.from_batches([batch])
                buckets = table['__bucket'].to_numpy().astype(np.int64, copy=False)
                order = np.argsort(buckets, kind='stable')
                ordered = buckets[order]
                boundaries = np.concatenate(([0],np.flatnonzero(np.diff(ordered))+1,[len(order)]))
                output = table.select(schema.names)
                for start,stop in zip(boundaries[:-1],boundaries[1:]):
                    # take() copies selected arrays: tiny bucket buffers must
                    # not retain an entire parent Arrow batch.
                    writer.append(int(ordered[start]), output.take(pa.array(order[start:stop])))
                seen += batch.num_rows
                if time.monotonic()-updated >= 10:
                    self.update(f'{name}: shuffled {seen:,} rows')
                    updated = time.monotonic()
            files = writer.finish()
        finally:
            if reader is not None:
                reader.close()
            writer.close()
        self.log(f'{name}: shuffle complete; {seen:,} rows, {len(files)} nonempty buckets')
        return PartitionedTable(schema,files,self.empty(root,schema),tuple(partition_keys),(seed,count))

    def reduce(self, source, template, name):
        """Replace __INPUT__ in SQL and reduce each co-located bucket separately.

        template must retain partition_keys unchanged. It may group by more
        columns, but never fewer than partition_keys. Thus concatenating bucket
        outputs requires no global DISTINCT or GROUP BY.
        """
        root = self.root/name
        root.mkdir(parents=True, exist_ok=True)
        example = template.replace('__INPUT__',
                                   f'(SELECT * FROM read_parquet({quote(source.empty_file)}))')
        schema = self.connection.execute(f'SELECT * FROM ({example}) LIMIT 0').to_arrow_table().schema
        missing = set(source.partition_keys)-set(schema.names)
        if missing:
            raise ValueError(f'{name}: reducer must retain partition keys {sorted(missing)}')
        files = {}
        total = len(source.buckets)
        self.log(f'{name}: reducing {total} buckets independently')
        for index,bucket in enumerate(sorted(source.buckets),1):
            files[bucket] = self._reduce_bucket(source,source.files(bucket),template,
                                                root/f'bucket-{bucket:05d}',0)
            self.update(f'{name}: reduced bucket {index}/{total}',completed=index,total=total)
        self.log(f'{name}: reduction complete')
        return PartitionedTable(schema,files,self.empty(root,schema),source.partition_keys,source.routing)

    def _copy_reduction(self, sql, destination):
        """Separate method permits deterministic OOM regression tests."""
        self.connection.execute(f'COPY ({sql}) TO {quote(destination)} '
                                f'(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {max(2048,self.batch_size)})')

    def combine(self, left, right, template, name):
        """Join matching hash buckets without hashing the whole population."""
        if left.partition_keys != right.partition_keys or left.routing != right.routing:
            raise ValueError('Combined tables must use the same partition keys and hash routing')
        root = self.root/name
        root.mkdir(parents=True, exist_ok=True)

        # No source rows are read to infer the output schema.
        empty = template.replace('__LEFT__',
            f'(SELECT * FROM read_parquet({quote(left.empty_file)}))').replace('__RIGHT__',
            f'(SELECT * FROM read_parquet({quote(right.empty_file)}))')
        schema = self.connection.execute(f'SELECT * FROM ({empty}) LIMIT 0').to_arrow_table().schema
        files = {}
        for index,bucket in enumerate(sorted(left.buckets),1):
            files[bucket] = self._combine_bucket(left,left.files(bucket),right,right.files(bucket),
                                                 template,root/f'bucket-{bucket:05d}',0)
            self.update(f'{name}: joined bucket {index}/{len(left.buckets)}',
                        completed=index,total=len(left.buckets))
        return PartitionedTable(schema,files,self.empty(root,schema),left.partition_keys,left.routing)

    def _combine_bucket(self, left, left_paths, right, right_paths, template, root, depth):
        import duckdb
        root.mkdir(parents=True,exist_ok=True)
        left_subset = PartitionedTable(left.schema,{0:left_paths},left.empty_file,
                                       left.partition_keys,left.routing)
        right_subset = PartitionedTable(right.schema,{0:right_paths},right.empty_file,
                                        right.partition_keys,right.routing)
        sql = template.replace('__LEFT__',f'({left_subset.query()})').replace(
            '__RIGHT__',f'({right_subset.query()})')
        destination = root/'joined.parquet'
        try:
            self._copy_reduction(sql,destination)
            return [destination]
        except duckdb.OutOfMemoryException:
            destination.unlink(missing_ok=True)
            if depth >= self.max_split_depth:
                raise RuntimeError(f'Join bucket {root} exceeded memory after {depth} recursive splits; '
                                   'increase --aggregation-partitions or --aggregation-max-split-depth') from None
            self.log(f'{root.name}: join memory limit reached; rehashing both inputs into 16 children')
            relative = root.relative_to(self.root)
            left_children = self.shuffle(left_subset.query(),left.partition_keys,
                str(relative/'left-split'),partitions=16,seed=depth+1)
            right_children = self.shuffle(right_subset.query(),right.partition_keys,
                str(relative/'right-split'),partitions=16,seed=depth+1)
            results = []
            for child in sorted(left_children.buckets):
                results.extend(self._combine_bucket(left_children,left_children.files(child),
                    right_children,right_children.files(child),template,root/f'child-{child:02d}',depth+1))
            shutil.rmtree(self.root/relative/'left-split')
            shutil.rmtree(self.root/relative/'right-split')
            return results

    def _reduce_bucket(self, source, paths, template, root, depth):
        import duckdb
        root.mkdir(parents=True, exist_ok=True)
        subset = PartitionedTable(source.schema,{0:paths},source.empty_file,source.partition_keys,source.routing)
        sql = template.replace('__INPUT__',f'({subset.query()})')
        destination = root/'reduced.parquet'
        try:
            self._copy_reduction(sql,destination)
            return [destination]
        except duckdb.OutOfMemoryException:
            destination.unlink(missing_ok=True)
            if depth >= self.max_split_depth:
                raise RuntimeError(f'Aggregation bucket {root} exceeded memory after '
                                   f'{depth} recursive splits; increase --aggregation-partitions '
                                   'or --aggregation-max-split-depth') from None
            self.log(f'{root.name}: memory limit reached; rehashing this bucket into 16 children')
            # Rehash only this bucket. No original corpus is rescanned, and
            # changing the seed avoids repeating the parent routing decision.
            relative = root.relative_to(self.root)/'split'
            children = self.shuffle(subset.query(),source.partition_keys,str(relative),
                                    partitions=16,seed=depth+1)
            results = []
            for child in sorted(children.buckets):
                results.extend(self._reduce_bucket(children,children.files(child),template,
                                                    root/f'child-{child:02d}',depth+1))
            shutil.rmtree(self.root/relative)
            return results
