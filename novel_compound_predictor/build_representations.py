#!/usr/bin/env python3
r"""Build shared-basis compound/constituent embeddings from context count tables.

Example (Google V3's lemma/POS feature-input export)::

    python build_representations.py \
      --pair-contexts "$DATA_ROOT/bigrams.parquet" \
      --pair-counts "$DATA_ROOT/bigram_counts.parquet" \
      --output representations_300

Default: CompoundAware noun constituents (separate modifier/head roles), raw
count SVD, 1800--1999 only. Without --temporal, pooling is across decades only;
modifier and head roles remain separate. Unigrams are not read by default.
``--word-contexts .../unigrams.parquet`` selects context-agnostic constituents.
``--temporal`` retains a separate row for each constituent/compound and decade,
still using one SVD basis for all rows. ``--constituents-only`` omits compound
rows from both SVD fitting and export, reducing memory for prediction jobs.
This fits a constituent-only basis instead of a joint three-role basis.
Outputs use tagged lexical keys; e.g.
``oil_NOUN``. ``compound`` is a readable, untagged phrase.

Accepted sources: exported V3 Parquet tables; raw reduced contexts/word_contexts
Parquet (normalized surfaces + POS, without lemmatization); headered
CSV/TSV context tables. Pair occurrence counts MUST be a separate table: sums of
context events are not compound frequencies. Already-aggregated V3 contexts
cannot reconstruct the old five-gram no-adjacent-noun filter. First-seen decisions use *all*
input attestation dates, before any frequency or embedding-vocabulary filters.

DuckDB projections are hash-partitioned on disk and reduced one bucket at a time.
Large sparse arrays can be memory-mapped to scratch. The default shared SVD
uses row-bounded randomized Gram iterations; --svd-solver sklearn selects the
earlier randomized implementation with larger row-dependent allocations.
Stage messages and RSS/progress heartbeats are flushed every --log-interval
seconds, including while SQL/SVD is busy. Use --max-contexts/--min-context-count
deliberately and record those settings.
No corpus is downloaded or modified. Existing output files are refused.
"""

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from contextlib import nullcontext

# Also support loading this CLI by file path from external test runners.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from progress_logging import ProgressReporter
from src.source_info import sha256_file, source_identity_paths


def implementation_identity():
    """Record local and shared implementations without publishing local paths."""
    paths = [Path(__file__), Path(__file__).with_name('representation_aggregation.py'),
             Path(__file__).with_name('representation_svd.py'), *source_identity_paths()]
    return {path.name: sha256_file(path) for path in paths}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def source_query(path):
    path = str(Path(path).expanduser())
    if path.endswith(('.parquet', '.pq')) or Path(path).is_dir():
        if Path(path).is_dir():
            path = str(Path(path) / '**/*.parquet')
        return f"SELECT * FROM read_parquet({literal(path)}, union_by_name=true)"
    if path.endswith(('.csv', '.tsv', '.csv.gz', '.tsv.gz')):
        return f"SELECT * FROM read_csv_auto({literal(path)}, header=true, sample_size=100000)"
    raise ValueError(f"Expected Parquet or headered CSV/TSV, received {path}")


def column_names(connection, query):
    return {row[0] for row in connection.execute(f'DESCRIBE ({query})').fetchall()}


def token_expression(columns, role):
    """Normalize supported noun POS suffixes while preserving lexical identity."""
    if role in columns:
        value = f'CAST({identifier(role)} AS VARCHAR)'
    elif f'{role}_norm' in columns and f'{role}_pos' in columns:
        value = f"CAST({identifier(role + '_norm')} AS VARCHAR) || '_' || {identifier(role + '_pos')}"
    elif f'{role}_surface' in columns and f'{role}_pos' in columns:
        value = f"CAST({identifier(role + '_surface')} AS VARCHAR) || '_' || {identifier(role + '_pos')}"
    else:
        raise ValueError(f"Missing {role} lexical key/POS columns; found {sorted(columns)}")
    value = f"regexp_replace({value}, '(?i)_(noun|n)$', '_NOUN')"
    if role == 'context':
        for pattern, tag in [('verb|v','VERB'),('adj|a','ADJ'),('adv|r','ADV')]:
            value = f"regexp_replace({value}, '(?i)_({pattern})$', '_{tag}')"
    return value


def canonical_query(connection, path, roles):
    query = source_query(path)
    columns = column_names(connection, query)
    time_column = next((c for c in ['time', 'decade', 'year'] if c in columns), None)
    if time_column is None or 'count' not in columns:
        raise ValueError(f'{path}: count and time/decade/year columns are required')
    terms = [f'{token_expression(columns, role)} AS {identifier(role)}' for role in roles]
    terms += [f'TRY_CAST({identifier(time_column)} AS DOUBLE) AS source_time',
              'TRY_CAST(count AS DOUBLE) AS count']
    return f"SELECT {', '.join(terms)} FROM ({query})"


def validate_source(connection, name, roles):
    null_tokens = ' OR '.join(f'{identifier(r)} IS NULL OR trim({identifier(r)})=\'\'' for r in roles)
    bad = connection.execute(f"""
        SELECT count(*) FROM {identifier(name)} WHERE {null_tokens}
          OR source_time IS NULL OR NOT isfinite(source_time)
          OR source_time != floor(source_time)
          OR count IS NULL OR NOT isfinite(count) OR count < 0
          OR count != floor(count)
    """).fetchone()[0]
    if bad:
        raise ValueError(f'{name}: {bad:,} rows have invalid lexical keys, dates, or nonnegative integer counts')


def register_source(connection, name, path, roles):
    query = canonical_query(connection, path, roles)
    connection.execute(f'CREATE VIEW {identifier(name)} AS {query}')
    validate_source(connection, name, roles)


def noun_predicate(*roles):
    return ' AND '.join(f"ends_with({identifier(role)}, '_NOUN')" for role in roles)


def prepare_tables(connection, args, reporter=None, work_directory=None):
    """Prepare exact count tables without a whole-corpus target/context hash map.

    High-cardinality aggregations are hash-partitioned on disk. Canonical past
    pair contexts are shuffled once, without the failing global four-key GROUP
    BY. Top-context selection precedes target aggregation: SUM is distributive,
    so this preserves the old context ranking while reducing shuffle volume.
    """
    import contextlib
    from representation_aggregation import BoundedAggregation

    def stage(label):
        return reporter.stage(label,query_progress=True) if reporter else contextlib.nullcontext()

    if work_directory is None:
        database = connection.execute('PRAGMA database_list').fetchone()[2]
        work_directory = (Path(database).parent if database else Path(args.output))/'bounded_aggregation'
    aggregate = BoundedAggregation(
        connection,work_directory,
        partitions=getattr(args,'aggregation_partitions',256),
        batch_size=getattr(args,'aggregation_batch_size',65536),
        buffer_mb=getattr(args,'aggregation_buffer_mb',256),
        max_split_depth=getattr(args,'aggregation_max_split_depth',4),reporter=reporter)

    for name,path,roles in [
        ('pair_source',args.pair_contexts,['modifier','head','context']),
        ('count_source',args.pair_counts,['modifier','head']),
        *([('word_source',args.word_contexts,['word','context'])] if args.word_contexts else []),
        *([('modifier_source',args.modifier_contexts,['modifier','context']),
           ('head_source',args.head_contexts,['head','context'])] if args.modifier_contexts else []),
    ]:
        with stage(f'Validate {name}'):
            register_source(connection,name,path,roles)

    # Keep every attestation date, including rare pre-window occurrences.
    with stage('Partition and aggregate pair occurrence counts'):
        counts_input = aggregate.shuffle(f"""SELECT modifier,head,
            CAST(floor(source_time/10)*10 AS INTEGER) AS decade,count
            FROM count_source WHERE {noun_predicate('modifier','head')} AND count > 0""",
            ['modifier','head'],'occurrence-shuffle')
        counts = aggregate.reduce(counts_input,"""SELECT modifier,head,decade,SUM(count) AS count
            FROM __INPUT__ GROUP BY modifier,head,decade""",'occurrence-counts')
        counts.register(connection,'pair_counts')
    with stage('Compute first attestations and eligible compounds per pair bucket'):
        metadata = aggregate.reduce(counts,f"""SELECT modifier,head,MIN(decade) AS first_seen,
            SUM(count) AS count,
            SUM(CASE WHEN decade >= {args.novel_start} AND decade < {args.novel_end}
                     THEN count ELSE 0 END) AS novel_period_count,
            (MIN(decade) >= {args.novel_start} AND MIN(decade) < {args.novel_end}
             AND SUM(CASE WHEN decade >= {args.novel_start} AND decade < {args.novel_end}
                          THEN count ELSE 0 END) >= {args.novel_min_count}) AS novel
            FROM __INPUT__ GROUP BY modifier,head""",'pair-metadata')
        metadata.register(connection,'pair_metadata')
        eligible = aggregate.reduce(counts,f"""SELECT modifier,head FROM __INPUT__
            WHERE decade >= {args.year_start} AND decade < {args.train_before}
            GROUP BY modifier,head HAVING SUM(count) >= {args.min_compound_count}""",
            'eligible-compounds')
        eligible.register(connection,'eligible_compounds')
        training = aggregate.reduce(counts,f"""SELECT modifier,head,SUM(count) AS count
            FROM __INPUT__ WHERE decade >= {args.year_start} AND decade < {args.train_before}
            GROUP BY modifier,head""",'training-compounds')
        training.register(connection,'training_compounds')
        historical = aggregate.reduce(counts,f"""SELECT modifier,head,SUM(count) AS count
            FROM __INPUT__ WHERE decade < {args.train_before}
            GROUP BY modifier,head""",'historical-compounds')
        historical.register(connection,'historical_compounds')

    # Materialize small role vocabularies once, rather than building a semijoin
    # hash table from all pair/decade rows for every pass over the contexts.
    for role in ['modifier','head']:
        with stage(f'Collect historical {role} vocabulary'):
            role_input = aggregate.shuffle(f"""SELECT {role} FROM pair_counts
                WHERE decade < {args.train_before}""",[role],f'{role}-vocabulary-shuffle')
            vocabulary = aggregate.reduce(role_input,
                f'SELECT {role} FROM __INPUT__ GROUP BY {role}',f'{role}-vocabulary')
            vocabulary.register(connection,f'past_{role}s')

    with stage('Normalize and partition past pair contexts once'):
        pair_contexts = aggregate.shuffle(f"""SELECT modifier,head,context,
            CAST(floor(source_time/10)*10 AS INTEGER) AS decade,count FROM pair_source
            WHERE {noun_predicate('modifier','head')} AND count > 0
              AND source_time >= {args.year_start} AND source_time < {args.train_before}""",
            ['modifier','head'],'pair-context-shuffle')
        pair_contexts.register(connection,'pair_contexts')
    # Validate before context-POS or top-k filtering. Even an unknown pair whose
    # only context is a discarded function word must still be rejected.
    with stage('Check context/count pair compatibility per bucket'):
        missing = 0
        total = len(pair_contexts.buckets)
        for index,bucket in enumerate(sorted(pair_contexts.buckets),1):
            missing += connection.execute(f"""SELECT COUNT(*) FROM
                (SELECT modifier,head FROM ({pair_contexts.query(bucket)})
                 EXCEPT SELECT modifier,head FROM ({metadata.query(bucket)}))""").fetchone()[0]
            if reporter:
                reporter.update(detail=f'Checked context bucket {index}/{total}',completed=index,total=total)
        if missing:
            raise ValueError(f'{missing:,} context pairs have no matching occurrence count; check POS/lemma schemas')

    decade = 'p.decade' if args.temporal else 'CAST(-1 AS INTEGER)'
    terms = [] if args.constituents_only else [f"""SELECT 'compound' AS kind,
        p.modifier,p.head,{decade} AS decade,p.context,p.count
        FROM pair_contexts p JOIN eligible_compounds USING(modifier,head)"""]
    for role in ['modifier','head']:
        if args.word_contexts:
            table,token = 'word_source','word'
        elif args.modifier_contexts:
            table,token = f'{role}_source',role
        else:
            table,token = 'pair_contexts',role
        time_column = 'decade' if table == 'pair_contexts' else 'source_time'
        decade = f'CAST(floor({time_column}/10)*10 AS INTEGER)' if args.temporal else 'CAST(-1 AS INTEGER)'
        terms.append(f"""SELECT {literal(role)} AS kind,
            {token if role == 'modifier' else "''"} AS modifier,
            {token if role == 'head' else "''"} AS head,{decade} AS decade,context,count
            FROM {table} WHERE {token} IN (SELECT {role} FROM past_{role}s)
              AND {noun_predicate(token)} AND count > 0
              AND {time_column} >= {args.year_start} AND {time_column} < {args.train_before}""")
    # A lazy projection, not a materialized multi-billion-group aggregation.
    connection.execute('CREATE VIEW raw_events AS SELECT * FROM ('+' UNION ALL '.join(terms)
                       +") WHERE regexp_matches(context, '_(NOUN|VERB|ADJ|ADV)$')")
    limit = f'LIMIT {args.max_contexts}' if args.max_contexts else ''
    with stage('Select exact past content-context vocabulary'):
        connection.execute(f"""CREATE TABLE contexts AS
            SELECT context,row_number() OVER (ORDER BY context)-1 AS context_id FROM
                (SELECT context FROM raw_events GROUP BY context
                 HAVING SUM(count) >= {args.min_context_count}
                 ORDER BY SUM(count) DESC,context {limit})""")
    row_keys = ['kind','modifier','head','decade']
    with stage('Partition retained target/context events and aggregate bounded buckets'):
        event_input = aggregate.shuffle("""SELECT e.kind,e.modifier,e.head,e.decade,
            c.context_id,e.count FROM raw_events e JOIN contexts c USING(context)""",
            row_keys,'event-shuffle')
        events = aggregate.reduce(event_input,"""SELECT kind,modifier,head,decade,context_id,
            SUM(count) AS count FROM __INPUT__
            GROUP BY kind,modifier,head,decade,context_id""",'events')
        events.register(connection,'events')
    with stage('Deduplicate row identifiers within target buckets'):
        rows = aggregate.reduce(events,"""SELECT kind,modifier,head,decade FROM __INPUT__
            GROUP BY kind,modifier,head,decade""",'row-identifiers')
        rows.register(connection,'row_identifiers')
    with stage('Assign ordered row identifiers with external sorting'):
        connection.execute("""CREATE TABLE rows AS SELECT kind,modifier,head,decade,
            row_number() OVER (ORDER BY kind,modifier,head,decade)-1 AS row_id
            FROM row_identifiers""")
    with stage('Compute compound embedding coverage per pair bucket'):
        observed_input = aggregate.shuffle("""SELECT modifier,head FROM row_identifiers
            WHERE kind='compound'""",['modifier','head'],'observed-pair-shuffle')
        observed = aggregate.reduce(observed_input,"""SELECT modifier,head FROM __INPUT__
            GROUP BY modifier,head""",'observed-pairs')
        compound_metadata = aggregate.combine(metadata,observed,"""SELECT p.*,
            o.modifier IS NOT NULL AS has_embedding FROM __LEFT__ p
            LEFT JOIN __RIGHT__ o USING(modifier,head)""",'compound-metadata')
        compound_metadata.register(connection,'compound_metadata')
    connection.execute('DROP VIEW raw_events')


def memory_bytes(value):
    match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([KMGT]?I?B)?\s*', str(value).upper())
    if not match:
        raise ValueError(f'Invalid memory size {value!r}; use e.g. 128GB or 64GiB')
    unit = match.group(2) or 'B'
    power = 0 if unit == 'B' else 'KMGT'.index(unit[0]) + 1
    return int(float(match.group(1)) * (1024 if 'I' in unit else 1000) ** power)


def numerical_budget(args):
    if args.numerical_memory_limit:
        return memory_bytes(args.numerical_memory_limit)
    limits = []
    slurm_memory = os.environ.get('SLURM_MEM_PER_NODE')
    if slurm_memory and slurm_memory.isdigit():
        limits.append(int(slurm_memory) * 1024**2)
    cpu_memory = os.environ.get('SLURM_MEM_PER_CPU')
    cpu_count = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpu_memory and cpu_count and cpu_memory.isdigit() and cpu_count.isdigit():
        limits.append(int(cpu_memory) * int(cpu_count) * 1024**2)
    for path in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        try:
            text = Path(path).read_text().strip()
            if text.isdigit() and int(text) < 2**60:
                limits.append(int(text))
        except OSError:
            pass
    if not limits:
        limits.append(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    return int(min(limits) * .65)


def _stage(reporter, name, **fields):
    return reporter.stage(name, **fields) if reporter else nullcontext()


def build_sparse_matrix(connection, batch_size, reporter=None, scratch=None,
                        storage='auto', memory_limit=None):
    import numpy as np
    from scipy.sparse import csr_matrix

    n_rows = connection.execute('SELECT count(*) FROM rows').fetchone()[0]
    n_contexts = connection.execute('SELECT count(*) FROM contexts').fetchone()[0]
    nnz = connection.execute('SELECT count(*) FROM events').fetchone()[0]
    if min(n_rows, n_contexts) < 2 or nnz == 0:
        raise ValueError('Need at least two retained rows and two contexts; review dates/POS/frequency filters')
    index_dtype = np.int64 if max(n_rows, n_contexts, nnz) >= 2**31 else np.int32
    csr_bytes = nnz * (4 + np.dtype(index_dtype).itemsize) + (n_rows + 1) * np.dtype(index_dtype).itemsize
    storage = ('disk' if memory_limit and csr_bytes > memory_limit // 4 else 'memory') if storage == 'auto' else storage
    if storage == 'memory' and memory_limit and csr_bytes + n_rows * 8 > memory_limit:
        raise ValueError(f'CSR construction needs at least {(csr_bytes+n_rows*8)/1024**3:.1f} GiB, '
                         'above the numerical memory budget; use --matrix-storage disk')
    if reporter:
        reporter.log('Sparse matrix memory plan', rows=n_rows, contexts=n_contexts,
                     nonzeros=nnz, csr_gib=round(csr_bytes/1024**3,3), storage=storage,
                     numerical_budget_gib=round(memory_limit/1024**3,3) if memory_limit else None)
    scratch = Path(scratch) if scratch else None
    if storage == 'disk' and scratch is None:
        raise ValueError('Disk CSR storage requires a scratch directory')
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)

    def allocate(name, length, dtype, zero=False):
        result = (np.memmap(scratch/f'{name}.bin', dtype=dtype, mode='w+', shape=(length,))
                  if storage == 'disk' else np.empty(length, dtype=dtype))
        if zero:
            result.fill(0)
        return result

    data = allocate('data', nnz, np.float32)
    indices = allocate('indices', nnz, index_dtype)
    row_lengths = allocate('row-lengths', n_rows, np.int64, zero=True)
    reader = connection.execute("""SELECT r.row_id,e.context_id,e.count FROM events e
        JOIN rows r USING(kind,modifier,head,decade) ORDER BY r.row_id,e.context_id
    """).to_arrow_reader(batch_size)
    offset = 0
    for batch in reader:
        length = batch.num_rows
        row_ids = batch.column(0).to_numpy(zero_copy_only=False)
        indices[offset:offset+length] = batch.column(1).to_numpy(zero_copy_only=False)
        data[offset:offset+length] = batch.column(2).to_numpy(zero_copy_only=False)
        unique, sizes = np.unique(row_ids, return_counts=True)
        row_lengths[unique] += sizes
        offset += length
        if reporter:
            reporter.update('Reading sparse coordinates', completed=offset, total=nnz)
    if offset != nnz:
        raise RuntimeError(f'Sparse row count changed during construction: {offset} != {nnz}')
    indptr = allocate('indptr', n_rows+1, index_dtype)
    indptr[0] = 0
    np.cumsum(row_lengths, out=indptr[1:])
    matrix = csr_matrix((data, indices, indptr), shape=(n_rows,n_contexts), copy=False)
    matrix.storage_mode = storage
    return matrix


def weight_matrix(matrix, weighting, reporter=None):
    """Apply the selected weighting; raw occurrence counts are the default."""
    import numpy as np
    if weighting == 'raw':
        return matrix
    if weighting == 'log1p':
        np.log1p(matrix.data, out=matrix.data)
        return matrix
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    total = float(row_sums.sum(dtype=np.float64))
    positive_entries = 0
    # Work row by row to avoid allocating a repeated row-index vector of nnz.
    for row in range(matrix.shape[0]):
        start,end = matrix.indptr[row:row+2]
        values = matrix.data[start:end]
        denom = row_sums[row] * col_sums[matrix.indices[start:end]].astype(np.float64)
        values[:] = np.maximum(np.log(values.astype(np.float64)*total/denom),0)
        positive_entries += int(np.count_nonzero(values))
        if reporter and row % 10000 == 0:
            reporter.update('Weighting sparse rows', completed=row, total=matrix.shape[0])
    if getattr(matrix,'storage_mode','memory') != 'disk':
        matrix.eliminate_zeros()
    # SciPy may copy heavily pruned arrays back into RAM. Explicit zeros have
    # the same algebraic effect and preserve mapped storage for large jobs.
    matrix.logical_nonzeros = positive_entries
    return matrix


def readable_compound(modifier, head):
    def strip(token):
        return token[:-5] if token.endswith('_NOUN') else token
    return strip(modifier)+' '+strip(head)


def export_embeddings(connection, matrix, svd, args, destination, reporter=None):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.preprocessing import normalize

    dims = [f'dim_{i}' for i in range(args.dimensions)]
    outputs = {}
    role_outputs = [('modifier','modifiers'),('head','heads')]
    if not args.constituents_only:
        role_outputs.insert(0,('compound','compounds'))
    for kind, plural in role_outputs:
        metadata = ['compound','modifier','head'] if kind == 'compound' else [kind]
        if args.temporal:
            metadata.append('decade')
        schema = pa.schema([(name,pa.int32() if name == 'decade' else pa.string())
                            for name in metadata] + [(name,pa.float32()) for name in dims])
        path = destination / f'dense_embeddings_{plural}.{args.format}'
        if reporter:
            reporter.log(f'Writing {plural} embeddings', output=str(path))
        reader = connection.execute(f"""SELECT row_id,modifier,head,decade FROM rows
            WHERE kind={literal(kind)} ORDER BY row_id""").to_arrow_reader(args.batch_size)
        writer = pq.ParquetWriter(path,schema,compression='zstd') if args.format == 'parquet' else None
        csv_file = path.open('w',newline='') if args.format == 'csv' else None
        csv_writer = csv.writer(csv_file) if csv_file else None
        if csv_writer:
            csv_writer.writerow(metadata+dims)
        total = 0
        try:
            for batch in reader:
                row_ids = batch.column(0).to_numpy(zero_copy_only=False)
                vectors = svd.transform(matrix[row_ids]).astype(np.float32,copy=False)
                if args.normalize:
                    normalize(vectors,copy=False)
                values = batch.to_pydict()
                columns = {}
                if kind == 'compound':
                    columns['compound'] = [readable_compound(m,h) for m,h in zip(values['modifier'],values['head'])]
                    columns['modifier'] = values['modifier']
                    columns['head'] = values['head']
                else:
                    columns[kind] = values[kind]
                if args.temporal:
                    columns['decade'] = values['decade']
                if writer:
                    table = pa.Table.from_pydict({**columns,**{name:vectors[:,i] for i,name in enumerate(dims)}},schema=schema)
                    writer.write_table(table)
                else:
                    for i in range(batch.num_rows):
                        csv_writer.writerow([columns[name][i] for name in metadata]+vectors[i].tolist())
                total += batch.num_rows
                if reporter:
                    reporter.update(f'{plural}: {total:,} rows written', completed=total)
        finally:
            if writer:
                writer.close()
            if csv_file:
                csv_file.close()
        outputs[plural] = {'path':path.name,'rows':total,'dimensions':args.dimensions}
        if reporter:
            reporter.log(f'Finished {plural} embeddings', rows=total, output=str(path))
    return outputs


def copy_csv(connection, query, destination, reporter=None):
    with _stage(reporter, f'Write {Path(destination).name}', query_progress=True):
        connection.execute(f'COPY ({query}) TO {literal(destination)} (FORMAT CSV, HEADER TRUE)')


def run(args):
    import duckdb
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    validate_arguments(args)
    destination = Path(args.output).expanduser()
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f'Output directory must be empty to protect existing data: {destination}')
    scratch = Path(args.scratch).expanduser() if args.scratch else destination
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='embedding-build-',dir=scratch) as work, \
            ProgressReporter(interval=args.log_interval) as reporter:
        reporter.log('Starting representation build', compound_aspect='CompoundAgnostic' if args.word_contexts else 'CompoundAware',
                     temporal=args.temporal, max_contexts=args.max_contexts, dimensions=args.dimensions,
                     sql_memory_limit=args.memory_limit, svd_solver=args.svd_solver,
                     output=str(destination), scratch=str(work))
        connection = duckdb.connect(str(Path(work)/'aggregation.duckdb'))
        try:
            connection.execute(f'SET memory_limit={literal(args.memory_limit)}')
            connection.execute(f'SET threads={args.threads}')
            connection.execute('SET preserve_insertion_order=false')
            connection.execute(f'SET temp_directory={literal(Path(work)/"spill")}')
            connection.execute('SET enable_progress_bar=true')
            connection.execute('SET enable_progress_bar_print=false')
            connection.execute('SET progress_bar_time=0')
            reporter.connection = connection
            with reporter.stage('Validate sources and prepare partitioned context tables'):
                prepare_tables(connection,args,reporter=reporter,work_directory=Path(work)/'aggregation')
            budget = numerical_budget(args)
            with reporter.stage('Build sparse matrix', query_progress=True):
                matrix = build_sparse_matrix(connection,args.batch_size,reporter=reporter,
                                             scratch=Path(work)/'csr',storage=args.matrix_storage,
                                             memory_limit=budget)
            with reporter.stage(f'Apply {args.weighting} weights'):
                matrix = weight_matrix(matrix,args.weighting,reporter=reporter)
            # An explicit request for 300 dimensions never silently becomes 299.
            if args.dimensions > min(matrix.shape):
                raise ValueError(f'Requested {args.dimensions} dimensions but matrix is {matrix.shape}; '
                                 'need more rows/contexts or explicitly select fewer dimensions')
            # Release the aggregation buffer cache before numerical allocations.
            reporter.connection = None
            connection.close()
            connection = duckdb.connect(str(Path(work)/'aggregation.duckdb'))
            metadata_budget = min(memory_bytes(args.memory_limit), 8 * 1000**3)
            connection.execute(f'SET memory_limit={literal(str(metadata_budget)+"B")}')
            connection.execute(f'SET threads={args.threads}')
            connection.execute('SET preserve_insertion_order=false')
            connection.execute(f'SET temp_directory={literal(Path(work)/"spill")}')
            connection.execute('SET enable_progress_bar=true')
            connection.execute('SET enable_progress_bar_print=false')
            connection.execute('SET progress_bar_time=0')
            reporter.connection = connection
            reporter.log('Released aggregation cache; metadata connection uses a smaller memory cap',
                         metadata_memory_gib=round(metadata_budget/1024**3,3))
            with reporter.stage(f'Fit {args.dimensions}d shared SVD ({args.svd_solver})'):
                if args.svd_solver == 'streaming':
                    from representation_svd import fit_streaming_svd, estimate_svd_workspace
                    workspace = estimate_svd_workspace(
                        matrix.shape[1],args.dimensions,batch_size=args.batch_size,
                        max_block_nnz=min(matrix.nnz,min(args.batch_size,matrix.shape[0])*matrix.shape[1]),
                        sparse_index_bits=matrix.indices.dtype.itemsize*8)
                    resident = (matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
                                if getattr(matrix,'storage_mode','memory') == 'memory' else 0)
                    reporter.log('Numerical workspace plan', workspace_gib=round(workspace/1024**3,3),
                                 resident_csr_gib=round(resident/1024**3,3), budget_gib=round(budget/1024**3,3))
                    if workspace + resident > budget:
                        raise ValueError('Numerical workspace exceeds --numerical-memory-limit; reduce '
                                         '--batch-size, use --matrix-storage disk, or allocate more memory')
                    svd = fit_streaming_svd(matrix,args.dimensions,n_iter=args.svd_iterations,
                                            seed=args.seed,batch_size=args.batch_size,reporter=reporter)
                else:
                    # This solver may allocate several full rows x dimensions arrays.
                    peak = (matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
                            + matrix.shape[0] * (args.dimensions+10) * 4 * 4)
                    if peak > budget:
                        raise ValueError('Estimated sklearn SVD memory exceeds numerical budget; '
                                         'use --svd-solver streaming')
                    svd = TruncatedSVD(n_components=args.dimensions,n_iter=args.svd_iterations,random_state=args.seed)
                    svd.fit(matrix)
            with reporter.stage('Export dense embeddings'):
                outputs = export_embeddings(connection,matrix,svd,args,destination,reporter=reporter)
            np.savez_compressed(destination/'svd_basis.npz',components=svd.components_,
                                singular_values=svd.singular_values_,
                                explained_variance_ratio=svd.explained_variance_ratio_)
            copy_csv(connection,'SELECT context_id,context FROM contexts ORDER BY context_id',destination/'contexts.csv',reporter)
            copy_csv(connection,'SELECT * FROM pair_counts ORDER BY modifier,head,decade',destination/'compound_counts.csv',reporter)
            copy_csv(connection,'SELECT * FROM compound_metadata ORDER BY modifier,head',destination/'compound_metadata.csv',reporter)
            copy_csv(connection,'SELECT modifier,head FROM pair_metadata WHERE novel ORDER BY modifier,head',
                     destination/'novel_compounds.csv',reporter)
            copy_csv(connection,'SELECT modifier,head FROM pair_metadata ORDER BY modifier,head',
                     destination/'attested_compounds.csv',reporter)
            copy_csv(connection,'SELECT * FROM training_compounds ORDER BY modifier,head',destination/'training_compounds.csv',reporter)
            # Ranking must also exclude compounds attested before the vector
            # window (e.g. a 1750-only pair when vectors start at 1800).
            copy_csv(connection,'SELECT * FROM historical_compounds ORDER BY modifier,head',destination/'historical_compounds.csv',reporter)
            manifest = {
                'schema_version':2,'status':'complete','parameters':vars(args),
                'code_sha256':implementation_identity(),
                'source_files':{key:source_record(getattr(args,key)) for key in
                                ['pair_contexts','pair_counts','word_contexts','modifier_contexts','head_contexts']
                                if getattr(args,key)},
                'basis':('one joint basis across modifier and head, and all retained decades'
                         if args.constituents_only else
                         'one joint basis across compound, modifier, head, and all retained decades'),
                'constituent_mode':'word-context-agnostic' if args.word_contexts else 'compound-role-aware',
                'compound_aspect':'CompoundAgnostic' if args.word_contexts else 'CompoundAware',
                'context_dimensions':'lexical_POS; NOUN/VERB/ADJ/ADV only; side and distance summed',
                'lexical_keys':'POS retained; _n/_noun suffixes normalized to _NOUN; no new lemmatization',
                'upstream_selection':'Past-only filters apply to supplied counts. Any earlier vocabulary '
                                     'selection across future decades cannot be undone here; strict forecasting '
                                     'requires a past-only upstream vocabulary and extraction.',
                'time_window':{'start_inclusive':args.year_start,'end_exclusive':args.train_before},
                'novelty':'first positive occurrence across all supplied count dates; novel-period count threshold',
                'role_vocabulary':'noun roles attested before train_before; future-only role keys are excluded',
                'novelty_scope':'relative to supplied corpus counts; not proof of linguistic coinage',
                'matrix':{'rows':matrix.shape[0],'contexts':matrix.shape[1],
                          'nonzeros':getattr(matrix,'logical_nonzeros',matrix.nnz),'stored_entries':matrix.nnz,
                          'csr_bytes':matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes,
                          'storage':getattr(matrix,'storage_mode','memory')},
                'svd_solver':args.svd_solver,
                'svd_implementation':getattr(svd,'solver','sklearn_randomized'),
                'svd_oversamples':getattr(svd,'n_oversamples',10),
                'svd_accumulation_dtype':'float64' if args.svd_solver == 'streaming' else 'float32',
                'explained_variance_ratio_sum':float(np.sum(svd.explained_variance_ratio_)),
                'outputs':outputs,
                'software':{name:importlib.metadata.version(name) for name in
                            ['duckdb','numpy','scipy','scikit-learn','pyarrow']},
                'visualization':'Pair prediction uses constituent vectors. Create visualization exports '
                                'separately with export_visualization_dataset.py.',
            }
            (destination/'representation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps({'output':str(destination),'embeddings':outputs},indent=2),flush=True)
            return manifest
        finally:
            reporter.connection = None
            connection.close()


def source_record(path):
    path = Path(path).expanduser().resolve()
    record = {'path':str(path)}
    if path.is_file():
        stat = path.stat()
        record.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    sidecar = Path(str(path)+'.json')
    if sidecar.is_file():
        record['sidecar'] = str(sidecar)
    directory_manifest = path.parent / 'corpus_manifest.json'
    if directory_manifest.is_file():
        record['directory_manifest'] = str(directory_manifest)
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pair-contexts',required=True,help='modifier,head,context,time/decade/year,count table')
    p.add_argument('--pair-counts',required=True,help='independent modifier,head,time/decade/year,count occurrences')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--word-contexts',help='general word,context,time,count table for context-agnostic constituents')
    mode.add_argument('--modifier-contexts',help='modifier,context,time,count table; also provide --head-contexts')
    p.add_argument('--head-contexts',help='head,context,time,count table')
    p.add_argument('--output',required=True,help='empty output directory')
    p.add_argument('--dimensions',type=int,default=300)
    p.add_argument('--year-start',type=int,default=1800)
    p.add_argument('--train-before',type=int,default=2000)
    p.add_argument('--novel-start',type=int,default=2000)
    p.add_argument('--novel-end',type=int,default=2010)
    p.add_argument('--novel-min-count',type=int,default=3)
    p.add_argument('--min-compound-count',type=int,default=1)
    p.add_argument('--min-context-count',type=int,default=1)
    p.add_argument('--max-contexts',type=int,default=50000,help='most frequent context limit (default: 50000)')
    p.add_argument('--weighting',choices=['raw','log1p','ppmi'],default='raw')
    p.add_argument('--temporal',action='store_true',help='one row per item/decade, still in a shared basis')
    p.add_argument('--constituents-only',action='store_true',help='fit/export heads and modifiers only; changes the SVD basis')
    p.add_argument('--no-normalize',dest='normalize',action='store_false',default=True)
    p.add_argument('--seed',type=int,default=1991)
    p.add_argument('--svd-iterations',type=int,default=10)
    p.add_argument('--svd-solver',choices=['streaming','sklearn'],default='streaming',
                   help='streaming uses row-bounded randomized Gram iterations; sklearn retains the older implementation')
    p.add_argument('--batch-size',type=int,default=10000)
    p.add_argument('--format',choices=['parquet','csv'],default='parquet')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-limit',default='8GB',help='DuckDB aggregation limit; SVD memory is additional')
    p.add_argument('--numerical-memory-limit',help='CSR/SVD planning budget, e.g. 128GB; default 65%% of Slurm/cgroup RAM')
    p.add_argument('--matrix-storage',choices=['auto','memory','disk'],default='auto',
                   help='auto memory-maps large CSR arrays into scratch instead of allocating them in RAM')
    p.add_argument('--aggregation-partitions',type=int,default=256)
    p.add_argument('--aggregation-batch-size',type=int,default=65536)
    p.add_argument('--aggregation-buffer-mb',type=int,default=256)
    p.add_argument('--aggregation-max-split-depth',type=int,default=4)
    p.add_argument('--log-interval',type=float,default=30,help='heartbeat interval in seconds (default: 30)')
    p.add_argument('--scratch',help='scratch directory for aggregation database and disk spill')
    return p


def validate_arguments(args):
    if bool(args.modifier_contexts) != bool(args.head_contexts):
        raise ValueError('--modifier-contexts and --head-contexts must be provided together')
    if args.year_start >= args.train_before or args.novel_start >= args.novel_end:
        raise ValueError('Time windows must have start < end')
    if args.train_before > args.novel_start:
        raise ValueError('Embedding window overlaps the novel period; --train-before must be <= --novel-start')
    if any(value%10 for value in [args.year_start,args.train_before,args.novel_start,args.novel_end]):
        raise ValueError('Time boundaries must be decade-aligned because output counts are decadal')
    for name in ['dimensions','novel_min_count','min_compound_count','min_context_count',
                 'svd_iterations','batch_size','threads','aggregation_partitions',
                 'aggregation_batch_size','aggregation_buffer_mb']:
        if getattr(args,name) < 1:
            raise ValueError(f'--{name.replace("_","-")} must be positive')
    if args.max_contexts is not None and args.max_contexts < 1:
        raise ValueError('--max-contexts must be positive')
    if args.aggregation_max_split_depth < 0:
        raise ValueError('--aggregation-max-split-depth must be nonnegative')
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        raise ValueError('--log-interval must be positive and finite')
    if args.numerical_memory_limit and memory_bytes(args.numerical_memory_limit) < 1:
        raise ValueError('--numerical-memory-limit must be positive')


if __name__ == '__main__':
    try:
        run(parser().parse_args())
    except (ValueError,ImportError) as error:
        print(f'error: {error}',file=sys.stderr)
        sys.exit(2)
