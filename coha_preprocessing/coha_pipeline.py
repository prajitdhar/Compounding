#!/usr/bin/env python3
"""COHA v3: conservative raw-text preparation and two aligned context modes."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import gzip
import gc
import hashlib
import html
import io
import json
from itertools import islice
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
import unicodedata
import zipfile

import pandas as pd
import spacy

from src.count_store import CountStore
from src import count_store as count_store_implementation
from src.source_info import (artifact, canonical_time, source_info, software_record,
                             sha256_file, validate_feature_input_manifest)

VERSION = 3
MODES = ('local5', 'sentence')
_WORKER_SPACY = None
CONTEXT_POS = {'NOUN', 'PROPN', 'ADJ', 'VERB', 'NUM', 'ADV'}
INTEGER_COUNT_COLUMNS = {'time', 'distance', 'direct_dependency', 'sentence_complete'}
# Numeric markers end at whitespace/EOF (optionally after a period) or .txt. Some
# academic headers have no space after .txt; prose "##1/2" is not document 1.
DOC_MARKER = re.compile(r'(?<![@#])(?:@@|##)(\d+)(?:\.txt(?:[ \t]*\|)?|\.?(?=\s|$))')
GAP = re.compile(r'@+(?:\s+@+)*')


def pipeline_software():
    return software_record([__file__, Path(count_store_implementation.__file__)],
                           ['spacy', 'pandas', 'numpy', 'pyarrow'])


def digest_file(path):
    return sha256_file(path)


@contextmanager
def gzip_writer(path):
    # A fixed timestamp and empty original filename make identical input runs
    # byte-reproducible, including the manifest checksum.
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=3) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='\n') as text:
                yield text


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def text_sources(path):
    """Yield archive member names and bytes; nested ZIPs are spooled to disk."""
    path = Path(path)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for file in files:
        name = str(file.relative_to(path)) if path.is_dir() else file.name
        if file.suffix.lower() == '.txt':
            yield name, file.read_bytes()
        # Check the outer TAR first: ZIP's EOF-based detection also accepts a
        # TAR containing ZIPs, exposing only the final ZIP and losing the rest.
        elif tarfile.is_tarfile(file):
            with tarfile.open(file) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.name.lower().endswith('.txt'):
                        with archive.extractfile(member) as stream:
                            yield f'{name}!{member.name}', stream.read()
                    elif member.name.lower().endswith('.zip'):
                        with archive.extractfile(member) as stream, tempfile.SpooledTemporaryFile(max_size=8*1024*1024) as spool:
                            shutil.copyfileobj(stream, spool)
                            spool.seek(0)
                            with zipfile.ZipFile(spool) as zipped:
                                for child in sorted(zipped.namelist()):
                                    if child.lower().endswith('.txt'):
                                        yield f'{name}!{member.name}!{child}', zipped.read(child)
        elif zipfile.is_zipfile(file):
            with zipfile.ZipFile(file) as archive:
                for member in sorted(archive.namelist()):
                    if member.lower().endswith('.txt'):
                        yield f'{name}!{member}', archive.read(member)


def source_metadata(name):
    basename = name.split('!')[-1].split('/')[-1]
    match = re.match(r'(?:text_)?([a-zA-Z]+)_(\d{4})(?:_(\d+))?', basename)
    if not match:
        raise ValueError(f'Cannot infer domain/time from {name}; expected text_fic_1820.txt or fic_1821_123.txt')
    domain, time, doc_id = match.groups()
    return {'domain': domain.lower(), 'time': int(time), 'file_doc_id': doc_id,
            'time_precision': 'year' if doc_id else 'decade'}


def documents(text, fallback_id):
    markers = list(DOC_MARKER.finditer(text))
    if not markers:
        yield fallback_id, text
        return
    if text[:markers[0].start()].strip():
        # Keep the prefix rather than silently discarding possible prose.
        yield fallback_id + ':prefix', text[:markers[0].start()]
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        yield marker.group(1), text[marker.end():end]


def normalize(text):
    text = unicodedata.normalize('NFC', html.unescape(text))
    text = text.translate(str.maketrans({'“': '"', '”': '"', '‘': "'", '’': "'"}))
    text = re.sub(r'<\s*/?\s*(?:p|br)\s*/?\s*>', '\n', text, flags=re.I)
    text = re.sub(r'</?[A-Za-z][^>]*>', ' ', text)
    # COHA's /p/ and /q/ are structural markers, not lexical tokens.
    text = re.sub(r'/p/', '\n', text, flags=re.I)
    text = re.sub(r'/q/', '"', text, flags=re.I)
    text = text.replace('\x00', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text


def fragments(text):
    gaps = list(GAP.finditer(text))
    starts = [0] + [m.end() for m in gaps]
    ends = [m.start() for m in gaps] + [len(text)]
    for i, (start, end) in enumerate(zip(starts, ends)):
        raw = text[start:end]
        if not raw.strip():
            continue
        stripped = len(raw) - len(raw.lstrip())
        yield {'text': raw.strip(), 'normalized_start': start + stripped,
               'left_gap': i > 0, 'right_gap': i < len(gaps),
               'left_gap_length': gaps[i - 1].group().count('@') if i > 0 else 0,
               'right_gap_length': gaps[i].group().count('@') if i < len(gaps) else 0}


def validate_stage_paths(source, output, scratch=None):
    """Keep generated files and temporary work outside the source tree."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input and output paths must be separate, non-overlapping locations')
    if scratch is not None:
        scratch = Path(scratch).resolve()
        if scratch == source or source in scratch.parents:
            raise ValueError('Scratch directory must be outside the input directory')


def prepare(args):
    output = Path(args.output)
    validate_stage_paths(args.input, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'prepared_manifest.json').unlink(missing_ok=True)
    metadata = {}
    source_metadata_keys = False
    if args.metadata:
        frame = canonical_time(pd.read_csv(args.metadata, sep='\t', dtype={'doc_id': str, 'source': str}))
        source_metadata_keys = 'source' in frame.columns
        keys = ['source', 'doc_id'] if source_metadata_keys else ['doc_id']
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f'Metadata {keys} keys must be present and unique')
        metadata = frame.set_index(keys).to_dict('index')
    stats = Counter()
    domains = Counter()
    times = Counter()
    precisions = Counter()
    sources = []
    seen = set()
    original_ids = {}
    temporary = output / 'fragments.jsonl.gz.tmp'
    with gzip_writer(temporary) as writer:
        for name, content in text_sources(args.input):
            info = source_metadata(name)
            text = content.decode(args.encoding, errors='strict')
            sources.append({'member': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            stats['source_members'] += 1
            for doc_id, raw in documents(text, info['file_doc_id'] or name):
                identity = (name, doc_id)
                if identity in seen:
                    raise ValueError(f'Duplicate document ID {doc_id!r} within source {name!r}; check input collection')
                seen.add(identity)
                if doc_id in original_ids:
                    stats['document_id_reuses_across_sources'] += 1
                    if args.metadata and not source_metadata_keys:
                        raise ValueError(f'Metadata doc_id {doc_id!r} is ambiguous between '
                                         f'{original_ids[doc_id]!r} and {name!r}; '
                                         'add a source column to the metadata TSV')
                else:
                    original_ids[doc_id] = name
                document_uid = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode('utf-8')).hexdigest()
                entry = metadata.get(identity if source_metadata_keys else doc_id, {})
                if args.metadata and not entry:
                    raise ValueError(f'No metadata for document {doc_id!r} in source {name!r}')
                domain = str(entry.get('domain', info['domain']))
                time = int(entry.get('time', info['time']))
                if args.domains and domain not in args.domains:
                    stats['documents_excluded_by_domain'] += 1
                    continue
                normalized = normalize(raw)
                precision = entry.get('time_precision', 'year') if entry else info['time_precision']
                if precision not in {'year', 'decade'}:
                    raise ValueError(f'Unknown time_precision {precision!r} for {doc_id!r}')
                times[time] += 1
                precisions[precision] += 1
                stats['documents'] += 1
                stats['gap_runs'] += len(list(GAP.finditer(normalized)))
                stats['missing_token_markers'] += normalized.count('@')
                domains[f'{time // 10 * 10}:{domain}'] += 1
                for number, fragment in enumerate(fragments(normalized)):
                    record = dict(fragment, doc_id=doc_id, document_uid=document_uid,
                                  fragment_id=number, source=name,
                                  time=time, domain=domain,
                                  time_precision=precision)
                    writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats['fragments'] += 1
    if not sources:
        raise ValueError('No .txt members found in the input')
    temporary.replace(output / 'fragments.jsonl.gz')
    manifest = {
        'schema_version': VERSION, 'sources': sources, 'stats': dict(stats),
        'documents_by_decade_domain': dict(domains), 'encoding': args.encoding,
        'metadata_sha256': digest_file(args.metadata) if args.metadata else None,
        'fragments_sha256': digest_file(output / 'fragments.jsonl.gz'),
        'normalization': 'NFC, HTML decoding, quote normalization, structural markup; no spelling repair',
        'language_filter': 'none', 'gap_policy': 'split before annotation; never join across gaps',
        'document_identity': 'document_uid = SHA256 of UTF-8 JSON [source, doc_id]; original doc_id retained',
        'offsets': 'character offsets in each normalized document',
        'code_sha256': digest_file(__file__),
        'count_store_sha256': digest_file(Path(count_store_implementation.__file__)),
        'time_column': 'time', 'domains': args.domains or 'all'}
    inputs = [{'name': source['member'], 'container': str(Path(args.input).resolve()),
               'bytes': source['bytes'], 'sha256': source['sha256']} for source in sources]
    if args.metadata:
        inputs.append(artifact(args.metadata, 'document_metadata'))
    manifest['source_info'] = source_info('prepare',
        {'name': 'COHA', 'release': getattr(args, 'corpus_release', None), 'subset': args.domains or 'all_supplied'},
        inputs, {key: manifest[key] for key in ['encoding', 'normalization', 'language_filter', 'gap_policy', 'domains']},
        {'column': 'time', 'unit': 'calendar_year', 'precision_counts': dict(precisions),
         'observed_min': min(times) if times else None, 'observed_max': max(times) if times else None,
         'binning': 'source dates; decade collections use decade start'},
        pipeline_software(), outputs=[artifact(output / 'fragments.jsonl.gz', sha256=manifest['fragments_sha256'])],
        statistics={'counts': dict(stats), 'documents_by_decade_domain': dict(domains)})
    write_json(output / 'prepared_manifest.json', manifest)
    print(f'Prepared {stats["documents"]} documents and {stats["fragments"]} observed fragments', flush=True)


def read_fragments(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            if 'year' in record:
                if 'time' in record and record['time'] != record['year']:
                    raise ValueError('Conflicting time and legacy year in prepared record')
                record['time'] = record.pop('year')
            if 'year_precision' in record:
                record.setdefault('time_precision', record.pop('year_precision'))
            yield record['text'], record


def lexical(token):
    return not token.is_space and not token.is_punct and '@' not in token.text


def pos(token, proper_nouns):
    return 'NOUN' if proper_nouns == 'map' and token.pos_ == 'PROPN' else token.pos_


def lemma(token, proper_nouns):
    return f'{token.lemma_.lower()}_{pos(token, proper_nouns)}'


def context_events(sentence, start, end, mode, proper_nouns):
    # Whitespace tokens are not positions; punctuation remains a position but
    # is not a lexical context. No path can leave this observed sentence span.
    tokens = [t for t in sentence if not t.is_space]
    positions = {token.i: i for i, token in enumerate(tokens)}
    first, last = positions[start], positions[end - 1]
    for i, token in enumerate(tokens):
        if first <= i <= last or not lexical(token) or token.pos_ not in CONTEXT_POS:
            continue
        side = 'L' if i < first else 'R'
        distance = first - i if side == 'L' else i - last
        if mode == 'sentence' or distance <= 3:
            yield lemma(token, proper_nouns), side, distance


def opportunities(sentence, start, end, uncertain_left, uncertain_right):
    """Six local positions per target, distinguishing gaps from true boundaries."""
    tokens = [token for token in sentence if not token.is_space]
    positions = {token.i: i for i, token in enumerate(tokens)}
    for side in ['L', 'R']:
        for distance in [1, 2, 3]:
            index = positions[start] - distance if side == 'L' else positions[end - 1] + distance
            if index < 0 or index >= len(tokens):
                uncertain = uncertain_left if side == 'L' else uncertain_right
                status = 'gap' if uncertain else 'sentence_boundary'
            else:
                token = tokens[index]
                status = 'context' if lexical(token) and token.pos_ in CONTEXT_POS else 'filtered_token'
            yield side, distance, status


def schemas():
    result = {'unigram_counts': ['word', 'time', 'domain'],
              'bigram_counts': ['modifier', 'head', 'time', 'domain'],
              'bigram_document_counts': ['modifier', 'head', 'time', 'domain', 'document_uid'],
              'bigram_annotations': ['modifier', 'head', 'time', 'domain', 'modifier_surface', 'head_surface',
                                   'modifier_pos', 'head_pos', 'direct_dependency', 'modifier_dep',
                                   'modifier_ner', 'head_ner', 'sentence_complete'],
              'bigram_opportunities_local5': ['modifier', 'head', 'time', 'domain', 'side', 'distance', 'status'],
              'unigram_opportunities_local5': ['word', 'time', 'domain', 'side', 'distance', 'status']}
    for mode in MODES:
        result[f'bigrams_{mode}'] = ['modifier', 'head', 'time', 'domain', 'context']
        result[f'unigrams_{mode}'] = ['word', 'time', 'domain', 'context']
        result[f'bigram_coverage_{mode}'] = ['modifier', 'head', 'time', 'domain', 'status']
        result[f'unigram_coverage_{mode}'] = ['word', 'time', 'domain', 'status']
    return result


def add_document(store, doc, record, sentence_writer, proper_nouns):
    pending = defaultdict(list)
    stats = Counter()
    sentences = list(doc.sents)
    time, domain = record['time'], record['domain']
    for number, sentence in enumerate(sentences):
        uncertain_left = number == 0 and record['left_gap']
        terminal = re.search(r'''[.!?][\"'”’\)\]]*\s*$''', sentence.text) is not None
        uncertain_right = number == len(sentences) - 1 and record['right_gap'] and not terminal
        complete = not (uncertain_left or uncertain_right)
        saved = dict(record, text=sentence.text, sentence_id=number,
                     normalized_start=record['normalized_start'] + sentence.start_char,
                     sentence_complete=complete, left_gap=uncertain_left, right_gap=uncertain_right,
                     left_gap_length=record['left_gap_length'] if uncertain_left else 0,
                     right_gap_length=record['right_gap_length'] if uncertain_right else 0)
        sentence_writer.write(json.dumps(saved, ensure_ascii=False) + '\n')
        stats['sentences'] += 1
        stats['complete_sentences'] += complete
        tokens = [token for token in sentence if not token.is_space]
        for token in tokens:
            if not lexical(token):
                continue
            word = lemma(token, proper_nouns)
            pending['unigram_counts'].append(((word, time, domain), 1))
            stats['lexical_tokens'] += 1
            for side, distance, status in opportunities(sentence, token.i, token.i + 1, uncertain_left, uncertain_right):
                pending['unigram_opportunities_local5'].append(((word, time, domain, side, distance, status), 1))
            for mode in MODES:
                events = list(context_events(sentence, token.i, token.i + 1, mode, proper_nouns)) if mode == 'local5' or complete else []
                status = 'gap_incomplete' if mode == 'sentence' and not complete else ('observed' if events else 'no_context')
                pending[f'unigram_coverage_{mode}'].append(((word, time, domain, status), 1))
                for context, side, distance in events:
                    event = (word, time, domain, context)
                    pending[f'unigrams_{mode}'].append((event, 1))
        for first, second in zip(tokens, tokens[1:]):
            if not lexical(first) or not lexical(second):
                continue
            modifier, head = lemma(first, proper_nouns), lemma(second, proper_nouns)
            pending['bigram_counts'].append(((modifier, head, time, domain), 1))
            key = (modifier, head, time, domain)
            for side, distance, status in opportunities(sentence, first.i, second.i + 1, uncertain_left, uncertain_right):
                pending['bigram_opportunities_local5'].append(((*key, side, distance, status), 1))
            pending['bigram_document_counts'].append(((*key, record.get('document_uid', record['doc_id'])), 1))
            pending['bigram_annotations'].append(((*key, first.text, second.text, first.pos_, second.pos_,
                                                  int(first.head.i == second.i), first.dep_,
                                                  first.ent_type_ or 'NONE', second.ent_type_ or 'NONE', int(complete)), 1))
            stats['bigram_occurrences'] += 1
            stats['eligible_pair_occurrences'] += int(pos(first, proper_nouns) in {'ADJ', 'NOUN'}
                                                       and pos(second, proper_nouns) == 'NOUN')
            for mode in MODES:
                events = list(context_events(sentence, first.i, second.i + 1, mode, proper_nouns)) if mode == 'local5' or complete else []
                status = 'gap_incomplete' if mode == 'sentence' and not complete else ('observed' if events else 'no_context')
                pending[f'bigram_coverage_{mode}'].append(((*key, status), 1))
                for context, side, distance in events:
                    event = (*key, context)
                    pending[f'bigrams_{mode}'].append((event, 1))
    for name, rows in pending.items():
        store.add(name, rows)
    return stats


def audit_table_path(name):
    """Each context matrix is saved once; diagnostics live beside their mode."""
    for mode in MODES:
        if name.endswith('_' + mode):
            base = name.removesuffix('_' + mode)
            filename = base if base in {'unigrams', 'bigrams'} else name
            return f'feature_inputs/{mode}/{filename}.parquet'
    return f'tables/{name}.parquet'


def copy_count_table(source, destination):
    """Expose identical counts in each mode without another disk copy if possible."""
    temporary = destination.with_suffix('.parquet.tmp')
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def canonical_inputs(store, output, mode, manifest, domains=None, shared_outputs=None):
    """Export all-POS Parquet inputs in bounded row groups; never infer tokens from contexts."""
    output.mkdir(parents=True, exist_ok=True)
    (output / 'corpus_manifest.json').unlink(missing_ok=True)
    mappings = {'unigrams': f'unigrams_{mode}', 'bigrams': f'bigrams_{mode}',
                'unigram_counts': 'unigram_counts', 'bigram_counts': 'bigram_counts',
                f'unigram_coverage_{mode}': f'unigram_coverage_{mode}',
                f'bigram_coverage_{mode}': f'bigram_coverage_{mode}'}
    if mode == 'local5':
        mappings.update({f'{unit}_opportunities_local5': f'{unit}_opportunities_local5'
                         for unit in ['unigram', 'bigram']})
    outputs, statistics, table_files = [], {}, {}
    shared_outputs = {} if shared_outputs is None else shared_outputs
    for name, table in mappings.items():
        destination = output / f'{name}.parquet'
        table_files[name] = destination.name
        if name in shared_outputs:
            source, statistics[name] = shared_outputs[name]
            copy_count_table(source, destination)
        else:
            columns = [column for column in store.schemas[table] if column != 'domain']
            statistics[name] = store.export_parquet(table, destination, columns=columns,
                integer_columns=INTEGER_COUNT_COLUMNS, filters={'domain': domains} if domains else None)
            if name in {'unigram_counts', 'bigram_counts'}:
                shared_outputs[name] = destination, statistics[name]
        outputs.append(artifact(destination, f'{mode}/{name}.parquet'))
    result = {'schema_version': 3, 'extraction_version': VERSION,
        'representation_version': 5, 'format': 'parquet',
        'parser_policy': manifest, 'raw_counts': True, 'context_mode': mode,
        'classification': 'all_lexical_adjacent_bigrams; eligible compounds derived before POS pooling',
        'context_dimensions': 'lemma_POS; side/distance pooled in context matrices',
        'domains': domains or 'pooled', 'domain_column': 'omitted; selected domains pooled',
        'time_column': 'time', 'word_target_pos': 'all_lexical', 'bigram_target_pos': 'all_lexical',
        'count_population': 'all observed lexical occurrences, independent of context availability',
        'table_files': table_files}
    parent = manifest.get('source_info', {})
    # Count/context matrices come directly from this extraction's transient DB;
    # do not claim discarded CSVs as source_info inputs.
    result['source_info'] = source_info('feature_inputs', parent.get('corpus', {'name': 'COHA', 'release': None}),
        parent.get('inputs', []),
        {key: result[key] for key in ['context_mode', 'domains', 'table_files', 'word_target_pos',
          'bigram_target_pos', 'context_dimensions', 'classification', 'count_population', 'format']},
        parent.get('time', {'column': 'time', 'unit': 'calendar_year'}), pipeline_software(),
        model=parent.get('model'), outputs=outputs, statistics={'tables': statistics})
    validate_feature_input_manifest(result)
    write_json(output / 'corpus_manifest.json', result)
    return result


def initialize_worker(model, max_length):
    """Each spawned process owns one model; spaCy itself stays single-process."""
    global _WORKER_SPACY
    _WORKER_SPACY = spacy.load(model)
    _WORKER_SPACY.max_length = max_length


@contextmanager
def sqlite_scratch(directory):
    """Configure spill storage while this process owns the SQLite connections.

    temp_store_directory is deprecated/process-global, but an environment-only
    override does not work in every build. Set and verify it BEFORE count stores
    open, restore it AFTER they close, and never use concurrent SQLite threads.
    Each spawned worker configures its own process independently.
    """
    directory = str(Path(directory).resolve())
    previous = os.environ.get('SQLITE_TMPDIR')
    os.environ['SQLITE_TMPDIR'] = directory

    def configure(value):
        connection = sqlite3.connect(':memory:')
        try:
            old = connection.execute('PRAGMA temp_store_directory').fetchone()
            # PRAGMA values do not accept parameter binding; quote the SQL string.
            quoted = value.replace("'", "''")
            connection.execute(f"PRAGMA temp_store_directory='{quoted}'")
            actual = connection.execute('PRAGMA temp_store_directory').fetchone()
            if value and actual != (value,):
                raise RuntimeError('This SQLite build cannot configure scratch for SQL temporary files. '
                                   'Use a build supporting PRAGMA temp_store_directory; refusing to '
                                   'silently fall back to a system temporary directory.')
            return old[0] if old else ''
        finally:
            connection.close()

    old_directory = None
    try:
        old_directory = configure(directory)
        yield
    finally:
        try:
            if old_directory is not None:
                configure(old_directory)
        finally:
            if previous is None:
                os.environ.pop('SQLITE_TMPDIR', None)
            else:
                os.environ['SQLITE_TMPDIR'] = previous


@contextmanager
def worker_thread_environment():
    # Spawned interpreters read these before importing NumPy/spaCy. Avoid N
    # processes each starting a full BLAS/OpenMP thread pool.
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: '1' for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def count_shard(shard_id, records, directory, batch_size, proper_nouns):
    # Worker processes have their own SQLite global state; do not rely solely
    # on the parent's environment or on settings surviving a previous task.
    with sqlite_scratch(directory):
        return _count_shard(shard_id, records, directory, batch_size, proper_nouns)


def _count_shard(shard_id, records, directory, batch_size, proper_nouns):
    """Parse, enumerate contexts, and aggregate entirely inside a worker."""
    database = Path(directory) / f'shard-{shard_id:08d}.sqlite'
    sentences = Path(directory) / f'shard-{shard_id:08d}.jsonl.gz'
    store = CountStore(database, disk_temp=True)
    stats = Counter()
    try:
        for name, columns in schemas().items():
            store.create(name, columns)
        with gzip_writer(sentences) as writer:
            for number, (doc, record) in enumerate(
                    _WORKER_SPACY.pipe(records, as_tuples=True, n_process=1, batch_size=batch_size), 1):
                stats.update(add_document(store, doc, record, writer, proper_nouns))
                if number % batch_size == 0:
                    store.connection.commit()
        store.connection.commit()
    except Exception as error:
        raise RuntimeError(f'Extraction failed in shard {shard_id}: {error}') from error
    finally:
        store.close()
    return {'database': str(database), 'sentences': str(sentences),
            'stats': dict(stats), 'fragments': len(records)}


def extract_shards(args, fragment_path, directory, store, writer):
    """Bound outstanding tasks and consume results in input order."""
    stats = Counter()
    stream = iter(read_fragments(fragment_path))
    shard_size = getattr(args, 'shard_size', 4096)
    processed = 0
    started = time.monotonic()
    with worker_thread_environment(), ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(args.model, args.max_length)) as pool:
        pending = deque()
        shard_id = 0

        def submit_next():
            nonlocal shard_id
            records = list(islice(stream, shard_size))
            if not records:
                return False
            pending.append(pool.submit(count_shard, shard_id, records, str(directory),
                                       args.batch_size, args.proper_nouns))
            shard_id += 1
            return True

        try:
            for _ in range(args.workers):
                if not submit_next():
                    break
            while pending:
                result = pending.popleft().result()
                # Refill before merging so annotation/counting can overlap the
                # serial merge. At most workers + one merging shard are on disk.
                submit_next()
                store.merge_from(result['database'])
                with gzip.open(result['sentences'], 'rt', encoding='utf-8', newline='') as source:
                    shutil.copyfileobj(source, writer)
                stats.update(result['stats'])
                processed += result['fragments']
                Path(result['database']).unlink()
                Path(result['sentences']).unlink()
                elapsed = time.monotonic() - started
                print(f'Parsed and merged {processed} fragments ({processed / max(elapsed, 0.001):.1f}/s)',
                      flush=True)
        finally:
            for future in pending:
                future.cancel()
            stream.close()
    return stats


def extract_serial(args, spacy_model, fragment_path, store, writer):
    """Keep the original direct path for one worker, avoiding redundant merges."""
    stats = Counter()
    for number, (doc, record) in enumerate(spacy_model.pipe(read_fragments(fragment_path), as_tuples=True,
                                                   n_process=1, batch_size=args.batch_size), 1):
        stats.update(add_document(store, doc, record, writer, args.proper_nouns))
        if number % args.batch_size == 0:
            store.connection.commit()
            print(f'Parsed {number} fragments', flush=True)
    store.connection.commit()
    return stats


def extract(args):
    # Fail before expensive annotation if the Parquet dependency is unavailable.
    import pyarrow.parquet
    prepared = Path(args.input)
    output = Path(args.output)
    scratch = Path(getattr(args, 'scratch_dir', None) or output).expanduser().resolve()
    validate_stage_paths(prepared, output, scratch)
    input_manifest = json.loads((prepared / 'prepared_manifest.json').read_text())
    if input_manifest['schema_version'] != VERSION:
        raise ValueError('Prepare raw text with the v3 prepare command first; legacy sentence pickles are unsupported')
    fragment_path = prepared / 'fragments.jsonl.gz'
    if digest_file(fragment_path) != input_manifest['fragments_sha256']:
        raise ValueError('Prepared fragments do not match their manifest')
    output.mkdir(parents=True, exist_ok=True)
    if any(any((output / subdir).rglob(pattern))
           for subdir in ['feature_inputs', 'tables'] for pattern in ['*.pkl', '*.csv.gz']):
        raise ValueError('Old extraction files exist in this output directory. Use a fresh output directory for the all-POS Parquet layout.')
    scratch.mkdir(parents=True, exist_ok=True)
    (output / 'extraction_manifest.json').unlink(missing_ok=True)
    spacy_model = spacy.load(args.model)
    spacy_model.max_length = args.max_length
    model_path = Path(spacy_model.path)
    model_hashes = {str(p.relative_to(model_path)): digest_file(p) for p in sorted(model_path.rglob('*')) if p.is_file()}
    manifest = {'schema_version': VERSION, 'input': input_manifest, 'spacy_version': spacy.__version__,
                'model_meta': spacy_model.meta, 'model_sha256': model_hashes,
                'proper_nouns': args.proper_nouns,
                'modes': list(MODES), 'local_window': 3, 'context_anchor': 'pair_span_or_individual_word',
                'sentence_policy': 'exclude first sentence after a gap; exclude last before a gap unless visible terminal punctuation',
                'context_pos': sorted(CONTEXT_POS), 'word_target_pos': 'all_lexical',
                'representation_version': 5, 'bigram_target_pos': 'all_lexical', 'table_format': 'parquet',
                'code_sha256': digest_file(__file__), 'count_store_sha256': digest_file(Path(count_store_implementation.__file__)),
                'time_column': 'time',
                'execution': {'strategy': ('worker_parse_and_count_then_ordered_integer_merge'
                                           if args.workers > 1 else 'serial_parse_and_count'),
                              'workers': args.workers, 'batch_size': args.batch_size,
                              'shard_size': getattr(args, 'shard_size', 4096),
                              'scratch_dir': str(scratch),
                              'sqlite_temp_store': 'FILE',
                              'sqlite_temp_directory_policy': 'verified process-local PRAGMA plus SQLITE_TMPDIR',
                              'worker_numeric_threads': 1 if args.workers > 1 else 'environment',
                              'start_method': 'spawn' if args.workers > 1 else None}}
    parent = input_manifest.get('source_info', {})
    parameters = {key: manifest[key] for key in ['proper_nouns', 'modes', 'local_window',
                  'context_anchor', 'sentence_policy', 'context_pos', 'word_target_pos', 'bigram_target_pos', 'representation_version', 'table_format']}
    parameters.update({'max_length': args.max_length, 'feature_domains': args.domains or 'pooled',
                       'context_vocabulary': 'all observed contexts passing POS rules',
                       'pair_population': 'all observed lexical adjacencies; eligible ADJ-NOUN/NOUN-NOUN derived in features',
                       'unigram_population': 'observed non-space, non-punctuation tokens without @'})
    manifest['source_info'] = source_info('extract', parent.get('corpus', {'name': 'COHA', 'release': None}),
        [artifact(prepared / 'prepared_manifest.json'),
         artifact(fragment_path, sha256=input_manifest['fragments_sha256'])], parameters,
        parent.get('time', {'column': 'time', 'unit': 'calendar_year', 'precision': 'see prepared records'}),
        pipeline_software(), model={'name': args.model, 'metadata': spacy_model.meta, 'files_sha256': model_hashes},
        execution=manifest['execution'])
    if args.workers > 1:
        del spacy_model
        gc.collect()
    stats = Counter()
    for mode in MODES:
        (output / 'feature_inputs' / mode / 'corpus_manifest.json').unlink(missing_ok=True)
    print(f'Extracting with {args.workers} parse/count workers; scratch: {scratch}', flush=True)
    with tempfile.TemporaryDirectory(dir=scratch, prefix='.coha-v3-') as temporary, sqlite_scratch(temporary):
        manifest['execution']['sqlite_tmpdir'] = temporary
        manifest['source_info']['execution']['sqlite_tmpdir'] = temporary
        print(f'Count databases and SQLite temporary files: {temporary}; final outputs: {output}', flush=True)
        store = CountStore(Path(temporary) / 'counts.sqlite', disk_temp=True)
        try:
            for name, columns in schemas().items():
                store.create(name, columns)
            with gzip_writer(output / 'sentences.jsonl.gz.tmp') as writer:
                if args.workers == 1:
                    stats = extract_serial(args, spacy_model, fragment_path, store, writer)
                else:
                    stats = extract_shards(args, fragment_path, temporary, store, writer)
            store.connection.commit()
            (output / 'sentences.jsonl.gz.tmp').replace(output / 'sentences.jsonl.gz')
            sentence_artifact = artifact(output / 'sentences.jsonl.gz')
            manifest['source_info']['outputs'].append(sentence_artifact)
            manifest['stats'] = dict(stats)
            manifest['source_info']['statistics'] = dict(stats)
            tables = output / 'tables'
            tables.mkdir(exist_ok=True)
            print('All fragments merged; exporting Parquet count tables', flush=True)
            shared_outputs = {}
            for name in ['bigram_counts', 'bigram_document_counts', 'bigram_annotations']:
                path = tables / f'{name}.parquet'
                columns = [column for column in store.schemas[name] if column != 'domain']
                statistics = store.export_parquet(name, path, columns=columns,
                    integer_columns=INTEGER_COUNT_COLUMNS,
                    filters={'domain': args.domains} if args.domains else None)
                manifest['source_info']['outputs'].append(artifact(path, f'tables/{name}.parquet'))
                if name == 'bigram_counts':
                    shared_outputs[name] = path, statistics
            feature_artifacts = []
            for mode in MODES:
                print(f'Preparing feature inputs: {mode}', flush=True)
                feature_dir = output / 'feature_inputs' / mode
                feature_manifest = canonical_inputs(store, feature_dir, mode, manifest, args.domains, shared_outputs)
                feature_artifacts.extend(feature_manifest['source_info']['outputs'])
                feature_artifacts.append(artifact(feature_dir / 'corpus_manifest.json', f'{mode}/corpus_manifest.json'))
            manifest['source_info']['outputs'].extend(
                {item['path']: item for item in feature_artifacts}.values())
        except (sqlite3.DatabaseError, pd.errors.DatabaseError) as error:
            if 'database or disk is full' in str(error).lower():
                raise RuntimeError(
                    f'SQLite ran out of space during extraction/export. Database and SQL temporary '
                    f'files: {temporary}; output files: {output.resolve()}. '
                    'Check free space and your quota on both filesystems. '
                    'Use --scratch-dir to choose a larger scratch filesystem.') from error
            raise
        finally:
            store.close()
    manifest['stats'] = dict(stats)
    manifest['sentences_sha256'] = sentence_artifact['sha256']
    write_json(output / 'extraction_manifest.json', manifest)
    print(f'Extracted both modes: {dict(stats)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    sub = parser.add_subparsers(dest='command', required=True)

    prep = sub.add_parser('prepare', help='Read raw COHA linear text, ZIP, or TAR containing ZIP files')
    prep.add_argument('--input', required=True, help='Location of the COHA tar file')
    prep.add_argument('--output', required=True, help = 'Directory where prepared files are stored')
    prep.add_argument('--encoding', default='utf-8-sig')
    prep.add_argument('--corpus-release', help='Exact release label for source_info, e.g. COHA_2023; unknown if omitted')
    prep.add_argument('--metadata', help='Optional TSV with doc_id, time, domain; add source when IDs are reused')
    prep.add_argument('--domains', nargs='+')

    ext = sub.add_parser('extract', help='Parse once and generate local5 and sentence modes')
    ext.add_argument('--input', required=True, help='Directory containing prepared_manifest.json')
    ext.add_argument('--output', required=True, help = 'Directory where extracted files are stored')
    ext.add_argument('--model', default='en_core_web_lg', help= 'Name of spacy model used')
    ext.add_argument('--workers', type=int, default=1, help='Processes that each parse AND count')
    ext.add_argument('--batch-size', type=int, default=256)
    ext.add_argument('--shard-size', type=int, default=4096, help='Fragments per counting task; independent of worker count')
    ext.add_argument('--scratch-dir', help='Directory for worker/merged databases AND SQLite sorting/grouping temporary files; default: output directory')
    ext.add_argument('--max-length', type=int, default=10_000_000)
    ext.add_argument('--proper-nouns', choices=['map', 'exclude'], default='map',
                     help='Map PROPN to NOUN, or exclude PROPN from eligible pairs (retained in general words)')
    ext.add_argument('--domains', nargs='+', help='Domains included in all exported tables; domain columns are pooled away')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.workers < 1 or args.batch_size < 1 or args.shard_size < 1:
            parser.error('workers, batch-size, and shard-size must be positive')
        extract(args)


if __name__ == '__main__':
    main()
