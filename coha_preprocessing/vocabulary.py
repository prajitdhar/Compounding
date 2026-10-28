#!/usr/bin/env python3
"""Rank fixed COHA contexts from independent raw unigram occurrence counts."""

import argparse
import csv
import json
from pathlib import Path

from src.io import atomic_path
from src.partitioned_parquet import connection_scope
from src.source_info import (artifact, software_record, source_info,
                             validate_feature_input_manifest, write_json)
from src.context_vocabulary import equal_decade_ranking_query
from src import context_vocabulary as _vocabulary_implementation


CONTEXT_PATTERN = r'_(NOUN|PROPN|ADJ|VERB|NUM|ADV)$'


def vocabulary_size(value):
    if str(value).lower() == 'all':
        return 0
    return int(value)


def build_vocabulary(args):
    """Use equal-decade mean relative frequency, with deterministic tie ordering.

    The denominator includes all observed decades, including a decade without
    eligible contexts. Within a decade only positive aggregated counts of the
    eligible context POS enter its token denominator. SQL aggregation and sorting
    use the explicit memory/thread budget and can spill to the selected directory.
    """
    if args.size < 0 or args.threads < 1:
        raise ValueError('Vocabulary size must be nonnegative and threads positive')
    directory = Path(args.input)
    manifest_path = directory / 'corpus_manifest.json'
    manifest = validate_feature_input_manifest(json.loads(manifest_path.read_text()))
    counts = directory / manifest['table_files']['unigram_counts']
    count_record = artifact(counts, 'unigram_counts')
    identities = []
    artifact_names = {counts.name, 'unigram_counts', 'unigram_counts.parquet'}
    for item in manifest['source_info'].get('outputs', []):
        recorded_path = item.get('path') or ''
        recorded_name = item.get('name') or ''
        if ((recorded_path and Path(recorded_path).resolve() == counts.resolve())
                or Path(recorded_path).name in artifact_names
                or Path(recorded_name).name in artifact_names):
            identities.append(item)
    if not identities:
        raise ValueError('Corpus manifest must identify its unigram-count artifact')
    if len(identities) != 1:
        raise ValueError('Ambiguous unigram-count artifacts in corpus manifest')
    expected_hash = identities[0].get('sha256')
    if not expected_hash:
        raise ValueError('Corpus manifest must include the unigram-count checksum')
    if expected_hash != count_record['sha256']:
        raise ValueError('Unigram counts do not match their corpus manifest')
    output = Path(args.output)
    input_paths = {(directory / name).resolve()
                   for name in manifest['table_files'].values()}
    input_paths.add(manifest_path.resolve())
    receipt_path = Path(str(output) + '.source_info.json')
    if {output.resolve(), receipt_path.resolve()} & input_paths:
        raise ValueError('Vocabulary output must not replace its inputs')
    receipt_path.unlink(missing_ok=True)
    options = {'threads': args.threads, 'memory_limit': args.memory_limit,
               'max_temp_directory_size': args.max_temp_directory_size}
    scratch = Path(args.temp_directory or output.parent / '.vocabulary_tmp')
    with connection_scope(options, scratch) as connection:
        connection.read_parquet(str(counts)).create_view('raw_counts')
        decades = [row[0] for row in connection.execute(
            'SELECT DISTINCT (floor(time / 10.0) * 10)::BIGINT AS decade '
            'FROM raw_counts ORDER BY decade').fetchall()]
        connection.execute('''CREATE TEMP TABLE decade_counts AS
            SELECT word AS context, (floor(time / 10.0) * 10)::BIGINT AS decade,
                   SUM(count)::BIGINT AS count
            FROM raw_counts WHERE regexp_matches(word, ?)
            GROUP BY word, decade HAVING SUM(count) > 0''', [CONTEXT_PATTERN])
        # Every observed decade has equal weight; missing word/decade cells
        # contribute zero through the total number of observed decades.
        ranking_query = equal_decade_ranking_query(
            'decade_counts', ['context'], observed_decades=max(1, len(decades)))
        connection.execute('CREATE TEMP TABLE ranked AS ' + ranking_query)
        query = 'SELECT rank, context, score, decades_observed, tokens FROM ranked ORDER BY rank'
        if args.size:
            query += f' LIMIT {int(args.size)}'
        # Stream rows to a small Python buffer; do not materialize the vocabulary
        # or the much larger raw unigram table as a pandas DataFrame.
        with atomic_path(output) as temporary:
            cursor = connection.execute(query)
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
                writer.writerow(['rank', 'context', 'score', 'decades_observed', 'tokens'])
                selected = 0
                while rows := cursor.fetchmany(65536):
                    writer.writerows(rows)
                    selected += len(rows)
    parent = manifest['source_info']
    software = software_record([__file__, _vocabulary_implementation.__file__], ['duckdb'])
    receipt = source_info('context_vocabulary', parent.get('corpus', {'name': 'COHA'}),
        [artifact(manifest_path), count_record],
        {'policy': 'mean relative frequency across all observed decades',
         'context_pos': ['ADJ', 'ADV', 'NOUN', 'NUM', 'PROPN', 'VERB'],
         'rank_order': ['score descending', 'context ascending'],
         'size': args.size, 'source_context_mode': manifest.get('context_mode')},
        parent.get('time', {'column': 'time', 'unit': 'calendar_year'}),
        software,
        outputs=[artifact(output)],
        statistics={'observed_decades': decades, 'selected_contexts': selected},
        execution={**options, 'temp_directory': str(scratch.resolve())})
    write_json(receipt_path, {'source_info': receipt})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Feature-input mode directory with corpus_manifest.json')
    parser.add_argument('--output', required=True, help='Ranked TSV for --context-vocabulary')
    parser.add_argument('--size', type=vocabulary_size, default=0,
                        help='Maximum contexts; 0 or all retains all (default)')
    parser.add_argument('--memory-limit', default='1GB', help='DuckDB buffer memory budget')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--temp-directory', help='DuckDB spill root; default: output-parent/.vocabulary_tmp')
    parser.add_argument('--max-temp-directory-size', default='100GB', help='DuckDB spill-space cap')
    args = parser.parse_args()
    if args.size < 0 or args.threads < 1:
        parser.error('--size must be nonnegative and --threads positive')
    result = build_vocabulary(args)
    print(f"Ranked {result['statistics']['selected_contexts']} contexts: {args.output}")


if __name__ == '__main__':
    main()
