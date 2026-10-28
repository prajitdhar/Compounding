"""Feature-table and ratings helpers shared by static and temporal analyses."""
import argparse
import json
import hashlib
import platform
import numpy as np
from pathlib import Path

import pandas as pd

from coha_features import (PAIR, CONDITIONS, FEATURE_TABLES, aggregate, load_inputs, time_bin, ppmi,
                           condition_matrices, condition_counts, eligible_pairs,
                           cosine_features, frequency_features, temporal_features, feature_input_path,
                           setting_comparison, context_vocabulary, context_coverage)
from pipeline_source_info import artifact, source_info, software_record
from table_io import write_parquet


def parse_args(temporal=False):
    parser = argparse.ArgumentParser(description='Compositionality features from independent raw counts and context matrices')
    parser.add_argument('--inputdir', required=True, help='Directory containing the extracted manifest file. Normally hardcoded in Makefile')
    parser.add_argument('--outputdir', required=True, help='Directory where the feature tables are to be stored. Normally hardcoded in Makefile')
    for name in ['reddy90', 'cordeiro90', 'cordeiro100']:
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--tag', action='store_true')
    parser.add_argument('--ppmi', action='store_true')
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[0, 10, 50, 100, 500, 1000])
    parser.add_argument('--temporal', type=int, nargs='+',
                        default=[10_000, 10, 20, 50, 100] if temporal else [10000, 10, 20, 50, 100])
    parser.add_argument('--association-population', choices=['by-setting'], default='by-setting',
                        help='Aware/Pooled use eligible compounds; Agnostic uses all lexical bigrams')
    parser.add_argument('--context-sizes', '--context-size', type=int, nargs='+', default=[0],
                        help='Fixed decade-balanced vocabulary sizes, e.g. 5000 10000 25000 50000; 0 keeps all dimensions')
    if temporal:
        parser.add_argument('--plot', action='store_true', help='Plot mean adjacent-bin cosine for each condition')
    args = parser.parse_args()
    if min(args.cutoffs) < 0 or min(args.temporal) < 1 or min(args.context_sizes) < 0:
        parser.error('Cutoffs/context sizes must be nonnegative and time bins positive')
    return args


def read_ratings(args):
    frames = []
    for name in ['reddy90', 'cordeiro90', 'cordeiro100']:
        frame = pd.read_csv(getattr(args, name), sep='\t')
        frame['source'] = name
        frames.append(frame)
    ratings = pd.concat(frames, ignore_index=True)
    if args.tag:
        noun = ratings.copy()
        noun['modifier'] += '_NOUN'
        noun['head'] += '_NOUN'
        if 'is_adj' in ratings:
            mixed = ratings.copy()
            adjective = mixed.is_adj.astype(str).str.lower().isin(['true', '1'])
            mixed['modifier'] += adjective.map({True: '_ADJ', False: '_NOUN'})
            mixed['head'] += '_NOUN'
            ratings = pd.concat([noun, mixed], ignore_index=True).drop_duplicates()
        else:
            ratings = noun
    return ratings


def restrict_vectors(frame, evaluation, keys):
    return frame.merge(evaluation[keys].drop_duplicates(), on=keys, how='inner', validate='many_to_one')


def write_features(features, ratings, selected, path):
    # Long form retains exactly the shared selected pair/time observations.
    write_parquet(features, path.with_name(path.stem + '_long.parquet'))
    marked = features.merge(selected.assign(eligible=1), on=PAIR, how='right')
    values = [c for c in marked if c not in PAIR]
    if marked.empty:
        wide = ratings.copy()
    else:
        wide = marked.set_index(PAIR)[values].unstack('time')
        wide.columns = [f'{feature}:{time}' for feature, time in wide.columns]
        wide = ratings.merge(wide.reset_index(), on=['modifier', 'head'], how='left')
    write_parquet(wide, path)


def run(args, temporal_mode=False):
    if getattr(args, 'association_population', 'by-setting') != 'by-setting':
        raise ValueError('Use --association-population by-setting (the default); a global population is no longer supported')
    sizes = list(dict.fromkeys(getattr(args, 'context_sizes', [0])))
    if not sizes or min(sizes) < 0:
        raise ValueError('Context sizes must be nonnegative')
    tables = load_inputs(args.inputdir, tagged=args.tag)
    # Always rank tagged raw unigrams, independent of target --tag and bin size.
    ranked, decade_counts, decades = context_vocabulary(
        pd.read_parquet(feature_input_path(args.inputdir, 'unigram_counts')))
    for size in sizes:
        vocabulary = ranked.head(size) if size else ranked
        selected_tables = dict(tables)
        if size:
            for name in ['compounds', 'unigrams']:
                selected_tables[name] = tables[name].loc[tables[name].context.isin(vocabulary.context)].copy()
        run_configuration(args, temporal_mode, selected_tables, size, vocabulary,
                          context_coverage(vocabulary.context, decade_counts, decades), decades)


def run_configuration(args, temporal_mode, tables, context_size, vocabulary, coverage, decades):
    output = Path(args.outputdir)
    output.mkdir(parents=True, exist_ok=True)
    mode = 'temporal' if temporal_mode else 'features'
    weight_name = 'PPMI' if args.ppmi else 'RAW'
    tag_name = ('Tagged' if args.tag else 'UnTagged') + (f'_ctx{context_size}' if context_size else '')
    manifest_path = output / f'{mode}_{weight_name}_{tag_name}_manifest.json'
    manifest_path.unlink(missing_ok=True)
    ratings = read_ratings(args)
    evaluation_pairs = ratings[['modifier', 'head']].drop_duplicates()
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ['coha_feature_pipeline.py', 'coha_features.py', 'table_io.py']}
    manifest = {'schema_version': 3, 'format': 'parquet', 'conditions': list(CONDITIONS),
                'code_sha256': code_hashes, 'python_version': platform.python_version(),
                'pandas_version': pd.__version__, 'numpy_version': np.__version__,
                'compound_vectors': 'identical compound occurrences in all conditions',
                'compound_population': 'adj_noun_and_noun_noun' if args.tag else 'all_lexical_bigrams',
                'association_population': {'CompoundAware': 'eligible' if args.tag else 'bigram_counts',
                                           'CompoundPooled': 'eligible' if args.tag else 'bigram_counts',
                                           'CompoundAgnostic': 'bigram_counts'},
                'context_vocabulary': {'requested_size': context_size, 'actual_size': len(vocabulary),
                    'selection': 'mean within-decade relative frequency among context-POS unigrams; lexical tie-break',
                    'decades': [int(t) for t in decades], 'fixed_across_time_bins_and_settings': True},
                'setting_features': {'cosine': 'current RAW/PPMI weighting, full vector norms',
                    'token_ratio': 'raw role incidences / raw unigram occurrences',
                    'type_ratio': 'distinct observed contexts after vocabulary selection, before PPMI',
                    'pooled_pair_columns': 'constituent_mod and constituent_head; no averaging'},
                'frequency_definitions': {'mod_freq/head_freq': 'role counts (Aware), sum of roles (Pooled), unigrams (Agnostic)',
                    'comp_freq': 'selected pair population occurrences in all settings',
                    'assoc_pair_freq': 'joint count in setting-specific association population',
                    'association_marginals': 'left/right bigram-position marginals; distinct from unigram frequencies'},
                'ppmi': args.ppmi, 'ppmi_background': 'full per-condition event population before cutoff or rating filtering',
                'frequency_source': 'raw occurrence counts', 'cutoff_comparison': 'strictly greater than',
                'cutoffs': args.cutoffs, 'temporal': args.temporal,
                'imputation': 'none; fit inside training folds',
                'time_column': 'time', 'tagged_targets': args.tag,
                'input_manifest': json.loads((Path(args.inputdir) / 'corpus_manifest.json').read_text())}
    input_manifest_path = Path(args.inputdir) / 'corpus_manifest.json'
    parent = manifest['input_manifest'].get('source_info', {})
    input_paths = sorted({feature_input_path(args.inputdir, name) for name in FEATURE_TABLES})
    inputs = [artifact(input_manifest_path)] + [artifact(path) for path in input_paths]
    inputs += [artifact(getattr(args, name), name) for name in ['reddy90', 'cordeiro90', 'cordeiro100']]
    parameters = {key: manifest[key] for key in ['format', 'conditions', 'compound_population', 'association_population', 'ppmi',
                  'ppmi_background', 'frequency_source', 'cutoff_comparison', 'cutoffs',
                  'temporal', 'imputation', 'tagged_targets', 'context_vocabulary', 'setting_features', 'frequency_definitions']}
    parameters['plot'] = getattr(args, 'plot', False)
    manifest['source_info'] = source_info(mode, parent.get('corpus', {'name': 'unspecified', 'release': None}),
        inputs, parameters,
        {'column': 'time', 'unit': 'calendar_year', 'source': parent.get('time'),
         'bin_sizes': args.temporal, 'binning': 'floor(time / temporal) * temporal',
         'pooled_bin': {'temporal': 10000, 'time': 0}},
        software_record([__file__, Path(__file__).with_name('coha_features.py'),
                         Path(__file__).with_name('pipeline_source_info.py'),
                         Path(__file__).with_name('table_io.py')], ['pandas', 'numpy', 'pyarrow']),
        model=parent.get('model'))

    vocabulary_path = output / f'{mode}_{weight_name}_{tag_name}_contexts.parquet'
    write_parquet(vocabulary, vocabulary_path)
    coverage_path = output / f'{mode}_{weight_name}_{tag_name}_context_coverage.parquet'
    write_parquet(coverage, coverage_path)
    manifest['source_info']['outputs'].extend([artifact(vocabulary_path), artifact(coverage_path)])

    for temporal in args.temporal:
        print(f'Preparing time bins: {temporal}', flush=True)
        binned = {name: time_bin(frame, temporal) for name, frame in tables.items()}
        counts = binned['compound_counts']
        raw_conditions = condition_matrices(binned['compounds'], binned['unigrams'])
        occurrence_conditions = condition_counts(counts, binned['unigram_counts'])
        evaluation = evaluation_pairs.merge(pd.DataFrame({'time': sorted(counts.time.unique())}), how='cross')
        # Weight the complete population ONCE per binning. Cutoffs select targets
        # only; they never change PPMI marginals or operate on PPMI weights.
        compound_matrix = ppmi(binned['compounds']) if args.ppmi else binned['compounds'].copy()
        compound_matrix = restrict_vectors(compound_matrix, evaluation, PAIR)
        matrices = {}
        for condition, (modifiers, heads) in raw_conditions.items():
            if args.ppmi:
                modifiers, heads = ppmi(modifiers), ppmi(heads)
            matrices[condition] = (restrict_vectors(modifiers, evaluation, ['modifier', 'time']),
                                   restrict_vectors(heads, evaluation, ['head', 'time']))
        controls = {}
        setting_tables = {}
        if not temporal_mode:
            for condition in CONDITIONS:
                population = binned['bigram_counts'] if condition == 'CompoundAgnostic' else counts
                controls[condition] = frequency_features(counts, binned['unigram_counts'], population,
                                                         evaluation=evaluation, condition=condition)
            for condition in ['CompoundAware', 'CompoundPooled']:
                comparisons = []
                for index, key in enumerate(['modifier', 'head']):
                    label = key if condition == 'CompoundAware' else 'constituent'
                    comparisons.append(setting_comparison(
                        matrices[condition][index], matrices['CompoundAgnostic'][index],
                        raw_conditions[condition][index], raw_conditions['CompoundAgnostic'][index],
                        occurrence_conditions[condition][index], occurrence_conditions['CompoundAgnostic'][index],
                        key, evaluation, label))
                if condition == 'CompoundPooled':
                    constituents = pd.concat([table.rename(columns={key: 'word'})
                        for table, key in zip(comparisons, ['modifier', 'head'])]).drop_duplicates().sort_values(['word', 'time'])
                    path = output / f'setting_constituents_{weight_name}_{tag_name}_{temporal}.parquet'
                    write_parquet(constituents, path)
                    manifest['source_info']['outputs'].append(artifact(path))
                    comparisons = [table.rename(columns={column: f'{column}_{suffix}'
                        for column in table if column not in [key, 'time']})
                        for table, key, suffix in zip(comparisons, ['modifier', 'head'], ['mod', 'head'])]
                setting_tables[condition] = comparisons
        for cutoff in args.cutoffs:
            selected = eligible_pairs(evaluation, counts, occurrence_conditions, cutoff)
            print(f'Cutoff {cutoff}: {len(selected)} shared pair/time observations', flush=True)
            comp = restrict_vectors(compound_matrix, selected, PAIR)
            for condition in CONDITIONS:
                modifiers, heads = matrices[condition]
                modifiers = restrict_vectors(modifiers, selected, ['modifier', 'time'])
                heads = restrict_vectors(heads, selected, ['head', 'time'])
                if temporal_mode:
                    features = temporal_features(comp, modifiers, heads, selected, temporal)
                    if getattr(args, 'plot', False) and not features.empty:
                        import matplotlib.pyplot as plt
                        axis = features.groupby('time')[['change_comp', 'change_mod', 'change_head']].mean().plot()
                        axis.set_ylabel('Cosine with previous time bin')
                        plot_path = output / f'{mode}_{condition}_{weight_name}_{tag_name}_{temporal}_{cutoff}.png'
                        axis.figure.savefig(plot_path)
                        manifest['source_info']['outputs'].append(artifact(plot_path))
                        plt.close(axis.figure)
                else:
                    features = cosine_features(comp, modifiers, heads, selected)
                    features = features.merge(controls[condition], on=PAIR, how='left', validate='one_to_one')
                    for key, comparison in zip(['modifier', 'head'], setting_tables.get(condition, [])):
                        features = features.merge(comparison, on=[key, 'time'], how='left', validate='many_to_one')
                path = output / f'{mode}_{condition}_withSetting_{weight_name}_{tag_name}_{temporal}_{cutoff}_na.parquet'
                write_features(features, ratings, selected, path)
                manifest['source_info']['outputs'].extend([
                    artifact(path), artifact(path.with_name(path.stem + '_long.parquet'))])
    manifest['source_info']['statistics'] = {'output_files': len(manifest['source_info']['outputs']),
                                          'evaluation_pairs': len(evaluation_pairs)}
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)
