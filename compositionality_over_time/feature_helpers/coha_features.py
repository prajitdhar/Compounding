"""Pure count and sparse-vector calculations for compositionality experiments.

Occurrence tables have no context column. Embedding tables contain target-context
counts. Weighting never mutates input tables or supplies association frequencies.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline_source_info import canonical_time, validate_feature_input_manifest

PAIR = ['modifier', 'head', 'time']
CONDITIONS = ('CompoundAware', 'CompoundPooled', 'CompoundAgnostic')


def aggregate(frame, keys):
    return frame.groupby(keys, as_index=False, sort=True)['count'].sum()


def time_bin(frame, temporal):
    if temporal < 1:
        raise ValueError('Temporal bin size must be positive')
    result = canonical_time(frame)
    result['time'] = result['time'].astype('int64') // temporal * temporal
    return aggregate(result, [c for c in result if c != 'count'])


def ppmi(frame):
    """PPMI within the full supplied target/context population per time bin."""
    keys = [c for c in frame if c != 'count']
    result = aggregate(frame, keys)
    if result.empty:
        return result
    if (result['count'] < 0).any():
        raise ValueError('PPMI requires nonnegative raw counts')
    result = result.loc[result['count'] > 0].copy()
    target = [c for c in keys if c != 'context']
    xy = result['count'].astype(float)
    n = result.groupby('time')['count'].transform('sum').astype(float)
    x = result.groupby(target)['count'].transform('sum').astype(float)
    y = result.groupby(['context', 'time'])['count'].transform('sum').astype(float)
    result['count'] = np.maximum(0, np.log2(xy) + np.log2(n) - np.log2(x) - np.log2(y))
    return result.loc[result['count'] > 0].reset_index(drop=True)


FEATURE_TABLES = ('unigrams', 'unigram_counts', 'bigrams', 'bigram_counts')


def eligible_compounds(frame):
    """Select tagged ADJ-NOUN/NOUN-NOUN BEFORE merging POS variants."""
    return frame.loc[frame.modifier.str.endswith(('_ADJ', '_NOUN')) &
                     frame['head'].str.endswith('_NOUN')].copy()


def read_eligible_contexts(path):
    """Load only eligible rows from an all-POS Parquet context matrix."""
    import pyarrow.parquet as pq
    source = pq.ParquetFile(path)
    frames = []
    try:
        for batch in source.iter_batches(batch_size=65536):
            frame = eligible_compounds(batch.to_pandas())
            if not frame.empty:
                frames.append(frame)
        return (pd.concat(frames, ignore_index=True) if frames
                else source.schema_arrow.empty_table().to_pandas())
    finally:
        source.close()


def load_inputs(directory, tagged=False):
    directory = Path(directory)
    path = directory / 'corpus_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    validate_feature_input_manifest(manifest)
    tables = {name: canonical_time(pd.read_parquet(feature_input_path(directory, name)))
              for name in ['unigrams', 'unigram_counts', 'bigram_counts']}
    tables['compounds'] = canonical_time(
        read_eligible_contexts(feature_input_path(directory, 'bigrams')) if tagged else
        pd.read_parquet(feature_input_path(directory, 'bigrams')))
    tables['compound_counts'] = (eligible_compounds(tables['bigram_counts']) if tagged
                                 else tables['bigram_counts'].copy())
    for name, frame in tables.items():
        if not tagged:
            for column in ['modifier', 'head', 'word']:
                if column in frame:
                    frame[column] = frame[column].str.replace(r'_[A-Z]+$', '', regex=True)
        tables[name] = aggregate(frame, [c for c in frame if c != 'count'])
    return tables


def feature_input_path(directory, name):
    directory = Path(directory)
    manifest = json.loads((directory / 'corpus_manifest.json').read_text())
    return directory / manifest.get('table_files', {}).get(name, f'{name}.parquet')


def condition_matrices(compounds, words):
    modifiers = aggregate(compounds, ['modifier', 'time', 'context'])
    heads = aggregate(compounds, ['head', 'time', 'context'])
    pooled = aggregate(pd.concat([modifiers.rename(columns={'modifier': 'word'}),
                                  heads.rename(columns={'head': 'word'})]),
                       ['word', 'time', 'context'])
    return {'CompoundAware': (modifiers, heads),
            'CompoundPooled': (pooled.rename(columns={'word': 'modifier'}),
                               pooled.rename(columns={'word': 'head'})),
            'CompoundAgnostic': (words.rename(columns={'word': 'modifier'}),
                                 words.rename(columns={'word': 'head'}))}


def condition_counts(compounds, words):
    modifiers = aggregate(compounds, ['modifier', 'time'])
    heads = aggregate(compounds, ['head', 'time'])
    pooled = aggregate(pd.concat([modifiers.rename(columns={'modifier': 'word'}),
                                  heads.rename(columns={'head': 'word'})]), ['word', 'time'])
    return {'CompoundAware': (modifiers, heads),
            'CompoundPooled': (pooled.rename(columns={'word': 'modifier'}),
                               pooled.rename(columns={'word': 'head'})),
            'CompoundAgnostic': (words.rename(columns={'word': 'modifier'}),
                                 words.rename(columns={'word': 'head'}))}


def eligible_pairs(evaluation, counts, constituent_counts, cutoff):
    """One shared evaluation set across all three conditions, based on tokens."""
    selected = evaluation.merge(counts.loc[counts['count'] > cutoff, PAIR], on=PAIR)
    for modifiers, heads in constituent_counts.values():
        selected = selected.merge(modifiers.loc[modifiers['count'] > cutoff, ['modifier', 'time']],
                                  on=['modifier', 'time'])
        selected = selected.merge(heads.loc[heads['count'] > cutoff, ['head', 'time']],
                                  on=['head', 'time'])
    return selected[PAIR].drop_duplicates().sort_values(PAIR).reset_index(drop=True)


def vector_map(frame, keys):
    result = defaultdict(dict)
    for row in frame[[*keys, 'context', 'count']].itertuples(index=False, name=None):
        if row[-1] != 0:
            result[tuple(row[:-2])][row[-2]] = float(row[-1])
    return result


def norm(vector):
    return np.sqrt(sum(v * v for v in vector.values()))


def cosine(left, right):
    denominator = norm(left) * norm(right)
    if denominator == 0:
        return np.nan
    if len(left) > len(right):
        left, right = right, left
    return float(np.clip(sum(v * right.get(k, 0) for k, v in left.items()) / denominator, -1, 1))


def cosine_features(compounds, modifiers, heads, evaluation):
    """Full-support cosine and convex composition; missing vectors remain missing.

    Beta weights the normalized head and is clipped to [0, 1]. Degenerate or
    missing inputs retain the beta=0.5 fallback.
    """
    comp_vectors = vector_map(compounds, PAIR)
    mod_vectors = vector_map(modifiers, ['modifier', 'time'])
    head_vectors = vector_map(heads, ['head', 'time'])
    rows = []
    feature_names = ['sim_with_modifier', 'sim_with_head', 'sim_bw_constituents',
                     'beta', 'geom_mean_sim', 'arith_mean_sim',
                     *[f'sim_cpf_{s}' for s in ['0', '25', '50', '75', '100', 'beta']]]
    for modifier, head, time in evaluation[PAIR].itertuples(index=False, name=None):
        comp = comp_vectors.get((modifier, head, time), {})
        mod = mod_vectors.get((modifier, time), {})
        hd = head_vectors.get((head, time), {})
        cm, ch, mh = cosine(comp, mod), cosine(comp, hd), cosine(mod, hd)
        denominator = (cm + ch) * (1 - mh)
        beta = (ch - cm * mh) / denominator if np.isfinite(denominator) and abs(denominator) > 1e-12 else 0.5
        beta = float(np.clip(beta, 0.0, 1.0))
        values = [cm, ch, mh, beta, np.sqrt(cm * ch), (cm + ch) / 2]
        mod_norm, head_norm = norm(mod), norm(hd)
        for weight in [0, .25, .5, .75, 1, beta]:
            if not mod_norm or not head_norm:
                values.append(np.nan)
                continue
            composed = {c: weight * hd.get(c, 0) / head_norm +
                        (1 - weight) * mod.get(c, 0) / mod_norm for c in mod.keys() | hd.keys()}
            values.append(cosine(comp, composed))
        rows.append([modifier, head, time, *values])
    return pd.DataFrame(rows, columns=PAIR + feature_names)


def association_features(population, targets=None):
    """Exact 2x2 association over a declared population of pair occurrences."""
    result = aggregate(population, PAIR)
    if result.empty:
        return pd.DataFrame(columns=PAIR + ['assoc_pair_freq', 'assoc_left_freq', 'assoc_right_freq', 'assoc_population_size', 'ppmi', 'local_mi', 'log_ratio'])
    # Compute background marginals before selecting evaluation targets.
    left = result.groupby(['modifier', 'time'])['count'].sum().rename('_x')
    right = result.groupby(['head', 'time'])['count'].sum().rename('_y')
    total = result.groupby('time')['count'].sum().rename('_n')
    if targets is not None:
        result = result.merge(targets[PAIR].drop_duplicates(), on=PAIR, validate='one_to_one')
    result = result.join(left, on=['modifier', 'time']).join(right, on=['head', 'time']).join(total, on='time')
    a = result['count'].to_numpy(dtype=float)
    x = result.pop('_x').to_numpy(dtype=float)
    y = result.pop('_y').to_numpy(dtype=float)
    n = result.pop('_n').to_numpy(dtype=float)
    observed = np.column_stack([a, x - a, y - a, n - x - y + a])
    expected = np.column_stack([x * y, x * (n - y), (n - x) * y, (n - x) * (n - y)]) / n[:, None]
    if (observed < 0).any():
        raise ValueError('Inconsistent association population')
    terms = np.zeros_like(observed)
    positive = observed > 0
    terms[positive] = observed[positive] * np.log(observed[positive] / expected[positive])
    pmi = np.log2(a) + np.log2(n) - np.log2(x) - np.log2(y)
    result = result.rename(columns={'count': 'assoc_pair_freq'})
    result['assoc_left_freq'], result['assoc_right_freq'], result['assoc_population_size'] = x, y, n
    result['ppmi'] = np.maximum(0, pmi)
    result['local_mi'] = a * pmi
    result['log_ratio'] = 2 * terms.sum(axis=1)
    return result


def frequency_features(compounds, unigrams, population, evaluation=None, condition='CompoundAgnostic'):
    """Setting-specific raw frequencies; pair marginals use the supplied population."""
    result = aggregate(compounds, PAIR).rename(columns={'count': 'comp_freq'})
    if evaluation is not None:
        result = result.merge(evaluation[PAIR].drop_duplicates(), on=PAIR, validate='one_to_one')
    types = compounds[PAIR].drop_duplicates()
    mod_prod = types.groupby(['modifier', 'time']).size().rename('mod_prod').reset_index()
    head_prod = types.groupby(['head', 'time']).size().rename('head_prod').reset_index()
    result = result.merge(mod_prod, on=['modifier', 'time']).merge(head_prod, on=['head', 'time'])
    totals = types.groupby('time').size().rename('comp_size')
    mod_types = types.groupby('time').modifier.nunique().rename('mod_size')
    head_types = types.groupby('time')['head'].nunique().rename('head_size')
    result = result.join(totals, on='time').join(mod_types, on='time').join(head_types, on='time')
    for side in ['mod', 'head']:
        result[f'{side}_family_size'] = np.log2(result[f'{side}_prod'] / result.comp_size)
        partner_size = result.head_size if side == 'mod' else result.mod_size
        result[f'{side}_family_size_new'] = np.log2(result[f'{side}_prod'] / partner_size)
    modifiers, heads = condition_counts(compounds, unigrams)[condition]
    result = result.merge(modifiers.rename(columns={'count': 'mod_freq'}),
                          on=['modifier', 'time'], how='left')
    result = result.merge(heads.rename(columns={'count': 'head_freq'}),
                          on=['head', 'time'], how='left')
    # Compound vectors/counts describe the selected pair population: all lexical
    # bigrams for UnTagged, ADJ/NOUN-to-NOUN pairs for Tagged, in every setting.
    masses = {'comp': compounds.groupby('time')['count'].sum(),
              'mod': modifiers.groupby('time')['count'].sum(),
              'head': heads.groupby('time')['count'].sum()}
    for prefix in ['comp', 'mod', 'head']:
        result[f'{prefix}_tf'] = np.log2(1 + result[f'{prefix}_freq'])
        result[f'log_{prefix}_freq'] = np.log2(result[f'{prefix}_freq'] / result.time.map(masses[prefix]))
    result['log_mod_freq_new'] = result.log_mod_freq
    result['log_head_freq_new'] = result.log_head_freq
    result = result.drop(columns=['comp_size', 'mod_size', 'head_size'])
    return result.merge(association_features(population, targets=result[PAIR]), on=PAIR, how='left')


def temporal_features(compounds, modifiers, heads, evaluation, temporal):
    """Compare exactly adjacent bins; never bridge an unobserved time interval."""
    comp_vectors = vector_map(compounds, PAIR)
    mod_vectors = vector_map(modifiers, ['modifier', 'time'])
    head_vectors = vector_map(heads, ['head', 'time'])
    rows = []
    for modifier, head, time in evaluation[PAIR].itertuples(index=False, name=None):
        rows.append([modifier, head, time,
                     cosine(comp_vectors.get((modifier, head, time), {}),
                            comp_vectors.get((modifier, head, time - temporal), {})),
                     cosine(mod_vectors.get((modifier, time), {}),
                            mod_vectors.get((modifier, time - temporal), {})),
                     cosine(head_vectors.get((head, time), {}),
                            head_vectors.get((head, time - temporal), {}))])
    return pd.DataFrame(rows, columns=PAIR + ['change_comp', 'change_mod', 'change_head'])


def setting_comparison(vectors, general_vectors, raw, general_raw, counts, general_counts,
                       key, targets, label):
    """One row per constituent/time; types are measured BEFORE PPMI.

    Token evidence is raw occurrences, not target-context event mass. Pooled
    occurrences are role incidences and may exceed general token occurrences.
    """
    keys = [key, 'time']
    result = targets[keys].drop_duplicates().copy()
    left, right = vector_map(vectors, keys), vector_map(general_vectors, keys)
    result[f'sim_bw_settings_{label}'] = [cosine(left.get(k, {}), right.get(k, {}))
        for k in result.itertuples(index=False, name=None)]
    token_label = 'mod' if label == 'modifier' else label
    for kind, numerator, denominator in [
            ('token', counts.set_index(keys)['count'], general_counts.set_index(keys)['count']),
            ('type', raw.loc[raw['count'] > 0].groupby(keys).context.nunique(),
                     general_raw.loc[general_raw['count'] > 0].groupby(keys).context.nunique())]:
        index = pd.MultiIndex.from_frame(result[keys])
        num = numerator.reindex(index).fillna(0).to_numpy(dtype=float)
        den = denominator.reindex(index).to_numpy(dtype=float)
        result[f'perc_{kind}_{token_label}'] = np.divide(num, den, out=np.full(len(num), np.nan),
                                                       where=np.isfinite(den) & (den > 0))
    return result


def context_vocabulary(unigrams):
    """Rank lemma_POS dimensions by mean relative frequency across decades.

    Every observed decade has equal weight; absent contexts contribute zero.
    Select from raw tagged unigrams so target --tag cannot change vocabulary.
    """
    frame = canonical_time(unigrams).copy()
    frame['time'] = frame['time'].astype('int64') // 10 * 10
    decades = sorted(frame.time.unique())
    frame = frame.loc[frame.word.str.contains(r'_(?:NOUN|PROPN|ADJ|VERB|NUM|ADV)$', regex=True)]
    frame = aggregate(frame, ['word', 'time'])
    frame = frame.loc[frame['count'] > 0].copy()
    frame['relative_frequency'] = frame['count'] / frame.groupby('time')['count'].transform('sum')
    ranked = frame.groupby('word').agg(score=('relative_frequency', 'sum'),
                                       decades_observed=('time', 'nunique'), tokens=('count', 'sum')).reset_index()
    ranked['score'] /= max(1, len(decades))
    ranked = ranked.rename(columns={'word': 'context'}).sort_values(
        ['score', 'context'], ascending=[False, True], kind='stable').reset_index(drop=True)
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    return ranked, frame, decades


def context_coverage(vocabulary, decade_counts, decades):
    selected = decade_counts.loc[decade_counts.word.isin(vocabulary)]
    total = decade_counts.groupby('time')['count'].sum()
    retained = selected.groupby('time')['count'].sum()
    types = selected.groupby('time').word.nunique()
    return pd.DataFrame([{'time': int(t), 'eligible_context_tokens': int(total.get(t, 0)),
        'retained_context_tokens': int(retained.get(t, 0)), 'retained_context_types': int(types.get(t, 0)),
        'token_coverage': retained.get(t, 0) / total[t] if total.get(t, 0) else np.nan}
        for t in decades], columns=['time', 'eligible_context_tokens', 'retained_context_tokens',
                                   'retained_context_types', 'token_coverage'])
