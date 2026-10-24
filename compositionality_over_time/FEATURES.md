# Compositionality features

`features.py` consumes the reusable [preprocessing output](../google_preprocessing/README.md).
It neither downloads nor parses a corpus. Run the commands below from repository
root after installing `.[compositionality]`.

## Inputs

Supply a directory containing the canonical `corpus_manifest.json` specification and
four Parquet datasets:

| Dataset | Columns |
|---|---|
| `bigrams` | modifier, head, time, context, count |
| `bigram_counts` | modifier, head, time, count |
| `unigrams` | word, time, context, count |
| `unigram_counts` | word, time, count |

The corpus-neutral manifest records the schema and source information of these tables.
Counts must be unweighted occurrence counts. Context-event sums
must not be substituted for occurrence frequencies. Also supply the fixed
ranked context-vocabulary TSV and three separately obtained rating TSVs.

```bash
python compositionality_over_time/features.py prepare \
  --inputdir "$FEATURE_INPUT" --outputdir "$FEATURE_OUTPUT" \
  --cache-directory "$CACHE_DIRECTORY" --temp-directory "$TEMP_DIRECTORY" \
  --context-vocabulary "$CONTEXT_VOCABULARY" \
  --reddy90 "$RATINGS_DIR/reddy_90.txt" \
  --cordeiro90 "$RATINGS_DIR/cordeiro_90.txt" \
  --cordeiro100 "$RATINGS_DIR/cordeiro_100.txt" \
  --untagged --threads 1 --memory-limit 8GB
```

Preparation is specific to target-tagging mode. Omit `--untagged` for Tagged
targets and keep its cache/output identities separate. Compatible preparations
are reused; incompatible inputs/settings are rejected.

## Feature grid

| Configuration | Values |
|---|---|
| Context vocabulary | 1000, 3000, 5000, 10000, 30000, 50000 |
| Temporal width | 10000, 10, 20, 50, 100 |
| Constituent representation | CompoundAware, CompoundPooled, CompoundAgnostic |
| Vector weighting | RAW, PPMI |
| Target tagging | Tagged, UnTagged |
| Raw-count cutoff | 0, 10, 50, 100, 500, 1000 |

Use `features` with the same input/preparation arguments and an explicit
`--context-size K --temporal WIDTH`, or a `--grid-index`. The available
`--weights`, `--conditions`, `--cutoffs` and `--families` flags restrict a run.
For example, calculate one prepared configuration directly:

```bash
python compositionality_over_time/features.py features \
  --inputdir "$FEATURE_INPUT" --outputdir "$FEATURE_OUTPUT" \
  --cache-directory "$CACHE_DIRECTORY" --temp-directory "$TEMP_DIRECTORY" \
  --context-vocabulary "$CONTEXT_VOCABULARY" \
  --reddy90 "$RATINGS_DIR/reddy_90.txt" \
  --cordeiro90 "$RATINGS_DIR/cordeiro_90.txt" \
  --cordeiro100 "$RATINGS_DIR/cordeiro_100.txt" \
  --untagged --context-size 5000 --temporal 10 --weights PPMI \
  --conditions CompoundAware --cutoffs 0 --threads 1 --memory-limit 8GB
```

Repeat explicit configurations or iterate over grid indices for a larger
experiment. Run Tagged and UnTagged preparations separately.

## Scientific definitions

- Tagged corpus compounds use ADJ–NOUN/NOUN–NOUN populations. The rating reader
  creates a NOUN–NOUN candidate for every rated expression and an additional
  ADJ–NOUN candidate for adjectival items.
- UnTagged targets pool all lexical bigram POS combinations before weighting,
  grouping and frequency filtering. Context dimensions retain lemma/POS tags.
- Aware constituents retain separate modifier/head roles. Pooled constituents
  combine those roles. Agnostic constituents use general word evidence.
- PPMI uses full eligible population margins, not benchmark-only matrices.
  Raw frequencies, association statistics, productivity and setting token/type
  ratios are not calculated from PPMI weights.
- Productivity counts distinct partner types in the applicable population.
  Wider time bins deduplicate types rather than sum decade productivity.
- Cutoffs are strict raw-count comparisons (`count > cutoff`) shared across
  the compound and required constituents. Zero still requires positive evidence.
- The CPF head weight is bounded to `[0,1]`; `sim_cpf_beta` is recomputed from
  the normalized-constituent mixture. Degenerate/missing cases use the guarded
  calculation, not extrapolating constituent weights.
- Temporal features are adjacent-bin cosine similarities: high means stable.
  Both periods must be eligible; missing decades are not skipped. Width 10000
  pools observations into time 0 and has no genuine adjacent-bin sequence.

Preparation scans raw data into bounded partitioned caches. Each configuration
reuses them without loading the complete corpus into pandas. The configured
memory limit and temporary-disk capacity are separate budgets.

Wide Parquet tables retain rating variants, including missing predictors.
Long tables contain observed eligible pair/time rows. Do not impute before
regression training folds. `regression_inputs.py` creates the verified CSV
snapshot required by R.
