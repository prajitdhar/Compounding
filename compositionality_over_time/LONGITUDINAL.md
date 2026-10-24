# Cumulative longitudinal compositionality analysis

This analysis freezes a modern-trained compositionality model and applies it
to cumulative historical aggregations. Inputs are completed raw feature
preparations and a verified modern CSV snapshot, not raw n-gram shards.
Complete [preprocessing](../google_preprocessing/README.md),
[features](FEATURES.md) and the [regression input snapshot](REGRESSIONS.md) first.

## Stages and paths

From repository root, set `PREPARED_ROOT` to a completed UnTagged preparation
and `MODERN_ROOT` to the verified modern regression snapshot. Use separate
`WORK_ROOT`, `CUMULATIVE_ROOT`, `TEMP_DIRECTORY` and `RESULTS_ROOT` directories.
The preparation must contain independent decade occurrence counts and exact
type-routing caches, not only already PPMI-weighted feature tables.

```bash
bash compositionality_over_time/run_longitudinal.sh plan
bash compositionality_over_time/run_longitudinal.sh prepare
bash compositionality_over_time/run_longitudinal.sh features
bash compositionality_over_time/run_longitudinal.sh fit
```

Default `plan` reads/checks inputs and prints commands without fitting a model.
`check` validates completed aggregations and dependencies. Run the local stages
in order; a successful preparation is required before feature calculation,
and completed features are required before fitting.

## Definitions

Twenty default endpoints cover observations through 1829, 1839, ... 2019.
Each aggregation includes observations from the first source decade through
its endpoint, not only the corresponding decade. `TIME_STEP=10` controls
endpoint spacing, not disjoint-bin width. Final static features must reproduce
the full-period modern inputs.

Raw context counts are accumulated before PPMI is recomputed from cumulative
full-population margins. Distinct partner/type introductions are counted once
within each cumulative prefix; decade productivity counts are not summed.
The ranked vocabulary stays fixed. UnTagged targets pool lexical POS forms
while context dimensions retain tags.

Launchers default to 30000 contexts, UnTagged, PPMI, CompoundAware, cutoff 0,
compound-rating prediction and modified 180/100 rating forms. Scoped temporal
mode supports this configuration at cutoff 0 or 500 only; unsupported temporal
settings are rejected. Static mode supports the base driver's options.

Three temporal predictors compare cumulative compound/modifier/head vectors
at each endpoint with vectors exactly ten years earlier. The policy is
`cumulative_prefix_cosine_lag10_v1`, not the ordinary grid's adjacent-disjoint-bin
definition. Both prefixes are independently PPMI weighted on the same context
basis. Absent/zero vectors or preceding prefixes yield missing values, not zero
similarity. High values measure stability.

Current compound/constituent counts must be strictly above the cutoff. At 0
this still requires positive raw evidence. Missing lagged predictors do not
remove otherwise eligible items: prediction uses frozen modern-training-fitted
imputation. Historical prediction additionally requires attestation and at
least one available fitted predictor. Coverage retains the full held-out
roster and identifies unavailable observations.

## Model and outputs

The Spearman-tuned elastic-net model and preprocessing fit training ratings
only, then freeze. `FEATURE_SET=all` includes static, setting and scoped
temporal predictors; `all_static` excludes the three temporal predictors,
`temporal` uses only them, and `representation_only` excludes setting/temporal
families. Use separate result directories for sensitivities.

Outputs include modern evaluation, tuning records, coefficients, frozen model
and preprocessing, historical trajectories, coverage, growth estimates and
completion receipts. Source identities include authoritative flat shared
implementations; do not alter old receipts to accept new source.

The descriptive growth model uses compound-specific random intercepts/time
slopes and restricted maximum likelihood. It summarizes model-implied scores,
not historical judgments. Cumulative aggregations share observations; usual
intervals do not fully capture dependence or modern-model estimation
uncertainty. Corpus growth, composition and extrapolation may affect trends.
Predictions remain unclipped.

## Plotting saved trajectories

Without refitting or replacing receipt-tracked figures:

```bash
Rscript compositionality_over_time/regression/longitudinal/plot_longitudinal.R \
  --results-dir "$RESULTS_ROOT"
```

This verifies saved hashes and writes `growth_trajectories_styled.png`, a vector
PDF and a record of the plot's source information. It uses decade endpoint
ticks, a color legend and no title/caption. `--output-prefix` selects another
name. Gray lines show frozen-model predictions, blue compound growth fits and
red the population fit with its conditional 95% interval.
