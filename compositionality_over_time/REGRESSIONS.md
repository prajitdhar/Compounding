# Regression experiments

This stage analyses compositionality features; it does not preprocess a corpus.
Complete [Google](../google_preprocessing/README.md) or
[COHA preprocessing](../coha_preprocessing/README.md), then
[feature generation](FEATURES.md) first. Commands below run from repository root.

## Environment and immutable inputs

Install `.[compositionality]` for Python. R fitting/reporting needs compatible
installations of `optparse`, `jsonlite`, `digest`, `caret`, `glmnet`, `ggplot2`
and `Matrix`; growth fitting additionally needs `lme4`. Use `RSCRIPT_BIN` to
select an activated environment's Rscript. No launcher installs packages.

Provide separately obtained rating files through `RATINGS_DIR`. The ratings
included as test data are invented software inputs, not human judgments.
`regression_inputs.py` converts completed Parquet features to a frozen verified
CSV snapshot. Input identities, bounded-beta policy, columns and rating payloads
are checked; fitting uses that snapshot rather than mutable feature output.

```bash
python compositionality_over_time/regression_inputs.py snapshot \
  --features-root "$FEATURE_OUTPUT" --snapshot-root "$INPUT_ROOT"
bash compositionality_over_time/run_regressions.sh plan
```

Override `INPUT_ROOT`, `OUT`, `VOCABS`, `SETTINGS`,
`WEIGHTINGS`, `TAGGINGS`, `TIMESPANS` and `CUTOFFS` for the intended grid.
Prepare the snapshot before fitting and keep outputs separate from inputs.

For COHA sentence features, select the corpus and mode explicitly when freezing
the snapshot and dispatching the regression grid:

```bash
python compositionality_over_time/regression_inputs.py snapshot \
  --corpus coha --context-mode sentence \
  --features-root "$FEATURE_OUTPUT" --snapshot-root "$INPUT_ROOT"

export CORPUS=coha MODES=sentence
export INPUT_ROOT OUT
bash compositionality_over_time/run_regressions.sh plan
bash compositionality_over_time/run_regressions.sh check
bash compositionality_over_time/run_regressions.sh fit
```

Set `INPUT_ROOT` and `OUT` to separate COHA snapshot and result directories before
running these commands. Use `local5` instead of `sentence` for COHA local-window
features. Each snapshot contains one corpus/context mode; use separate roots
when comparing corpora or modes. Google defaults remain `CORPUS=google` and
`MODES=local5`. The model fitting and evaluation definitions below apply to both
corpora. Cumulative longitudinal support has the scope described in
[LONGITUDINAL.md](LONGITUDINAL.md).

## Models and interpretation

Elastic-net regularization accommodates correlated predictors. Model tuning
uses five-fold training cross-validation repeated ten times by default, with
Spearman correlation as the ranking objective. Missing values are median
imputed and predictors centered/scaled within training folds. Preprocessing
is refitted on all training items for the final model. The fixed held-out
180-training/100-test rating partition is not a CV-fold average.

Tagged `nn` evaluates NOUN–NOUN representations for every rated item. Tagged
`an` uses ADJ–NOUN for adjectival items and NOUN–NOUN for other items. UnTagged
uses pooled target-POS representations. These retain the same rating roster,
not an adjective-only evaluation subset. The modified lexical-form split
retains the original partition and scores; its explicit replacement table is
in the regression-data helper.

Recipes compare composition, cosine, association, frequency, productivity,
setting-comparison and temporal predictors separately and jointly. Choose one
frequency transformation and one productivity transformation per combined
recipe. Correlated coefficient magnitudes and permutation losses are not
independent causal effects.

`TestRsquared` columns contain squared Pearson correlations. The statistics
adapter additionally exports residual-based `1 - SSE/SST` scores from the
Spearman-tuned held-out predictions. Report these definitions explicitly.

## Dispatch and checkpoint reuse

`run_regressions.sh` runs individual stages locally:

```bash
bash compositionality_over_time/run_regressions.sh check
bash compositionality_over_time/run_regressions.sh fit
bash compositionality_over_time/run_regressions.sh report
```

Use `plan` to inspect the commands first, and configure the same intended grid
for checking, fitting and reporting. Alternatively, fit a single configuration
directly with R:

```bash
Rscript compositionality_over_time/regression/modified_regressions/regression_exps.r \
  --input-root "$INPUT_ROOT" --output-dir "$OUT" \
  --context-mode local5 --context-size 5000 --setting Aware \
  --ppmi PPMI --tagging UnTagged --timespans 10000,10 --cutoffs 0 \
  --targets compound --feature-sets all
```

This direct R example uses Google defaults. For COHA, add `--corpus coha` and
select `--context-mode sentence` or `--context-mode local5` to match the snapshot.
Result filenames and completion receipts use the selected corpus prefix.

For fine-grained reuse, `regression_batches.py` can create an immutable plan and
run an explicitly selected task index locally. Its `plan`, `verify` and `run`
commands provide `--help`. Plans record code, inputs, tuning, target/recipe
order and output identities. Compatible completed checkpoints are reused;
incomplete fits restart. NFS locks use readable descriptors. Do not run
overlapping processes writing the same checkpoint directory or bypass
source-identity checks.

Each width, including 10000 (aggregated), has a distinct identity. Temporal
widths and cutoffs are not silently pooled during reporting. Set local process
and thread counts within the memory and CPU budget available to your machine.

## General reports

`run_regressions.sh report` invokes the general workflow reporter under
`regression/modified_regressions/`. It validates requested result coverage,
summarises held-out predictions, compares matched configurations and reports
coefficient/permutation importance. No manuscript-specific report generator
or figure suite is included. Missing configurations must not be silently
claimed as a completed grid.

## Tests

```bash
python -m unittest discover -s compositionality_over_time/tests -v
```

Run retained `regression/tests/test_*.R` scripts with Rscript from a temporary
working directory. Tests use synthetic rosters/features and mock external
commands rather than running real experiments.
