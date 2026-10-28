# Compositionality over time

Analyse compound compositionality using independently extracted occurrence
counts and context tables. This project contains feature generation, regression
experiments, general result summaries and cumulative longitudinal analysis.
Corpus downloading, extraction, lemmatization and NER are not part of this
project.

## Required preprocessing

First complete [Google preprocessing](../google_preprocessing/README.md) or
[COHA preprocessing](../coha_preprocessing/README.md).
Either output supplies four Parquet tables, including
`bigram_counts.parquet`, and a fixed context vocabulary. Both this project and
novel-compound prediction can consume those outputs without rerunning
preprocessing for every analysis configuration.

Select `extracted/feature_inputs/local5/` for Google or COHA local-window
contexts, or `extracted/feature_inputs/sentence/` for COHA sentence contexts.
The manifest identifies the corpus and context mode. Keep separate caches and
output directories for each corpus, context mode and target-tagging choice.

From repository root, install the analysis dependencies:

```bash
python -m pip install -e '.[compositionality]'
```

Add the `coha` extra when preparing COHA text: `.[coha,compositionality]`.
The extras cover Python dependencies, not corpora,
language models or R packages. Actual human ratings must be obtained separately.
Test results are toy examples, not actual human judgments.

## Analysis stages

1. [Features](FEATURES.md): prepare bounded raw-count caches, then calculate
   RAW/PPMI, constituent-setting, tagging, temporal-window and cutoff variants.
2. [Regression experiments](REGRESSIONS.md): freeze verified feature inputs,
   fit training-CV-tuned models and summarise held-out results.
3. [Longitudinal analysis](LONGITUDINAL.md): freeze a modern model and evaluate
   cumulative historical aggregations, then fit descriptive growth models.

The ordinary feature and regression workflow supports COHA and Google inputs.
Cumulative longitudinal programs currently support Google `local5` only.

Maintained command-line programs live directly in this directory. R
implementations are under `regression/`. The small
`feature_helpers/` modules provide the tested feature and ratings definitions.
Generic infrastructure is imported from the `src` package.

## Tests and source identities

From repository root:

```bash
python -m unittest discover -s compositionality_over_time/tests -v
```

R regression and longitudinal tests are in `regression/tests/`; run each with
`Rscript` from a temporary working directory. New source locations, entrypoint
names and shared implementations have new code identities. Use fresh output
directories; do not rewrite old completion receipts to force checkpoint reuse.
