# Learning to Model Compounding

Three projects share a small, installable Python library:

- [Google preprocessing](google_preprocessing/README.md): extract, reduce,
  lemmatize and annotate Google Books Ngrams V3 data, then export reusable
  context and occurrence tables.
- [Compositionality over time](compositionality_over_time/README.md): calculate
  compositionality features from those tables, run regression experiments,
  summarize results and estimate longitudinal trajectories.
- [Novel-compound prediction](novel_compound_predictor/README.md): build
  representations from the same tables and train temporal predictors for predicting novel compounds.

Shared modules live directly in [`src/`](src), with imports such as
`from src.source_info import sha256_file`.

## Workflow

```text
Google Books Ngrams V3
        |
google_preprocessing/
        |
canonical Parquet tables
        |---------------------------|
compositionality_over_time/    novel_compound_predictor/
```

Run preprocessing first, unless you already have a compatible completed export.
The extracted tables can be reused by the other projects. Downstream projects receive data paths; they do not trigger downloads or preprocessing implicitly.

## Installation

Use Python 3.10 or newer and an isolated environment. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[compositionality]'
```

Use `.[preprocessing]` for the Google preprocessing project alone,
`.[novel]` for representation building and neural prediction of novel compounds, or `.[all]` for all three projects plus optional spaCy support. The `compositionality`
extra includes the Python dependencies needed for preprocessing and analysis.
Select an appropriate PyTorch CPU/GPU build for your machine before installing the novel extra if needed. The extraction/feature workflow does not require a GPU.
See each project's README for R and optional NER requirements.

## Examples and input data

Small synthetic examples and tests are included. Full corpus tables, embeddings,
rating datasets, models and experiment results are not distributed here.
Obtain external datasets with the necessary permissions and supply their paths
explicitly.

- [Google preprocessing example](google_preprocessing/examples/README.md)
- [Novel-compound example](novel_compound_predictor/examples/README.md)

The preprocessing handoff is described by `corpus_manifest.json` and contains
four independent tables under `extracted/feature_inputs/local5/`:

| File | Columns |
|---|---|
| `unigrams.parquet` | `word`, `time`, `context`, `count` |
| `unigram_counts.parquet` | `word`, `time`, `count` |
| `bigrams.parquet` | `modifier`, `head`, `time`, `context`, `count` |
| `bigram_counts.parquet` | `modifier`, `head`, `time`, `count` |

Occurrence counts and context-event counts are different quantities and must
not be substituted for one another. The export includes schema definitions and
source information, and can additionally include bigram annotations under
`extracted/tables/`. See the [preprocessing README](google_preprocessing/README.md) for further information.

## Tests

After installing the relevant extras, run from the repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s google_preprocessing/tests -v
python -m unittest discover -s compositionality_over_time/tests -v
python -m unittest discover -s novel_compound_predictor/tests -v
```

See [TESTING.md](TESTING.md) for test dependencies, R checks and source-tree
validation. Tests use invented data and do not reproduce corpus-scale results.

## Reproducibility

Each completed run records its inputs, parameters, software and output
identities. Keep manifests with the data and fitted models they describe.
Resume only when the saved configuration and source identities match; use a
new output directory when changing the configuration.

## License

Code is distributed under the
[Apache License 2.0](LICENSE); external datasets and models have their own
terms.

## Use of AI

The documentation files and test suits have been enhanced using GPT-6.1 Sol.