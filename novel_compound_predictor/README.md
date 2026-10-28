# Predicting novel noun--noun compounds

Build compound/constituent embeddings from historical count tables, train a
compound-plausibility classifier, and rank candidate modifier--head pairs.
Both pooled-across-time and temporal LSTM models are available. This directory
contains scripts and synthetic examples; obtain corpus tables and fitted
models separately.

## Setup

From the repository root, in a dedicated Python 3.10+ environment:

```bash
python -m pip install -e '.[novel]'
```

Install the CPU or CUDA-enabled PyTorch build appropriate for your environment
before training. The scripts accept `--device cpu`, `--device cuda`, or
`--device auto`; the synthetic tests run on CPU.

## Quick example

```bash
bash novel_compound_predictor/examples/run_example.sh
```

The example builds two-dimensional representations from tiny invented counts
in a temporary directory, trains a tiny pooled classifier and ranks candidates.
It checks both pooled and decade-specific output. It is a schema/workflow
demonstration, not a scientifically meaningful prediction model.

## Maintained workflow

1. Obtain canonical extracted count tables using the upstream
   [Google preprocessing project](../google_preprocessing/README.md).
2. Run `build_representations.py` with separate context-event and occurrence
   tables. Never substitute context sums for compound occurrence counts.
3. Train with `predict_novel_compounds.py train`; rank with its `rank` command.
4. Optionally export an additive visualization table using
   `export_visualization_dataset.py` or convert headered vectors using
   `csv_embeddings_to_parquet.py`.

See [WORKFLOW.md](WORKFLOW.md) for schemas, full commands, and scientific
limitations. Each command also provides `--help`.

## Configuration and execution

Supply the input, output and temporary-storage paths explicitly. For example,
from repository root:

```bash
export DATA_ROOT=/data/feature_inputs/local5
export OUTPUT_ROOT=/data/novel_runs
export SCRATCH_ROOT="$PWD/work"
python novel_compound_predictor/build_representations.py \
  --pair-contexts "$DATA_ROOT/bigrams.parquet" \
  --pair-counts "$DATA_ROOT/bigram_counts.parquet" \
  --output "$OUTPUT_ROOT/pooled_300" \
  --scratch "$SCRATCH_ROOT/representation_work" \
  --dimensions 300 --max-contexts 5000 --min-compound-count 200 \
  --threads 4 --memory-limit 8GB --numerical-memory-limit 16GB
```

Add `--temporal` and choose a separate output directory for temporal
representations. Paths and resource budgets in examples are placeholders;
adjust them for your input size and available memory. The SQL cap and numerical
planning budget must leave room for other process allocations. No environment
is activated automatically.

## Tests

From the repository root:

```bash
python -m unittest discover -s novel_compound_predictor/tests -v
```

Tests use generated data, including complete 300-dimensional pooled and
temporal build/train/rank checks. No corpus download or external job submission occurs.

## Project boundary

This project consumes the canonical count tables produced by
`google_preprocessing/`; it does not depend on the compositionality-analysis
project. The input/output specifications and modeling choices are documented here
so the workflow can be used independently.
