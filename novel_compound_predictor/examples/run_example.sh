#!/usr/bin/env bash
# Run only tiny invented data; never downloads a corpus or submits a batch job.
set -euo pipefail
EXAMPLE_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIRECTORY=$(cd -- "$EXAMPLE_DIRECTORY/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
if [[ -n "${EXAMPLE_OUTPUT:-}" ]]; then
    OUTPUT_DIRECTORY=$EXAMPLE_OUTPUT
    mkdir -p "$OUTPUT_DIRECTORY"
else
    OUTPUT_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/compound-example.XXXXXX")
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for mode in pooled temporal; do
    temporal_arguments=()
    [[ "$mode" != temporal ]] || temporal_arguments=(--temporal)
    "$PYTHON_BIN" "$PROJECT_DIRECTORY/build_representations.py" \
        --pair-contexts "$EXAMPLE_DIRECTORY/contexts.csv" \
        --pair-counts "$EXAMPLE_DIRECTORY/counts.csv" \
        --output "$OUTPUT_DIRECTORY/$mode" \
        --year-start 1980 --dimensions 2 --threads 1 --max-contexts 4 \
        --aggregation-partitions 2 --aggregation-batch-size 8 \
        --batch-size 4 --memory-limit 256MB --numerical-memory-limit 256MB \
        --scratch "$OUTPUT_DIRECTORY/work" "${temporal_arguments[@]}"
done
"$PYTHON_BIN" "$PROJECT_DIRECTORY/predict_novel_compounds.py" train \
    --modifiers "$OUTPUT_DIRECTORY/pooled/dense_embeddings_modifiers.parquet" \
    --heads "$OUTPUT_DIRECTORY/pooled/dense_embeddings_heads.parquet" \
    --positives "$OUTPUT_DIRECTORY/pooled/novel_compounds.csv" \
    --attested "$OUTPUT_DIRECTORY/pooled/attested_compounds.csv" \
    --output-dir "$OUTPUT_DIRECTORY/model" --encoder pooled --dimensions 2 \
    --hidden-size 4 --epochs 2 --batch-size 2 --start-year 1980 --device cpu
"$PYTHON_BIN" "$PROJECT_DIRECTORY/predict_novel_compounds.py" rank \
    --checkpoint "$OUTPUT_DIRECTORY/model/model.pt" \
    --modifiers "$OUTPUT_DIRECTORY/pooled/dense_embeddings_modifiers.parquet" \
    --heads "$OUTPUT_DIRECTORY/pooled/dense_embeddings_heads.parquet" \
    --attested "$OUTPUT_DIRECTORY/pooled/historical_compounds.csv" \
    --candidates "$EXAMPLE_DIRECTORY/candidates.csv" \
    --output "$OUTPUT_DIRECTORY/ranking.csv" --top-k 3 --batch-size 2 --device cpu
printf 'Synthetic example outputs: %s\n' "$OUTPUT_DIRECTORY"
