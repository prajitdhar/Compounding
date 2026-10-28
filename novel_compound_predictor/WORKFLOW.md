# Representation and prediction workflow

## Inputs and interpretation

The representation builder consumes already extracted tables. It does not
download, reparse, or modify a corpus. Supported inputs are canonical Parquet
tables and headered CSV/TSV equivalents:

| Table | Required columns |
|---|---|
| Pair contexts | `modifier`, `head`, `time`, `context`, `count` |
| Pair occurrences | `modifier`, `head`, `time`, `count` |
| Optional general-word contexts | `word`, `time`, `context`, `count` |

`decade` or `year` can replace `time` at the input boundary. Separate surface/POS
columns and the noun suffixes `_n` and `_noun` are accepted by the builder, but no
new lemmatization is performed. Input occurrence counts must be independent of
the number of contexts extracted for a pair. Google match counts are not
document counts or sentence counts.

CompoundAware is the default: modifier and head roles are aggregated separately
from compound contexts. Unigrams are not read unless `--word-contexts` is
provided. The latter selects CompoundAgnostic constituents. CompoundPooled
roles are not implemented in this builder. In this project, **pooled means
pooled across decades**, not pooled modifier/head roles.

## Build representations

Run from this project directory after installing the repository package:

```bash
python build_representations.py \
  --pair-contexts "$DATA_ROOT/bigrams.parquet" \
  --pair-counts "$DATA_ROOT/bigram_counts.parquet" \
  --output "$OUTPUT_ROOT/pooled_300" \
  --scratch "$SCRATCH_ROOT/representation_work" \
  --dimensions 300 --max-contexts 5000 --min-compound-count 200 \
  --threads 4 --memory-limit 8GB --numerical-memory-limit 16GB \
  --aggregation-partitions 256 --matrix-storage auto \
  --svd-solver streaming --log-interval 30
```

For temporal representations, add `--temporal` and use a different output
directory. Existing outputs are protected. The example budgets are configurable
starting values, not measured full-corpus peak requirements.

By default, the SVD basis uses 1800--1999 only, raw count weighting, 300
dimensions, seed 1991, ten iterations and L2 row normalization. Modifier, head
and optionally compound rows share one basis across retained decades.
`--constituents-only` omits compound rows and therefore produces a different
basis. Contexts are ranked within the historical window; future-only role keys
are excluded. A past-only builder cannot undo any future-informed upstream
vocabulary selection: strict forecasting also requires past-only extraction
and vocabulary decisions upstream.

`--min-compound-count` is a pooled historical occurrence-count threshold.
First-attestation decisions use all supplied occurrence dates, before this
threshold or the embedding vocabulary is applied. The default positive period
is 2000--2009 with at least three occurrences. This is novelty relative to the
supplied corpus counts, not proof of a linguistic coinage.

## Memory and progress

Aggregation uses bounded Arrow batches and hash-partitioned Parquet buckets;
oversized buckets are recursively split. Large CSR arrays can be scratch-backed.
Streaming SVD applies row-bounded sparse Gram products with float64 accumulation.
The optional sklearn solver uses a different numerical implementation and may
need substantially more RAM. Their coordinates must not be mixed.

The SQL memory limit covers DuckDB, not all NumPy/SciPy/SVD allocations. Set the
numerical planning budget and job RAM separately and allow room for both.
Scratch must hold intermediate partitions, mapped arrays and spill. Temporary
aggregation work is not a cross-run checkpoint: a failed build restarts into
an empty/new output directory. Flushed stage messages and regular heartbeats
report observed progress and memory use without fabricated ETAs. Shared logging
uses UTC by default.

## Outputs and source information

Role tables contain `modifier` or `head`, optional `decade`, and
`dim_0` through `dim_299`. Compound vectors additionally contain the pair keys.
Keep role tables, `svd_basis.npz`, context ordering and
`representation_manifest.json` together. The manifest records the fitted time
window, source metadata, solver, parameters, dependencies, output names and
hashes of the local/shared implementation. It is generated run metadata and
may contain local paths; do not publish it without reviewing those paths.

Other outputs include `novel_compounds.csv`, `attested_compounds.csv`,
`historical_compounds.csv`, `training_compounds.csv` and first-attestation
metadata. Training exclusion uses all attested pairs, including future
positives. Ranking exclusion normally uses historical pairs only, allowing
genuinely future pairs to remain candidates.

## Train and rank

```bash
python predict_novel_compounds.py train \
  --modifiers "$OUTPUT_ROOT/pooled_300/dense_embeddings_modifiers.parquet" \
  --heads "$OUTPUT_ROOT/pooled_300/dense_embeddings_heads.parquet" \
  --positives "$OUTPUT_ROOT/pooled_300/novel_compounds.csv" \
  --attested "$OUTPUT_ROOT/pooled_300/attested_compounds.csv" \
  --output-dir "$OUTPUT_ROOT/model_pooled" --encoder pooled --device cpu

python predict_novel_compounds.py rank \
  --checkpoint "$OUTPUT_ROOT/model_pooled/model.pt" \
  --modifiers "$OUTPUT_ROOT/pooled_300/dense_embeddings_modifiers.parquet" \
  --heads "$OUTPUT_ROOT/pooled_300/dense_embeddings_heads.parquet" \
  --attested "$OUTPUT_ROOT/pooled_300/historical_compounds.csv" \
  --candidates candidates.csv --output "$OUTPUT_ROOT/ranking.csv" --device cpu
```

The pooled classifier concatenates modifier/head vectors and applies
`Linear(2D,H) -> ReLU -> Linear(H,2,bias=False)`. For temporal input, use
`--encoder lstm`: separate one-layer modifier/head LSTMs encode ordered decade
sequences. Their hidden size, zero-filled missing decades and joint supervised
training are explicit implementation choices. Pooled prediction averages a
supplied decade table; this differs from fitting SVD after pooling raw counts.

Positive pairs are split before corruption. Negative pairs are unique across
splits and exclude every pair supplied in the training attestation table. Model
selection uses validation loss; the test split is evaluated afterward.
Scores are uncalibrated `logit(plausible) - logit(implausible)` margins, not
probabilities. The experiment is supervised held-out-pair disambiguation, not
an unsupervised forecasting claim.

Model checkpoints use format version 2 and record implementation hashes.
Loading verifies the format, source identity and matching input vectors.
Resume with unchanged code and inputs; use a separate output directory when
changing the configuration or implementation.

## Optional visualization export

`export_visualization_dataset.py` writes a tab-separated table with `common`,
`novel` and vector-coordinate columns. Observed representations take precedence
over additive modifier-plus-head representations during deduplication. Its
`novel` flag identifies the exported representation type; it is not itself a
first-attestation label. No second dimensionality reduction is applied.

The number of exported coordinates is controlled explicitly. A truncated
export cannot recover omitted coordinates. Keep the fitted basis with the role
vectors; independently refitted bases are not interchangeable.
