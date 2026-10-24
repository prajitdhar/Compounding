# Google annotation enrichment

`google_v3_annotations.py` enriches the existing candidate
`bigram_annotations.parquet` without rerunning extraction, rebuilding the
context vocabulary or replacing the four feature-input count/context tables.
ADJ–NOUN and NOUN–NOUN candidates remain the annotation population; enrichment
is not limited to the Reddy/Cordeiro evaluation lists.

All commands below run from repository root. Set `OUTPUT_ROOT` to the completed
preprocessing dataset, with separate work and temporary-storage locations.
Enriched annotations are published atomically in that data root. Independent
copies of a dataset need their own annotation and manifest refresh.

## Interpretation

Google dependency exports provide directed, unlabeled parse evidence. They do
not provide `amod` or `compound` labels, establish adjacency, or prove that a
candidate is a lexical compound. Dependency counts and support indicators are
additional evidence, not replacement compound labels.

The two counts preserve `head=>dependent` and `dependent<=head` source notation
separately; the arrows are not claims about left/right word order. They are not
summed into a presumed unique dependency frequency or divided by adjacent
bigram frequency to manufacture a compound probability.

NER runs on the original cased, pre-tokenized canonical five-gram windows,
not on lemmas or lowercased pairs. Google POS and existing lemma/POS identifiers
are retained; the NER model does not retag or relemmatize them. The selected
`en_core_web_trf` model runs only its shared transformer and NER components;
parser, tagger and lemmatizer are not needed for this stage.

Predictions are deduplicated by cased token sequence and target indices across
decades before inference, then joined to the original snippet counts to obtain
occurrence-weighted surface/POS/decade summaries. A dominant label's confidence
is the fraction of classified occurrence mass supporting that label, **not a
calibrated model probability**. `ner_coverage` is classified canonical-window
mass divided by published bigram mass. Five-grams are frequency-thresholded and
fragmentary, so coverage is incomplete and this ratio is not guaranteed to
behave like a probability.

Not every published candidate has an available qualifying five-gram. Missing
evidence remains unknown/null, and low-coverage labels should be treated as
uncertain; `NONE` means a classified window contained no relevant recognized
entity. Do not replace nulls with `NONE`.

An annotation row remains one eligible **surface/POS/decade** pair, with its
original lemma/POS identifiers and adjacency `count`. Different surface forms
can share the same lemma/POS identifiers; pool annotation evidence before
joining this table to lemma-pooled feature counts.

Additional columns include:

| Column | Meaning |
| --- | --- |
| `same_entity_span_fraction` | Classified mass where both targets share an entity |
| `either_token_entity_fraction` | Classified mass with either target in any entity |
| `modifier_entity_fraction`, `head_entity_fraction` | Per-target entity membership |
| `ner_distribution_json` | Whole-pair labels and normalized weighted shares, including `NONE` |
| `available_ner_count`, `classified_ner_count` | Original available/classified snippet match counts |
| `snippet_classification_coverage` | Classified / available snippet mass |
| `ner_coverage`, `ner_coverage_exceeds_one` | Classified snippet / adjacent-pair mass and diagnostic flag |
| `ner_status` | `not_run`, `unavailable`, `below_threshold`, `partial`, or `assessed` |
| `dependency_head_to_dependent_count`, `dependency_dependent_to_head_count` | Separate source-arrow counts |
| `dependency_evidence_observed` | At least one published dependency observation |
| `compound_candidate_pos` | `ADJ-NOUN` or `NOUN-NOUN`, not a compound verdict |

`assessed` means all **available** snippets were classified, not that Google
supplies complete occurrence context. `partial` means some available snippet
mass failed the configured frequency cutoff. `unavailable` means no canonical
snippet evidence; `below_threshold` means evidence exists but none qualifies.
Dependency-only mode initially sets `not_run` while preserving existing NER
fields. It refuses to overwrite a later full-NER export with an unassessed
baseline.

## Stages and resources

The Python stages are `prepare`, `predict`, `aggregate` and `export`.
`gpu-check` optionally validates the inference environment before processing
data. Preparation performs resumable hash deduplication; each prediction shard
uses one persistent model process. Aggregation combines completed predictions,
and export refreshes annotations without replacing the scientific feature
tables. Do not aggregate until every prediction shard has completed, or run
export concurrently with another annotation writer.

Use `--threads`, `--memory-limit` and `--temp-directory` to set runtime budgets.
The SQL memory limit is not a total process-RAM cap: leave room for Arrow
buffers, Python, the language model and numerical libraries. Annotation
partitions run sequentially; keep `--aggregation-workers 1`. Multiple concurrent
prediction processes each need their own model memory and storage bandwidth.
Measure a representative shard before increasing concurrency or batch sizes.

Persistent partitions, unique snippets, predictions and enriched outputs can
require substantial storage in addition to the original dataset. Bounded RAM
use does not imply bounded total work storage. The original annotation table
is pinned by a hardlink for recoverability, or copied if filesystems differ.
Feature-table hashes are reused after path/size/stat checks, not by rereading
all four scientific tables on each export.

## Dependency-only enrichment

No NER preparation or prediction is needed for dependency evidence:

```bash
export OUTPUT_ROOT="$PWD/data/google"
export ANNOTATION_WORK="$PWD/work/annotations"
export ANNOTATION_SPILL="$PWD/work/annotation_spill"
export ANNOTATION_PROGRAM=google_preprocessing/google_v3_annotations.py

python "$ANNOTATION_PROGRAM" export \
  --output "$OUTPUT_ROOT" \
  --work-directory "$ANNOTATION_WORK" \
  --temp-directory "$ANNOTATION_SPILL" \
  --threads 4 --memory-limit 16GB --dependencies-only
```

This still reads large tables and is not necessarily an instant operation.
Missing NER evidence remains missing, and canonical feature count/context
tables are not regenerated.

## Full CPU NER

Install the `preprocessing` and `ner` extras and the chosen spaCy model in an
activated environment. The installed spaCy release must provide
`memory_zone()`; the program checks this requirement. Keep the same data root,
work directory, model, device, minimum snippet count and shard count throughout.

`--min-snippet-count 20` retains unique snippets whose summed occurrence count
over the selected corpus period is at least 20. Lowering the threshold increases
coverage and computation; it does not create missing five-grams. Model or cutoff
changes invalidate affected prediction checkpoints.

Using the variables from the dependency-only example:

```bash
NER_SHARDS=128
NER_OPTIONS=(
  --output "$OUTPUT_ROOT"
  --work-directory "$ANNOTATION_WORK"
  --temp-directory "$ANNOTATION_SPILL"
  --model en_core_web_trf --device cpu
  --shards "$NER_SHARDS" --min-snippet-count 20
  --threads 4 --memory-limit 16GB
  --ner-batch-size 32 --prediction-batch-rows 10000
)

python "$ANNOTATION_PROGRAM" prepare "${NER_OPTIONS[@]}"
for (( shard=0; shard<NER_SHARDS; shard++ )); do
  python "$ANNOTATION_PROGRAM" predict "${NER_OPTIONS[@]}" --shard-index "$shard"
done
python "$ANNOTATION_PROGRAM" aggregate "${NER_OPTIONS[@]}"
python "$ANNOTATION_PROGRAM" export "${NER_OPTIONS[@]}"
```

The loop is a serial reference workflow. Independent `predict` shard indices
may run in separate processes with the same settings, subject to memory and
storage limits. Never run two writers for the same shard concurrently. If a
shard fails, repeat only its `predict` command before aggregating:

```bash
python "$ANNOTATION_PROGRAM" predict "${NER_OPTIONS[@]}" --shard-index 17
```

Matching completed work is reused. Retain intermediates until the enriched
output and its source information have been verified. `--max-temp-directory-size` is a
per-connection spill cap, not a bound on persistent work files. Model batching
and checkpoint sizes are runtime controls, not calibrated confidence settings.
Do not edit source code during a checkpointed run.

## CUDA inference

CUDA is opt-in through `--device cuda`, with no silent CPU fallback. Preparation,
count-weighted aggregation and export remain CPU operations; passing the same
CUDA selection to them preserves prediction identity without initializing a
GPU. Only inference and the CUDA preflight need a visible, compatible GPU.

Use a separate compatible GPU environment. A pinned example dependency set is
provided in `google_preprocessing/requirements-ner-gpu.txt`; install the matching
PyTorch build and selected spaCy transformer model explicitly, then run
`python -m pip check`. That file is an example stack, not a claim about the
latest packages or a guarantee for every GPU/driver combination. The scripts
do not install packages or configure hardware automatically.

When using `CUDA_VISIBLE_DEVICES`, `--gpu-id 0` selects the first GPU visible to
the process, not necessarily device zero on the whole machine. Validate actual
kernels and a tiny model inference before a full run:

```bash
python "$ANNOTATION_PROGRAM" gpu-check \
  --output "$OUTPUT_ROOT" --model en_core_web_trf \
  --device cuda --gpu-id 0 --smoke-test
```

The preflight reports library/runtime versions and exercises Torch/CuPy kernels
and real model inference. Device visibility alone is not a successful check.
CPU or CUDA package/model changes can invalidate existing predictions; keep
the environment unchanged throughout a run.

For CUDA processing, replace the `--device cpu` entry in `NER_OPTIONS` with
`--device cuda --gpu-id 0`, then run the same four-stage workflow. Keep the CUDA
selection for retries, aggregation and export. Reduce `--ner-batch-size` after
a device-memory error rather than changing backend for an incomplete prediction
plan. GPU acceleration does not eliminate the large input population or I/O
costs, so no completion time is guaranteed.
