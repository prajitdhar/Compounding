# Preprocessing source information and reusable table specification

`src/source_info.py` creates deterministic records of source information.
`src/io.py` publishes metadata/tables atomically, and
`src/partitioned_parquet.py` provides bounded reduction. Preprocessing imports
these modules directly, and software identities include all authoritative
shared implementations alongside the stage's own code.

## Stage identities

Completed stage manifests describe:

- `stage` and `status`: planned work is not completed extraction.
- `corpus`: Google Books Ngrams, exact release `20200217` and chosen subset.
- `inputs`: actual input paths, byte sizes and SHA-256 hashes where known.
- `parameters`: scientific settings, including POS, dates, count definition,
  lemmatization, context geometry and vocabulary selection.
- `time`: numeric date coordinate, available precision and binning rule.
- `software`: Python/dependency versions and code-file SHA-256 identities.
- `model`: selected model and files when a language model is used.
- `outputs` and `statistics`: artifact inventory plus explicitly defined units.
- `execution`: workers, memory budgets, batching and other runtime settings.

Paths locate artifacts; they are not content identities. Runtime manifests
can contain private local paths and belong with external run outputs, not in
the published source. A manifest does not include its own hash. Unknown
model or source identities remain explicitly unknown.

An input or output artifact has the shape:

```json
{"name": "bigram_counts.parquet", "path": "/data/export/bigram_counts.parquet", "bytes": 123, "sha256": "HEX_DIGEST"}
```

The actual byte hashes are not a claim that different dependency versions
produce identical compressed Parquet bytes. Logical row equality and byte
identity are distinct checks.

## Counts, lexical keys and time

The sole propagated occurrence count is Google's `match_count`, renamed
`count`. Summed volume counts are not deduplicated document counts and are
not used. Context-event counts and occurrence counts remain separate tables.

Raw extraction retains the cased surface/POS representation needed for NER and
lemmatization. Canonical export joins those keys to a versioned rule-lemmatizer
lexicon and writes lowercased `lemma_POS` keys. Context keys also retain POS.
Missing lexical joins fail or are resolved through the explicitly recorded
coverage policy; they must not silently drop observations.

Raw `decade` is mapped to the canonical `time` field on export. A decade start
such as 1820 denotes the 1820s, not an exact publication year. Context distances
1, 2 and 3 are obtained from exposed 3-, 4- and 5-gram edges. Five-grams do not
establish sentence boundaries or complete occurrence contexts.

## Canonical downstream inputs

The shared schema is version 3 with representation version 5 and four tables:

| Table | Columns |
|---|---|
| `unigrams` | `word, time, context, count` |
| `unigram_counts` | `word, time, count` |
| `bigrams` | `modifier, head, time, context, count` |
| `bigram_counts` | `modifier, head, time, count` |

The corpus-neutral `corpus_manifest.json` declares
`word_target_pos=bigram_target_pos="all_lexical"`, relative `table_files`, and
source information for the completed `feature_inputs` stage. Downstream projects
validate this specification and then independently choose their target
populations, weighting, frequency cutoffs and model features. Preprocessing does
not require ratings, train/test splits, compositionality predictors or fitted
regressions.

The context vocabulary and coverage sidecars are hashed. Vocabulary ranking
uses mean within-decade relative frequency, with lexical tie breaks; the same
ranking is reusable across downstream configurations and time bins.

## Annotation evidence

The separate annotation table contains ADJ–NOUN/NOUN–NOUN surface/POS/decade
candidates. NER is computed on cased canonical windows, then joined to their
original counts. Confidence and coverage columns describe available snippet
evidence, not calibrated probabilities or proof of lexical compoundhood.
Dependency arrows preserve distinct Google source notations; they are not
silently summed into a presumed unique dependency frequency.

Annotation enrichment verifies that the four canonical analytical tables stay
unchanged. It refreshes annotation records and their source information
atomically and preserves unknown labels when context is missing. See
[ANNOTATIONS.md](ANNOTATIONS.md).

## Completion and reuse

Extraction skips a completed shard only when its settings and identity match.
Incompatible markers fail unless replacement is explicitly requested.
Reduction validates completed input markers before reading fragments; a partial
Parquet file alone is not evidence of completion. Bounded partition work has
its own compatibility identities and is reused only when those match.

Code changes create new source identities. Reuse a checkpoint only when its
recorded inputs, settings and source identities match the current run. Choose
a fresh output directory for incompatible work rather than editing completion
records or disabling validation.
