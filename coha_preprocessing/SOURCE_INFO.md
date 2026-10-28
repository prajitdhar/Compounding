# COHA table specification and source information

Every completed preparation, extraction and feature-input export records its
source information. Records identify input bytes, scientific settings, date
precision, implementation files, software versions, the selected model and
actual output hashes. The corpus release remains unknown when not supplied.
Runtime records can contain local paths and belong with external run outputs.

## Canonical specification

`feature_inputs/<mode>/corpus_manifest.json` uses schema version 3 and
representation version 5. It declares `format="parquet"`,
`word_target_pos=bigram_target_pos="all_lexical"`, raw counts, the context mode
and relative `table_files`. The shared validator in `src/source_info.py`
checks this specification before downstream use.

| Table | Key columns | Value |
|---|---|---|
| `unigrams` | `word, time, context` | Raw word–context event `count` |
| `unigram_counts` | `word, time` | Raw observed token `count` |
| `bigrams` | `modifier, head, time, context` | Raw pair–context event `count` |
| `bigram_counts` | `modifier, head, time` | Raw observed adjacent-pair `count` |

Target and context keys are lowercased `lemma_POS`, with the chosen proper-noun
policy applied. All selected domains are pooled. `time` is a numeric calendar
coordinate, with original year/decade precision recorded upstream. It is not
necessarily an exact publication year.

The context matrices pool side and distance. Extraction never estimates
occurrence counts by summing those matrices. Counts include observed lexical
tokens and adjacent pairs even when no retained context is available. Tagged
ADJ–NOUN/NOUN–NOUN selection, pooling target POS, RAW/PPMI weighting and
frequency cutoffs belong to feature generation.

## Annotation and coverage

`tables/bigram_annotations.parquet` retains the target lemmas, surfaces,
observed POS, direct-dependency evidence, modifier dependency label, modifier
and head NER labels, sentence completeness, time and occurrence count.
`tables/bigram_document_counts.parquet` retains document identities and pair
occurrence counts within each document. Deduplicate document identities for a
document-frequency calculation; summing the `count` field measures occurrences.

Mode directories contain unigram/bigram coverage tables. `local5` also records
each of six potential side/distance positions as a retained context, filtered
token, sentence boundary or missing-text gap. Sentence contexts exclude
sentences whose boundaries are uncertain under the gap policy. These audits
distinguish unavailable evidence from an observed zero vector.

## Vocabulary and completion

`vocabulary.py` reads `unigram_counts`, filters to the context POS inventory and
ranks by mean within-decade relative frequency. Missing word/decade cells
contribute zero; every observed decade has equal weight. Lexical order breaks
ties. Its TSV and `.source_info.json` receipt are separate from the extraction
specification, so one extraction can support multiple vocabulary sizes.

Preparation verifies source/member identities; extraction verifies prepared
fragment hashes. Stage manifests publish after their outputs succeed. A
partially written table or sentence file does not establish completion.
Extraction uses temporary count databases rather than resumable worker
checkpoints. Downstream feature caches have their own content and configuration
identities. Reuse only matching completed outputs; choose fresh directories
when inputs, mode, model, code or scientific settings change.

Output hashes identify actual bytes. They do not promise identical compressed
bytes across different dependency versions or hardware. Preserve the manifests
and vocabulary receipts with the data they describe.
