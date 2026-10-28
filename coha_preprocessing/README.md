# COHA preprocessing

This standalone upstream project converts COHA linear text into reusable
lemma/POS occurrence and context tables. Run it **before** compositionality
feature generation. The same exported tables can also be used directly by
later projects without running compositionality regressions.

This directory contains only text preparation, annotation, occurrence/context
extraction and context-vocabulary selection. It does not calculate
compositionality predictors or fit regression models.

## Installation and offline example

From repository root in an activated Python 3.10+ environment:

```bash
python -m pip install -e '.[coha]'
python coha_preprocessing/coha_pipeline.py prepare \
  --input coha_preprocessing/examples/raw \
  --output runs/coha-example/prepared --corpus-release synthetic
```

This preparation example uses invented local text and does not download
anything or need a language model. The [complete example](examples/README.md)
also runs extraction and vocabulary selection. Extraction requires a separately
installed English spaCy pipeline with its tagger, lemmatizer, parser and NER:

```bash
python -m spacy download en_core_web_lg
```

`--model` accepts another installed compatible pipeline or a local model
directory. Use the same model and settings when comparing runs. For
preprocessing and feature generation together, install
`.[coha,compositionality]`.

## Data specification

Obtain COHA text separately with the required permissions and record the exact
supplied release through `--corpus-release`. Omitting it records an unknown
release. Inputs can be a text file, directory of text files, ZIP, or TAR
containing text/ZIP members. Filenames identify domain and date, for example
`text_fic_1820.txt` for a decade file or `fic_1821_123.txt` for a dated document.
Embedded numeric document markers split documents.

An optional `--metadata metadata.tsv` supplies `doc_id`, `time`, `domain` and
optionally `time_precision`; add `source` when document IDs are reused across
source members. Preparation records the available date precision rather than
inferring an exact year from a decade. Unicode, quotes and structural markup
are normalized while observed wording is preserved. Missing-text markers split
fragments before annotation and texts are never joined across a gap.

After extraction, downstream programs use the same four datasets in each
`extracted/feature_inputs/sentence/` and `extracted/feature_inputs/local5/`
directory:

| File | Columns | Unit |
|---|---|---|
| `unigrams.parquet` | `word, time, context, count` | Word–context events |
| `unigram_counts.parquet` | `word, time, count` | Observed word occurrences |
| `bigrams.parquet` | `modifier, head, time, context, count` | Pair–context events |
| `bigram_counts.parquet` | `modifier, head, time, count` | Observed adjacent-pair occurrences |

The `corpus_manifest.json` in each mode records the source corpus, table paths,
count definitions and completed-export identities. Lexical identifiers are
lowercased `lemma_POS`; all lexical target POS and observed lexical adjacent
pairs are retained. No benchmark-vocabulary filtering is applied. With
`--proper-nouns map`, PROPN maps to NOUN in these keys; `exclude` keeps PROPN
distinct so later NOUN-target selection excludes those forms.

Occurrence tables have the same values across modes. The fixed vocabulary ranks context types by mean within-decade relative frequency, giving each observed decade equal weight with lexical tie breaks.

## First run: base extraction and vocabulary

All commands below run from repository root. Set `RAW_INPUT` to the COHA text/archive and `CORPUS_RELEASE` to its release label:

```bash
export PREPROCESSING=coha_preprocessing
export OUTPUT_ROOT="$PWD/data/coha"
export PYTHON_BIN=python3
python "$PREPROCESSING/coha_pipeline.py" prepare \
  --input "$RAW_INPUT" --output "$OUTPUT_ROOT/prepared" \
  --corpus-release "$CORPUS_RELEASE"
```

Preparation writes `fragments.jsonl.gz` and `prepared_manifest.json`.
`--domains` can restrict the selected domains. Input and output locations must
be separate, and scratch storage must stay outside the input tree.

COHA extracts occurrence counts and both context modes in one parsing pass.
The next section runs that pass and then builds the vocabulary from its occurrence counts. The vocabulary is then selected after extraction.

## Context extraction and reusable export

Parse the prepared fragments and export both context modes:

```bash
python "$PREPROCESSING/coha_pipeline.py" extract \
  --input "$OUTPUT_ROOT/prepared" --output "$OUTPUT_ROOT/extracted" \
  --model en_core_web_lg --workers 1 --batch-size 256 \
  --scratch-dir "$OUTPUT_ROOT/scratch" --proper-nouns map
```

| Mode | Context |
|---|---|
| `local5` | Up to three token positions on either side of a target word or adjacent pair, within the observed sentence span. |
| `sentence` | Eligible context tokens elsewhere in a sentence. |

Context POS are NOUN, PROPN, ADJ, VERB, NUM and ADV.
Once extraction completes, export a fixed ranked vocabulary:

```bash
python "$PREPROCESSING/vocabulary.py" \
  --input "$OUTPUT_ROOT/extracted/feature_inputs/sentence" \
  --output "$OUTPUT_ROOT/context_vocab_equal_decade_50000.tsv" --size 50000 \
  --memory-limit 1GB --threads 1 --temp-directory "$OUTPUT_ROOT/scratch"
```

`--size 0` retains all ranked contexts. The TSV has a `.source_info.json`
receipt recording the selection and contributing decades. Ranking is the same
for the two modes. 

For the feature generation, pass the selected mode directory and
vocabulary TSV to [compositionality features](../compositionality_over_time/FEATURES.md),
then follow the [regression guide](../compositionality_over_time/REGRESSIONS.md).


## Optional annotations

Extraction records surface forms, observed POS, direct-dependency evidence,
modifier dependency labels, modifier/head NER and sentence completeness in
`extracted/tables/bigram_annotations.parquet`. These annotations come from the
same parsing pass; their use in downstream analyses is optional.

Document-level pair evidence is stored in `bigram_document_counts.parquet`.
Mode directories contain coverage diagnostics. 

## Runtime configuration


Choose explicit input/output paths, worker counts and temporary storage for
your workload. Counts accumulate in disk-backed SQLite tables and stream to
Parquet. Extra `--workers` create independent model/counting processes and
`--shard-size` controls fragments per task. Choose a worker count that fits
memory, and provide enough scratch for databases/sorting and separate space
for final outputs.


## Tests and source information

```bash
python -m unittest discover -s coha_preprocessing/tests -v
```

Tests use synthetic inputs and controlled spaCy documents/models. They do not
download a model, import downstream analysis or read real corpus/rating data.
See [SOURCE_INFO.md](SOURCE_INFO.md) for the schema.
