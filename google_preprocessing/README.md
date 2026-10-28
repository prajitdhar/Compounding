# Google Books Ngrams preprocessing

This standalone upstream project converts Google Books Ngrams V3 into reusable
lemma/POS occurrence and context tables. Run it **before** compositionality
feature generation. The same exported tables can also be used directly by
novel-compound prediction or later projects.

## Installation and offline example

From repository root in an activated Python 3.10+ environment:

```bash
python -m pip install -e '.[preprocessing]'
python google_preprocessing/examples/run_tiny_extraction.py
```

The example uses invented local n-gram observations and does not download
anything or need a language model. For lemmatization
or NER, install the optional `ner` extra and a chosen spaCy model separately.
The extraction/reduction stages do not require a GPU or PyTorch.

## Data specification

The program uses the English V3 release `20200217` and Google's own POS and
dependency annotations. The occurrence `count` is `match_count` and the volume counts
are not stored.

After reduction, lemmatization and export, the following files are created:

```text
OUTPUT_ROOT/
  extracted/
    feature_inputs/local5/
      unigrams.parquet        # word, time, context, count
      unigram_counts.parquet  # word, time, count
      bigrams.parquet         # modifier, head, time, context, count
      bigram_counts.parquet   # modifier, head, time, count
      corpus_manifest.json
    tables/
      bigram_annotations.parquet
```

The `corpus_manifest.json` file records the source corpus, table paths, count
definitions and completed-export identities. Bigram occurrence counts are
stored only in `feature_inputs/local5/bigram_counts.parquet`; `tables/` contains
annotations and optional dependency summaries.
Lexical identifiers are lowercased `lemma_POS`. All lexical
POS combinations are retained in the core tables. 
Context evidence comes from exposed edges of 3/4/5-grams at distances 1/2/3.
Google data do not provide recoverable sentences or occurrence identifiers.

The fixed vocabulary ranks lemma/POS context types by mean within-decade
relative frequency, giving decades equal weight. 

## First run: base extraction and vocabulary

All commands below run from repository root.

```bash
export PREPROCESSING=google_preprocessing
export OUTPUT_ROOT="$PWD/data/google"
export PYTHON_BIN=python3
python "$PREPROCESSING/google_v3_pipeline.py" manifest --orders 0,1 \
  --output "$OUTPUT_ROOT/base_manifest.json"
python "$PREPROCESSING/google_v3_pipeline.py" manifest --orders 2,3,4,5 \
  --output "$OUTPUT_ROOT/context_manifest.json"
```

For each zero-based entry of `base_manifest.json`, run:

```bash
python "$PREPROCESSING/google_v3_pipeline.py" extract \
  --manifest "$OUTPUT_ROOT/base_manifest.json" --task-id 0 \
  --output "$OUTPUT_ROOT"
```

The example extracts only task 0; repeat for all entries. Once the complete base
extraction is available:

```bash
python "$PREPROCESSING/google_v3_pipeline.py" reduce \
  --manifest "$OUTPUT_ROOT/base_manifest.json" --output "$OUTPUT_ROOT"
python "$PREPROCESSING/google_v3_pipeline.py" lemmatize \
  --output "$OUTPUT_ROOT" --model en_core_web_lg
python "$PREPROCESSING/google_v3_pipeline.py" vocab --output "$OUTPUT_ROOT" \
  --top-k 50000 --vocab-output "$OUTPUT_ROOT/context_vocab_equal_decade_50000.tsv"
```

Lemmatization uses the supplied Google POS.

## Context extraction and reusable export

For every entry of `context_manifest.json`, run:

```bash
python "$PREPROCESSING/google_v3_pipeline.py" extract \
  --manifest "$OUTPUT_ROOT/context_manifest.json" --task-id 0 \
  --context-vocab "$OUTPUT_ROOT/context_vocab_equal_decade_50000.tsv" \
  --output "$OUTPUT_ROOT"
```

After all entries complete:

```bash
python "$PREPROCESSING/google_v3_pipeline.py" reduce --output "$OUTPUT_ROOT" \
  --manifest "$OUTPUT_ROOT/base_manifest.json" \
  --manifest "$OUTPUT_ROOT/context_manifest.json"
python "$PREPROCESSING/google_v3_pipeline.py" feature-inputs --output "$OUTPUT_ROOT"
```

This completes the preprossing stage. Continue in
[compositionality analysis](../compositionality_over_time/README.md), or pass
the exported bigram tables to [novel-compound prediction](../novel_compound_predictor/README.md).


## Optional annotations

[ANNOTATIONS.md](ANNOTATIONS.md) describes candidate NER and dependency evidence.
These enrich `bigram_annotations.parquet` without replacing the core count and
context tables. NER uses original cased five-token windows.

## Runtime configuration

Activate the appropriate environment before running the Python commands.
Choose explicit input/output paths, thread counts, memory limits and temporary
storage for your workload. See `--help` for further information.


## Tests and source information

```bash
python -m unittest discover -s google_preprocessing/tests -v
```

Tests and the offline example use temporary constructed data. They do not
import a downstream analysis project or read real ratings. Shared infrastructure
lives directly in the root `src/` package. See [SOURCE_INFO.md](SOURCE_INFO.md)
for completion identities.

