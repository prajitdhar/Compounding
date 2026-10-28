# Small COHA extraction example

The text in `raw/` is invented prose with COHA-style document markers, dates and
missing-text gaps. `metadata.tsv` demonstrates document metadata. No actual
COHA text or human ratings are included.

Install `.[coha]` and a compatible English spaCy pipeline as described in the
[preprocessing guide](../README.md). From repository root:

```bash
python coha_preprocessing/coha_pipeline.py prepare \
  --input coha_preprocessing/examples/raw \
  --metadata coha_preprocessing/examples/metadata.tsv \
  --output runs/coha-example/prepared --corpus-release synthetic

python coha_preprocessing/coha_pipeline.py extract \
  --input runs/coha-example/prepared --output runs/coha-example/extracted \
  --model en_core_web_lg --workers 1 --scratch-dir runs/coha-example/scratch

python coha_preprocessing/vocabulary.py \
  --input runs/coha-example/extracted/feature_inputs/sentence \
  --output runs/coha-example/context_vocabulary.tsv --size 0 \
  --memory-limit 1GB --threads 1 --temp-directory runs/coha-example/scratch
```

Use a fresh output directory when rerunning with different settings. The
example retains outputs under the ignored `runs/` directory so you can inspect
both context modes, their manifests and annotation/coverage tables. The
vocabulary TSV can be used with either mode.

For feature generation, obtain the rating files separately and follow
[FEATURES.md](../../compositionality_over_time/FEATURES.md), setting
`FEATURE_INPUT` to the desired example mode directory and choosing a context
size no larger than the vocabulary produced. For example, pass
`--context-sizes 10 --temporal-bins 10` to both analysis commands, and
`--context-size 10 --temporal 10` to the `features` command. These tiny examples verify the
workflow; their statistics are not corpus results.
