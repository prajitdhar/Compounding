# Synthetic regression test data

These files are fabricated test inputs, not Reddy++ data or human judgments.
They reproduce the file columns and 180/100 split sizes so tests can run without
redistributing a benchmark. Seven spelling/lemma examples exercise the explicit
lexical-alternative specification; their scores are invented too.

Regenerate from the repository root with:

```bash
python compositionality_over_time/tests/test_data/generate_ratings.py \
  compositionality_over_time/tests/test_data/ratings
```

Do not use these test data to reproduce reported experimental scores. Supply
separately obtained benchmark files through `RATINGS_DIR` for actual studies.
