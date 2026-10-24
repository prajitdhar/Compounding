# Small offline example

After installing the repository's `preprocessing` extra, run from repository root:

```bash
python google_preprocessing/examples/run_tiny_extraction.py
```

The example compresses synthetic n-gram test data, extracts lexical
bigrams, and reduces their decade counts with a bounded single-worker query.
It does not download a corpus or require model files or benchmark ratings.
Temporary outputs are removed when the example exits. Pass `--output runs/tiny`
to retain them in a new directory. The observations are synthetic test data.

No benchmark ratings or downstream analysis code are needed. Actual corpus
tables must be obtained separately when preparing research inputs.
