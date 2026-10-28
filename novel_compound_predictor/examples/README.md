# Synthetic example

All lexical pairs and counts here are invented for a small execution example.
They are not a corpus extract, evaluation benchmark, or linguistic annotation.

`contexts.csv` contains context events. `counts.csv` independently defines pair
occurrences: a pair can have several context events for one occurrence.
The three 2000s pairs are positive examples relative to these supplied counts;
their large `FUTURE_ONLY_NOUN` contexts must not enter the historical basis.

From the repository root, after `python -m pip install -e '.[novel]'`:

```bash
bash novel_compound_predictor/examples/run_example.sh
```

The script creates a temporary output directory, builds pooled and temporal
two-dimensional representations, trains a tiny pooled classifier, and ranks
the supplied candidates. The historical candidate is excluded and the missing
constituent is reported. Files remain available for inspection at the printed
path. Set `EXAMPLE_OUTPUT` to a new destination to choose another location.

This verifies execution and schema specifications only. Its predictions are not
scientifically meaningful. Full 300-dimensional pooled/LSTM integration flows
are covered by the automated tests.
