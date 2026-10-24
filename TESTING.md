# Testing

Tests use small synthetic tables and temporary directories. They do not require
full corpus data, human rating datasets or downloaded language models.

## Python

Install the relevant dependency extras in an isolated environment. Use `.[all]`
to run every project's Python suite, or install the extra for the project being
tested. From repository root:

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s google_preprocessing/tests -v
python -m unittest discover -s compositionality_over_time/tests -v
python -m unittest discover -s novel_compound_predictor/tests -v
```

Shared tests cover atomic output, schema and source information validation,
source identities, resource-aware logging and project boundaries. Project suites
exercise extraction/export, feature calculations, model preparation and small
representation/prediction workflows. Optional checks report an explicit skip
when their local dependencies are unavailable.

The examples in each project's guide provide short command-line smoke tests.

## R

Install the R packages listed in the compositionality project. Run the regression
and longitudinal test scripts from a temporary working directory to keep R
graphics/session artifacts out of the source tree:

```bash
TEST_ROOT="$PWD/compositionality_over_time/regression/tests"
TEST_WORK=$(mktemp -d)
(
  cd "$TEST_WORK"
  for test_script in "$TEST_ROOT"/*.R; do
    TZ=UTC OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      Rscript "$test_script"
  done
)
```

These tests check training-only preprocessing, rating-roster selection,
reported statistics, model freezing, trajectory predictions and exact
checkpoint identities. Synthetic resampling cases can produce non-fatal
warnings; failed assertions and nonzero exit statuses require investigation.

## Source-tree checks

The root suite checks for unsupported cross-project imports, private paths,
credential-like literals, notebooks, large files and accidental generated
outputs. Keep runtime data, fitted models and temporary outputs outside the
source tree. Passing these checks complements manual review; it does not
certify external datasets or every possible credential format.
