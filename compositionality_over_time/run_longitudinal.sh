#!/usr/bin/env bash
# Cumulative Google feature preparation and frozen-model trajectories.
# No Slurm submission occurs here. The default action is a read-only plan.
set -euo pipefail
# Optional trusted local settings; no site environment is activated implicitly.
[[ -z "${COMPOUNDING_CONFIG:-}" ]] || source "$COMPOUNDING_CONFIG"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-plan}
ACTION=${ACTION,,}
if [[ $# -gt 0 ]]; then shift; fi
usage() {
  printf '%s\n' \
    'Usage: bash run_longitudinal.sh [plan|check|prepare|features|fit]' \
    'plan: read-only Python input plan and the frozen-model R command.' \
    'check: verify completed cumulative snapshots and R inputs/dependencies.' \
    'prepare: build exact cumulative counts from the completed feature cache.' \
    'features: write all requested cumulative benchmark snapshots.' \
    'fit: fit the modern model once, predict trajectories, and fit growth models.' \
    'Paths: PREPARED_ROOT, MODERN_ROOT, WORK_ROOT, CUMULATIVE_ROOT, TEMP_DIRECTORY, RESULTS_ROOT.' \
    'Study: CONTEXT_SIZE=30000, TAGGING=UnTagged, TIME_STEP=10, WEIGHTING=PPMI, SETTING=Aware, CUTOFF=0.' \
    'TIME_STEP=10 spaces cumulative endpoints; it does not select disjoint ten-year windows.' \
    'TEMPORAL_FEATURES=1 is the default for this configuration at cutoff 0 or 500; use 0 for the static study.' \
    'Cutoff 0 still requires positive raw counts; cutoff 500 remains available with unchanged scientific drivers.' \
    'Temporal mode defaults to FEATURE_SET=all: static + setting + lagged cumulative cosine predictors.' \
    'Other options: END_DECADES, FEATURE_SET, TRAIN_VARIANT, TEST_VARIANT, LONGITUDINAL_ROOT.' \
    'all_static excludes temporal predictors; representation_only excludes setting and temporal families.' \
    'Runtime: PYTHON_BIN, RSCRIPT_BIN, DUCKDB_MEMORY_LIMIT, MAX_TEMP_DIRECTORY_SIZE, PILOT, FORCE.'
}
case "$ACTION" in
  plan|check|prepare|features|fit) ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { echo 'This wrapper accepts no additional arguments; use the documented environment overrides.' >&2; exit 2; }
source "$SCRIPT_DIR/longitudinal_config.sh"

PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR}
PYTHON_BIN=${PYTHON_BIN:-python3}
R_ENV=${R_ENV:-longitudinal_env}
RSCRIPT_BIN=${RSCRIPT_BIN:-Rscript}
PREPARED_ROOT=${PREPARED_ROOT:-work/compositionality/features/prepared}
MODERN_ROOT=${MODERN_ROOT:-runs/compositionality/regressions/inputs_snapshot}
END_DECADES=${END_DECADES:-'1820 1830 1840 1850 1860 1870 1880 1890 1900 1910 1920 1930 1940 1950 1960 1970 1980 1990 2000 2010'}
if [[ "$TEMPORAL_FEATURES" == 1 ]]; then
  FEATURE_SET=${FEATURE_SET:-all}
else
  FEATURE_SET=${FEATURE_SET:-all_static}
fi
TRAIN_VARIANT=${TRAIN_VARIANT:-modified}
TEST_VARIANT=${TEST_VARIANT:-modified}
DUCKDB_MEMORY_LIMIT=${DUCKDB_MEMORY_LIMIT:-48GB}
FEATURES_DUCKDB_MEMORY_LIMIT=${FEATURES_DUCKDB_MEMORY_LIMIT:-12GB}
MAX_TEMP_DIRECTORY_SIZE=${MAX_TEMP_DIRECTORY_SIZE:-64GB}
FOLDS=${FOLDS:-5}
REPEATS=${REPEATS:-10}
ALPHA_COUNT=${ALPHA_COUNT:-10}
LAMBDA_COUNT=${LAMBDA_COUNT:-200}
PILOT=${PILOT:-0}
FORCE=${FORCE:-0}

for variable in CONTEXT_SIZE TIME_STEP FOLDS REPEATS ALPHA_COUNT LAMBDA_COUNT; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
done
[[ "$CUTOFF" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'CUTOFF must be a non-negative integer' >&2; exit 2; }
for variable in PILOT FORCE; do
  [[ "${!variable}" == 0 || "${!variable}" == 1 ]] || { echo "$variable must be 0 or 1" >&2; exit 2; }
done
if [[ "$TEMPORAL_FEATURES" == 1 ]]; then
  case "$FEATURE_SET" in
    all|all_static|representation_only|temporal) ;;
    *) echo 'Temporal FEATURE_SET must be all, all_static, representation_only, or temporal' >&2; exit 2 ;;
  esac
else
  case "$FEATURE_SET" in
    all_static|representation_only) ;;
    *) echo 'Static FEATURE_SET must be all_static or representation_only' >&2; exit 2 ;;
  esac
fi
for value in "$TRAIN_VARIANT" "$TEST_VARIANT"; do
  case "$value" in modified|original) ;; *) echo 'Rating variants must be modified or original' >&2; exit 2 ;; esac
done
if [[ "$TEMPORAL_FEATURES" == 1 && ( "$TRAIN_VARIANT" != modified || "$TEST_VARIANT" != modified ) ]]; then
  echo 'The scoped temporal study requires modified training and test rating forms' >&2
  exit 2
fi
read -r -a decade_values <<< "${END_DECADES//,/ }"
(( ${#decade_values[@]} > 0 )) || { echo 'END_DECADES must not be empty' >&2; exit 2; }
for decade in "${decade_values[@]}"; do
  [[ "$decade" =~ ^[0-9]{4}$ ]] && (( decade % 10 == 0 )) || {
    echo "END_DECADES requires decade starts, for example 1820 or 2010: $decade" >&2; exit 2;
  }
done

# Every stage uses one CPU; avoid hidden numerical-library thread pools.
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON_PROGRAM="$PROJECT_ROOT/cumulative_features.py"
R_PROGRAM="$PROJECT_ROOT/regression/longitudinal/longitudinal.R"
if [[ "$TEMPORAL_FEATURES" == 1 ]]; then
  PYTHON_PROGRAM="$PROJECT_ROOT/cumulative_temporal.py"
  R_PROGRAM="$PROJECT_ROOT/regression/longitudinal/longitudinal_temporal.R"
  if [[ "$CUTOFF" == 0 ]]; then
    PYTHON_PROGRAM="$PROJECT_ROOT/cumulative_temporal_cutoff0.py"
    R_PROGRAM="$PROJECT_ROOT/regression/longitudinal/longitudinal_temporal_cutoff0.R"
  fi
fi
[[ -f "$PYTHON_PROGRAM" ]] || { echo "Cumulative feature program not found: $PYTHON_PROGRAM" >&2; exit 2; }
[[ -f "$R_PROGRAM" ]] || { echo "Longitudinal R program not found: $R_PROGRAM" >&2; exit 2; }
if [[ "$ACTION" == fit ]]; then
  command -v flock >/dev/null || { echo 'Linux flock is required to guard concurrent fits.' >&2; exit 2; }
  RESULTS_ROOT=$(realpath -m -- "$RESULTS_ROOT")
  for protected in "$MODERN_ROOT" "$CUMULATIVE_ROOT"; do
    protected=$(realpath -m -- "$protected")
    if [[ "$RESULTS_ROOT/" == "$protected/"* || "$protected/" == "$RESULTS_ROOT/"* ]]; then
      echo "RESULTS_ROOT must not overlap modern or cumulative inputs: $protected" >&2
      exit 2
    fi
  done
fi

python_command() {
  local phase=$1 memory_limit=$DUCKDB_MEMORY_LIMIT
  [[ "$phase" != features ]] || memory_limit=$FEATURES_DUCKDB_MEMORY_LIMIT
  local -a command=("$PYTHON_BIN" -u "$PYTHON_PROGRAM" "$phase"
    --prepared-root "$PREPARED_ROOT" --modern-root "$MODERN_ROOT"
    --work-root "$WORK_ROOT" --output-root "$CUMULATIVE_ROOT"
    --temp-directory "$TEMP_DIRECTORY" --context-size "$CONTEXT_SIZE"
    --cutoff "$CUTOFF" --time-step "$TIME_STEP" --end-decades "${decade_values[@]}"
    --condition "Compound$SETTING" --weight "$WEIGHTING" --threads 1
    --memory-limit "$memory_limit" --max-temp-directory-size "$MAX_TEMP_DIRECTORY_SIZE")
  if [[ "$FORCE" == 1 && ( "$phase" == prepare || "$phase" == features ) ]]; then command+=(--force); fi
  "${command[@]}"
}

R_COMMAND=("$RSCRIPT_BIN" "$R_PROGRAM"
  --feature-root "$MODERN_ROOT" --cumulative-root "$CUMULATIVE_ROOT" --output-dir "$RESULTS_ROOT"
  --context-mode local5 --context-size "$CONTEXT_SIZE" --setting "$SETTING" --ppmi "$WEIGHTING"
  --tagging "$TAGGING" --pattern all --cutoff "$CUTOFF" --target compound --feature-set "$FEATURE_SET"
  --train-variant "$TRAIN_VARIANT" --test-variant "$TEST_VARIANT" --time-step "$TIME_STEP"
  --frequency-family freq --productivity-family prod --folds "$FOLDS" --repeats "$REPEATS"
  --alpha-count "$ALPHA_COUNT" --lambda-count "$LAMBDA_COUNT")
[[ "$PILOT" == 0 ]] || R_COMMAND+=(--pilot)

case "$ACTION" in
  plan)
    echo "Study: $LONGITUDINAL_STUDY_ID; cumulative history (not disjoint ten-year windows)"
    echo "Temporal features: $TEMPORAL_FEATURES; feature set: $FEATURE_SET"
    echo "Cumulative features: $CUMULATIVE_ROOT"
    echo "Results: $RESULTS_ROOT"
    echo "Logs: $LONGITUDINAL_LOG_ROOT"
    python_command plan
    printf 'Frozen-model fit command: '
    printf '%q ' "${R_COMMAND[@]}"
    printf '\n'
    ;;
  check)
    python_command check
    "${R_COMMAND[@]}" --check
    ;;
  prepare|features) python_command "$ACTION" ;;
  fit)
    [[ "$FORCE" == 0 ]] || R_COMMAND+=(--force)
    mkdir -p "$RESULTS_ROOT"
    flock -n "$RESULTS_ROOT/.fit.lock" "${R_COMMAND[@]}"
    ;;
esac
