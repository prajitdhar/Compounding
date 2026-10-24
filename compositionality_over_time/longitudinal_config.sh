#!/usr/bin/env bash
# Shared study defaults for the local, Slurm and submission launchers.
# Sourcing this file only validates configuration and exports paths; no I/O.
CONTEXT_SIZE=${CONTEXT_SIZE:-30000}
CUTOFF=${CUTOFF:-0}
TIME_STEP=${TIME_STEP:-10}
SETTING=${SETTING:-Aware}
WEIGHTING=${WEIGHTING:-PPMI}
TAGGING=${TAGGING:-UnTagged}

[[ "$CONTEXT_SIZE" =~ ^[1-9][0-9]*$ ]] || {
  echo 'CONTEXT_SIZE must be a positive integer' >&2; return 2;
}
[[ "$CUTOFF" =~ ^(0|[1-9][0-9]*)$ ]] || {
  echo 'CUTOFF must be a non-negative integer' >&2; return 2;
}
[[ "$TIME_STEP" == 10 ]] || {
  echo 'TIME_STEP must be 10: this workflow uses decade-spaced cumulative endpoints' >&2; return 2;
}
case "$SETTING" in
  Aware|Pooled|Agnostic) ;;
  *) echo 'SETTING must be Aware, Pooled, or Agnostic' >&2; return 2 ;;
esac
case "$WEIGHTING" in
  PPMI|RAW) ;;
  *) echo 'WEIGHTING must be PPMI or RAW' >&2; return 2 ;;
esac
[[ "$TAGGING" == UnTagged ]] || {
  echo 'TAGGING must be UnTagged: cumulative preparation does not support Tagged targets' >&2; return 2;
}

longitudinal_temporal_scope=0
if [[ "$CONTEXT_SIZE" == 30000 && ( "$CUTOFF" == 0 || "$CUTOFF" == 500 ) && "$SETTING" == Aware && "$WEIGHTING" == PPMI ]]; then
  longitudinal_temporal_scope=1
fi
TEMPORAL_FEATURES=${TEMPORAL_FEATURES:-$longitudinal_temporal_scope}
[[ "$TEMPORAL_FEATURES" == 0 || "$TEMPORAL_FEATURES" == 1 ]] || {
  echo 'TEMPORAL_FEATURES must be 0 or 1' >&2; return 2;
}
if [[ "$TEMPORAL_FEATURES" == 1 && "$longitudinal_temporal_scope" != 1 ]]; then
  echo 'Temporal features support only 30000 / UnTagged / 10 / PPMI / Aware / cutoff 0 or 500' >&2
  return 2
fi
LONGITUDINAL_BASE_STUDY_ID="ctx${CONTEXT_SIZE}_${TAGGING}_${WEIGHTING}_${SETTING}_${TIME_STEP}_${CUTOFF}"
LONGITUDINAL_STUDY_ID=$LONGITUDINAL_BASE_STUDY_ID
[[ "$TEMPORAL_FEATURES" == 0 ]] || LONGITUDINAL_STUDY_ID+=_cumulative_lag10
LONGITUDINAL_ROOT=${LONGITUDINAL_ROOT:-runs/compositionality/longitudinal/$LONGITUDINAL_STUDY_ID}
# Exact count/type summaries are shared with the matching static configuration;
# only features/models change. The temporal driver verifies this count identity.
WORK_ROOT=${WORK_ROOT:-work/compositionality/longitudinal/$LONGITUDINAL_BASE_STUDY_ID}
TEMP_DIRECTORY=${TEMP_DIRECTORY:-work/compositionality/longitudinal_tmp/$LONGITUDINAL_STUDY_ID}
CUMULATIVE_ROOT=${CUMULATIVE_ROOT:-$LONGITUDINAL_ROOT/features_cumulative}
RESULTS_ROOT=${RESULTS_ROOT:-$LONGITUDINAL_ROOT/results}
LONGITUDINAL_LOG_ROOT=${LONGITUDINAL_LOG_ROOT:-$LONGITUDINAL_ROOT/logs}
export CONTEXT_SIZE CUTOFF TIME_STEP SETTING WEIGHTING TAGGING TEMPORAL_FEATURES
export LONGITUDINAL_STUDY_ID LONGITUDINAL_ROOT WORK_ROOT TEMP_DIRECTORY
export CUMULATIVE_ROOT RESULTS_ROOT LONGITUDINAL_LOG_ROOT
