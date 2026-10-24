#!/usr/bin/env bash
# Run the latest COHA-compatible regression workflow on frozen Google features.
set -euo pipefail
# Optional trusted local settings; no site environment is activated implicitly.
[[ -z "${COMPOUNDING_CONFIG:-}" ]] || source "$COMPOUNDING_CONFIG"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
  cat <<'EOF'
Usage: bash run_regressions.sh [plan|check|fit|report|all]

Inputs must be a validated CSV snapshot made by regression_inputs.py.
Default: both taggings, six vocabularies, two weightings, three settings,
five timespans and six cutoffs, using Google local5 only.
Set BATCH_INDEX to select one tagging/vocabulary/weighting/setting batch.
Indices follow TAGGINGS > VOCABS > WEIGHTINGS > SETTINGS (last varies fastest).
Tagged batches automatically fit separate NN and adjective-aware AN rosters.

Paths: INPUT_ROOT, OUT, PYTHON_BIN, RSCRIPT_BIN (or R_ENV).
Grid: TAGGINGS, VOCABS, WEIGHTINGS, SETTINGS, TIMESPANS, CUTOFFS, TARGETS.
Fitting: JOBS, CORES, FOLDS, REPEATS, ALPHA_COUNT, LAMBDA_COUNT, PERMUTATIONS,
TRAIN_VARIANT, TEST_VARIANT, PILOT, FORCE. Reports always use the full grid.
JOBS defaults to one; JOBS * CORES must not exceed a Slurm CPU allocation.
EOF
}

ACTION=${1:-all}
case "$ACTION" in
  plan|check|fit|report|all) ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

INPUT_ROOT=${INPUT_ROOT:-runs/compositionality/regressions/inputs_snapshot}
if [[ -z "${OUT:-}" ]]; then
  OUT=runs/compositionality/regressions/results
  [[ "${PILOT:-0}" != 1 ]] || OUT+=_pilot
fi
PYTHON_BIN=${PYTHON_BIN:-python3}
R_ENV=${R_ENV:-r_env}
RSCRIPT_BIN=${RSCRIPT_BIN:-Rscript}
MODES=${MODES:-local5}
TAGGINGS=${TAGGINGS:-"UnTagged Tagged"}
VOCABS=${VOCABS:-"1000 3000 5000 10000 30000 50000"}
WEIGHTINGS=${WEIGHTINGS:-"RAW PPMI"}
SETTINGS=${SETTINGS:-"Agnostic Pooled Aware"}
TIMESPANS=${TIMESPANS:-"10000 10 20 50 100"}
CUTOFFS=${CUTOFFS:-"0 10 50 100 500 1000"}
TARGETS=${TARGETS:-"compound head modifier"}
CORES=${CORES:-1}
JOBS=${JOBS:-1}
FOLDS=${FOLDS:-5}
REPEATS=${REPEATS:-10}
ALPHA_COUNT=${ALPHA_COUNT:-10}
LAMBDA_COUNT=${LAMBDA_COUNT:-200}
PERMUTATIONS=${PERMUTATIONS:-5}
N_BOOT=${N_BOOT:-2000}
N_PERM=${N_PERM:-5000}
TRAIN_VARIANT=${TRAIN_VARIANT:-modified}
TEST_VARIANT=${TEST_VARIANT:-modified}
REPORT_CUTOFF=${REPORT_CUTOFF:-0}
RUN_TEMPORAL=${RUN_TEMPORAL:-1}
RUN_FAMILIES=${RUN_FAMILIES:-1}
RUN_DIRECT_BASELINE=${RUN_DIRECT_BASELINE:-1}
RUN_ENCODINGS=${RUN_ENCODINGS:-0}
REFERENCE_MODE=${REFERENCE_MODE:-local5}
REFERENCE_VOCAB=${REFERENCE_VOCAB:-5000}
REFERENCE_WEIGHTING=${REFERENCE_WEIGHTING:-PPMI}
REFERENCE_SETTING=${REFERENCE_SETTING:-Agnostic}
PILOT=${PILOT:-0}
FORCE=${FORCE:-0}
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

normalize_words() {
  local value=${1//,/ } word
  local -a words unique_words=()
  local -A seen=()
  read -r -a words <<< "$value"
  for word in "${words[@]}"; do
    if [[ -z "${seen[$word]:-}" ]]; then unique_words+=("$word"); seen[$word]=1; fi
  done
  printf '%s' "${unique_words[*]}"
}
for variable in MODES TAGGINGS VOCABS WEIGHTINGS SETTINGS TIMESPANS CUTOFFS TARGETS; do
  printf -v "$variable" '%s' "$(normalize_words "${!variable}")"
  [[ -n "${!variable}" ]] || { echo "Empty $variable" >&2; exit 2; }
done
[[ "$MODES" == local5 && "$REFERENCE_MODE" == local5 ]] || {
  echo "Google regression supports local5 only; there is no sentence mode" >&2; exit 2;
}
for value in $TAGGINGS; do
  case "$value" in Tagged|UnTagged) ;; *) echo "Invalid TAGGINGS: $value" >&2; exit 2 ;; esac
done
for value in $WEIGHTINGS; do
  case "$value" in RAW|PPMI) ;; *) echo "Invalid WEIGHTINGS: $value" >&2; exit 2 ;; esac
done
for value in $SETTINGS; do
  case "$value" in Agnostic|Aware|Pooled) ;; *) echo "Invalid SETTINGS: $value" >&2; exit 2 ;; esac
done
for value in $TARGETS; do
  case "$value" in compound|head|modifier) ;; *) echo "Invalid TARGETS: $value" >&2; exit 2 ;; esac
done
for variable in CORES JOBS FOLDS REPEATS ALPHA_COUNT LAMBDA_COUNT; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] || { echo "$variable must be a positive integer" >&2; exit 2; }
done
for variable in FOLDS ALPHA_COUNT LAMBDA_COUNT; do
  (( ${!variable} >= 2 )) || { echo "$variable must be at least 2" >&2; exit 2; }
done
for value in $VOCABS $TIMESPANS; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Vocabulary sizes and timespans must be positive integers" >&2; exit 2; }
done
for value in $CUTOFFS "$PERMUTATIONS" "$REPORT_CUTOFF"; do
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Cutoffs and permutations must be non-negative integers" >&2; exit 2; }
done
(( PERMUTATIONS >= 1 )) || { echo "PERMUTATIONS must be at least 1" >&2; exit 2; }
for variable in N_BOOT N_PERM; do
  [[ "${!variable}" =~ ^[1-9][0-9]*$ ]] && (( ${!variable} >= 100 )) || {
    echo "$variable must be at least 100" >&2; exit 2;
  }
done
for value in "$TRAIN_VARIANT" "$TEST_VARIANT"; do
  case "$value" in original|modified) ;; *) echo "Variants must be original or modified" >&2; exit 2 ;; esac
done
for variable in RUN_TEMPORAL RUN_FAMILIES RUN_DIRECT_BASELINE RUN_ENCODINGS PILOT FORCE; do
  [[ "${!variable}" == 0 || "${!variable}" == 1 ]] || { echo "$variable must be 0 or 1" >&2; exit 2; }
done
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && (( JOBS * CORES > SLURM_CPUS_PER_TASK )); then
  echo "JOBS * CORES ($(( JOBS * CORES ))) exceeds SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" >&2
  exit 2
fi
[[ "$RUN_TEMPORAL" == 1 ]] || TIMESPANS=10000
read -r -a tag_values <<< "$TAGGINGS"
read -r -a vocab_values <<< "$VOCABS"
read -r -a weighting_values <<< "$WEIGHTINGS"
read -r -a setting_values <<< "$SETTINGS"
read -r -a timespan_values <<< "$TIMESPANS"
read -r -a cutoff_values <<< "$CUTOFFS"
batch_count=$(( ${#tag_values[@]} * ${#vocab_values[@]} * ${#weighting_values[@]} * ${#setting_values[@]} ))
if [[ -n "${BATCH_INDEX:-}" ]]; then
  [[ "$BATCH_INDEX" =~ ^(0|[1-9][0-9]*)$ ]] && (( BATCH_INDEX < batch_count )) || {
    echo "BATCH_INDEX must be 0 through $(( batch_count - 1 ))" >&2; exit 2;
  }
  if [[ "$ACTION" == report || "$ACTION" == all ]]; then
    echo "Report/all require the complete requested grid; unset BATCH_INDEX" >&2; exit 2
  fi
fi
if [[ "$ACTION" == report || "$ACTION" == all ]]; then
  [[ " $CUTOFFS " == *" $REPORT_CUTOFF "* && " $TIMESPANS " == *" 10000 "* ]] || {
    echo "Report requires REPORT_CUTOFF and pooled timespan 10000 in the requested grid" >&2; exit 2;
  }
fi
TARGETS_CSV=${TARGETS// /,}
CUTOFFS_CSV=${CUTOFFS// /,}
TIMESPANS_CSV=${TIMESPANS// /,}

verify_snapshot() {
  local -a chosen_tags=("${tag_values[@]}") chosen_vocabs=("${vocab_values[@]}")
  local -a chosen_weights=("${weighting_values[@]}") chosen_settings=("${setting_values[@]}")
  if [[ -n "${BATCH_INDEX:-}" ]]; then
    local index=$BATCH_INDEX
    chosen_settings=("${setting_values[index % ${#setting_values[@]}]}"); index=$(( index / ${#setting_values[@]} ))
    chosen_weights=("${weighting_values[index % ${#weighting_values[@]}]}"); index=$(( index / ${#weighting_values[@]} ))
    chosen_vocabs=("${vocab_values[index % ${#vocab_values[@]}]}"); index=$(( index / ${#vocab_values[@]} ))
    chosen_tags=("${tag_values[index]}")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/regression_inputs.py" verify --snapshot-root "$INPUT_ROOT" \
    --vocabs "${chosen_vocabs[@]}" --taggings "${chosen_tags[@]}" \
    --weightings "${chosen_weights[@]}" --settings "${chosen_settings[@]}" \
    --timespans "${timespan_values[@]}" --cutoffs "${cutoff_values[@]}"
}

run_one() {
  local phase=$1 tagging=$2 vocab=$3 weighting=$4 setting=$5
  local frequency=${6:-freq} productivity=${7:-prod} timespans=${8:-$TIMESPANS_CSV}
  local sets=${9:-all_no_setting,representation_only,cordeiro,sim_cpf_beta,all}
  if [[ $# -lt 9 ]]; then
    [[ "$RUN_TEMPORAL" == 0 ]] || sets+=,all_static,temporal
    [[ "$setting" == Agnostic ]] || sets+=,with_setting
    if [[ "$RUN_FAMILIES" == 1 && "$vocab" == "$REFERENCE_VOCAB" &&
          "$weighting" == "$REFERENCE_WEIGHTING" && "$setting" == "$REFERENCE_SETTING" ]]; then
      sets+=,cosine_sim,info_theory,association_counts,freq,log_freq,tf,prod,family_size
    fi
  fi
  local -a command=("$RSCRIPT_BIN" "$SCRIPT_DIR/regression/modified_regressions/regression_exps.r"
    --input-root "$INPUT_ROOT" --output-dir "$OUT" --context-mode local5 --context-size "$vocab"
    --ppmi "$weighting" --setting "$setting" --tagging "$tagging"
    --train-variant "$TRAIN_VARIANT" --test-variant "$TEST_VARIANT"
    --timespans "$timespans" --cutoffs "$CUTOFFS_CSV" --targets "$TARGETS_CSV"
    --frequency-family "$frequency" --productivity-family "$productivity" --feature-sets "$sets"
    --cores "$CORES" --folds "$FOLDS" --repeats "$REPEATS" --alpha-count "$ALPHA_COUNT"
    --lambda-count "$LAMBDA_COUNT" --permutations "$PERMUTATIONS")
  [[ "$PILOT" == 0 ]] || command+=(--pilot)
  [[ "$FORCE" == 0 || "$phase" != fit ]] || command+=(--force)
  case "$phase" in
    plan) printf '%q ' "${command[@]}"; printf '\n' ;;
    check) "${command[@]}" --validate-only ;;
    fit)
      mkdir -p "$OUT/logs"
      local lock="$OUT/logs/local5_ctx${vocab}_${weighting}_${setting}_${frequency}_${productivity}_${tagging}_train${TRAIN_VARIANT}_test${TEST_VARIANT}.lock"
      if (( JOBS > 1 )); then exec setsid --wait flock --close "$lock" "${command[@]}"
      else flock --close "$lock" "${command[@]}"; fi
      ;;
  esac
}

job_pids=()
job_labels=()
wait_for_job() {
  local finished status=0 i running
  if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )); then
    wait -n -p finished "${job_pids[@]}" || status=$?
    if [[ -z "${finished:-}" ]]; then
      finished=${job_pids[0]}; status=0; wait "$finished" || status=$?
    fi
  else
    # Cluster login/compute nodes may still have Bash 4.4 (no wait -p).
    # Discover completed children before waiting, so very fast jobs are not lost.
    while [[ -z "${finished:-}" ]]; do
      running=$(jobs -pr)
      running=" ${running//$'\n'/ } "
      for i in "${!job_pids[@]}"; do
        if [[ "$running" != *" ${job_pids[i]} "* ]]; then
          finished=${job_pids[i]}; wait "$finished" || status=$?; break
        fi
      done
      [[ -n "${finished:-}" ]] || { wait -n || :; }
    done
  fi
  for i in "${!job_pids[@]}"; do
    if [[ "${job_pids[i]}" == "$finished" ]]; then
      (( status == 0 )) || echo "Fit failed ($status): ${job_labels[i]}; see $fit_log_dir" >&2
      unset 'job_pids[i]' 'job_labels[i]'; break
    fi
  done
  job_pids=("${job_pids[@]}"); job_labels=("${job_labels[@]}")
  return "$status"
}
wait_for_all_jobs() {
  local status=0
  while (( ${#job_pids[@]} )); do wait_for_job || status=$?; done
  return "$status"
}
stop_grid_jobs() {
  local status=$1 pid
  trap '' INT TERM
  for pid in "${job_pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || :; done
  for pid in "${job_pids[@]}"; do wait "$pid" 2>/dev/null || :; done
  exit "$status"
}
schedule_one() {
  local phase=$1; shift
  if [[ "$phase" != fit ]] || (( JOBS == 1 )); then run_one "$phase" "$@"; return; fi
  if (( ${#job_pids[@]} >= JOBS )); then
    local status=0
    wait_for_job || status=$?
    if (( status != 0 )); then wait_for_all_jobs || :; return "$status"; fi
  fi
  local label="local5_${1}_ctx${2}_${3}_${4}_${5:-freq}_${6:-prod}_train${TRAIN_VARIANT}_test${TEST_VARIANT}"
  echo "Starting $label ($CORES workers); log: $fit_log_dir/$label.log" >&2
  run_one fit "$@" >"$fit_log_dir/$label.log" 2>&1 &
  job_pids+=("$!"); job_labels+=("$label")
}
run_grid() {
  local phase=$1 tagging vocab weighting setting index=0 fit_log_dir
  if [[ "$phase" == fit ]] && (( JOBS > 1 )); then
    mkdir -p "$OUT/logs"
    fit_log_dir=$(mktemp -d "$OUT/logs/run.XXXXXXXX")
    trap 'stop_grid_jobs 130' INT
    trap 'stop_grid_jobs 143' TERM
  fi
  for tagging in "${tag_values[@]}"; do
    for vocab in "${vocab_values[@]}"; do
      for weighting in "${weighting_values[@]}"; do
        for setting in "${setting_values[@]}"; do
          if [[ -z "${BATCH_INDEX:-}" || "$index" == "$BATCH_INDEX" ]]; then
            schedule_one "$phase" "$tagging" "$vocab" "$weighting" "$setting"
            if [[ "$RUN_ENCODINGS" == 1 && "$vocab" == "$REFERENCE_VOCAB" &&
                  "$weighting" == "$REFERENCE_WEIGHTING" && "$setting" == "$REFERENCE_SETTING" ]]; then
              schedule_one "$phase" "$tagging" "$vocab" "$weighting" "$setting" log_freq prod 10000 all_no_setting
              schedule_one "$phase" "$tagging" "$vocab" "$weighting" "$setting" tf prod 10000 all_no_setting
              schedule_one "$phase" "$tagging" "$vocab" "$weighting" "$setting" freq family_size 10000 all_no_setting
            fi
          fi
          index=$(( index + 1 ))
        done
      done
    done
  done
  local status=0
  wait_for_all_jobs || status=$?
  trap - INT TERM
  return "$status"
}
write_report() {
  mkdir -p "$OUT/report"
  export GRID_MODES=local5 GRID_VOCABS="$VOCABS" GRID_WEIGHTINGS="$WEIGHTINGS"
  export GRID_SETTINGS="$SETTINGS" GRID_TAGGINGS="$TAGGINGS" GRID_TARGETS="$TARGETS"
  export GRID_TIMESPANS="$TIMESPANS_CSV" GRID_CUTOFFS="$CUTOFFS_CSV"
  export GRID_TRAIN_VARIANT="$TRAIN_VARIANT" GRID_TEST_VARIANT="$TEST_VARIANT"
  export GRID_FOLDS="$FOLDS" GRID_REPEATS="$REPEATS" GRID_ALPHA_COUNT="$ALPHA_COUNT"
  export GRID_LAMBDA_COUNT="$LAMBDA_COUNT" GRID_PERMUTATIONS="$PERMUTATIONS"
  if [[ "$PILOT" == 1 ]]; then
    export GRID_FOLDS=3 GRID_REPEATS=1 GRID_ALPHA_COUNT=3
    export GRID_LAMBDA_COUNT=12 GRID_PERMUTATIONS=1
  fi
  export GRID_INPUT_ROOT="$INPUT_ROOT" GRID_OUT="$OUT"
  COMPOUNDING_REGRESSION_PATH="$OUT" COMPOUNDING_SUMMARY_PATH="$OUT/report" \
    "$RSCRIPT_BIN" "$SCRIPT_DIR/regression/modified_regressions/regression_statistics.R"
  GRID_OUT="$OUT" GRID_INPUT_ROOT="$INPUT_ROOT" GRID_DIRECT_BASELINE="$RUN_DIRECT_BASELINE" \
    GRID_CUTOFF="$REPORT_CUTOFF" GRID_N_BOOT="$N_BOOT" GRID_N_PERM="$N_PERM" \
    GRID_REFERENCE_MODE=local5 GRID_REFERENCE_VOCAB="$REFERENCE_VOCAB" \
    GRID_REFERENCE_WEIGHTING="$REFERENCE_WEIGHTING" GRID_REFERENCE_SETTING="$REFERENCE_SETTING" \
    GRID_RUN_TEMPORAL="$RUN_TEMPORAL" GRID_RUN_ENCODINGS="$RUN_ENCODINGS" \
    "$RSCRIPT_BIN" "$SCRIPT_DIR/regression/modified_regressions/workflow_report.R"
}

if [[ "$ACTION" != plan ]]; then
  command -v "$RSCRIPT_BIN" >/dev/null 2>&1 || { echo "Rscript not executable: $RSCRIPT_BIN (set RSCRIPT_BIN or R_ENV)" >&2; exit 2; }
  verify_snapshot
fi
case "$ACTION" in
  plan|check|fit) run_grid "$ACTION" ;;
  report) write_report ;;
  all) run_grid check; run_grid fit; write_report ;;
esac
