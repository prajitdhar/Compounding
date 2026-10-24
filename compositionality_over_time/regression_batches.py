#!/usr/bin/env python3
"""Checkpoint-aware one-core regression cells using the unchanged R fitter.

Each task selects one tagging/vocabulary/weighting/setting/time-width/cutoff.
Tagged still runs NN and adjective-aware AN, with its completed pattern reused.
Plans are immutable and include the frozen snapshot and code identities. No
Slurm jobs are submitted here; the separate shell helper owns submission.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile

from regression_inputs import feature_filename, verify_snapshot

DIRECTORY = Path(__file__).resolve().parent
R_DIRECTORY = DIRECTORY / "regression" / "modified_regressions"
FITTER = R_DIRECTORY / "regression_exps.r"
STATUS_SCRIPT = R_DIRECTORY / "batch_status.R"
PLAN_VERSION = 1
BASE_SETS = ["all_no_setting", "representation_only", "cordeiro", "sim_cpf_beta", "all"]
FAMILY_SETS = ["cosine_sim", "info_theory", "association_counts", "freq", "log_freq",
               "tf", "prod", "family_size"]
DEFAULT_GRID = {"TAGGINGS": ["UnTagged", "Tagged"],
                "VOCABS": [1000, 3000, 5000, 10000, 30000, 50000],
                "WEIGHTINGS": ["RAW", "PPMI"], "SETTINGS": ["Agnostic", "Pooled", "Aware"],
                "TIMESPANS": [10000, 10, 20, 50, 100], "CUTOFFS": [0, 10, 50, 100, 500, 1000]}
TASK_FIELDS = ["task_id", "tagging", "vocab", "weight", "setting", "timespan", "cutoff",
               "input_root", "out", "feature_sets", "targets", "frequency_family",
               "productivity_family", "folds", "repeats", "alpha_count", "lambda_count",
               "permutations", "train_variant", "test_variant", "context_mode"]


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _integer(value, label, minimum=1):
    if not re.fullmatch(r"[0-9]+", str(value)) or int(value) < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _words(value):
    return list(dict.fromkeys(part for part in re.split(r"[,\s]+", value.strip()) if part))


def configuration(environ=None):
    env = os.environ if environ is None else environ
    pilot = _integer(env.get("PILOT", "0"), "PILOT", 0)
    if pilot not in (0, 1):
        raise ValueError("PILOT must be 0 or 1")
    if env.get("FORCE", "0") != "0":
        raise ValueError("The resumable batcher never forces replacement of completed fits")
    if _integer(env.get("CORES", "1"), "CORES") != 1 or _integer(env.get("JOBS", "1"), "JOBS") != 1:
        raise ValueError("Batched tasks require CORES=1 and JOBS=1; parallelism is across tasks")
    if env.get("MODES", "local5") != "local5" or env.get("REFERENCE_MODE", "local5") != "local5":
        raise ValueError("Google supports local5 only")
    grid = {}
    for name, default in DEFAULT_GRID.items():
        values = _words(env.get(name, " ".join(map(str, default))))
        if not values:
            raise ValueError(f"Empty {name}")
        if name in ("VOCABS", "TIMESPANS", "CUTOFFS"):
            values = [_integer(value, name, 0 if name == "CUTOFFS" else 1) for value in values]
        elif any(value not in default for value in values):
            raise ValueError(f"Invalid {name}: {values}")
        grid[name] = values
    flags = {}
    for name, default in (("RUN_TEMPORAL", "1"), ("RUN_FAMILIES", "1"), ("RUN_ENCODINGS", "0")):
        if env.get(name, default) not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
        flags[name] = env.get(name, default) == "1"
    if not flags["RUN_TEMPORAL"]:
        grid["TIMESPANS"] = [10000]
    targets = _words(env.get("TARGETS", "compound head modifier"))
    if not targets or any(value not in ("compound", "head", "modifier") for value in targets):
        raise ValueError("Invalid TARGETS")
    tuning = {name: _integer(env.get(variable, str(default)), variable, minimum)
              for name, variable, default, minimum in
              (("folds", "FOLDS", 5, 2), ("repeats", "REPEATS", 10, 1),
               ("alpha_count", "ALPHA_COUNT", 10, 2), ("lambda_count", "LAMBDA_COUNT", 200, 2),
               ("permutations", "PERMUTATIONS", 5, 1))}
    if pilot:
        tuning.update(folds=3, repeats=1, alpha_count=3, lambda_count=12, permutations=1)
    variants = {name: env.get(variable, "modified") for name, variable in
                (("train_variant", "TRAIN_VARIANT"), ("test_variant", "TEST_VARIANT"))}
    if any(value not in ("original", "modified") for value in variants.values()):
        raise ValueError("Variants must be original or modified")
    reference = {"vocab": _integer(env.get("REFERENCE_VOCAB", "5000"), "REFERENCE_VOCAB"),
                 "weight": env.get("REFERENCE_WEIGHTING", "PPMI"),
                 "setting": env.get("REFERENCE_SETTING", "Agnostic")}
    if reference["weight"] not in DEFAULT_GRID["WEIGHTINGS"] or reference["setting"] not in DEFAULT_GRID["SETTINGS"]:
        raise ValueError("Invalid reference weighting/setting")
    out = Path(env.get("OUT", "runs/compositionality/regressions/results" + ("_pilot" if pilot else ""))).resolve()
    executable = env.get("RSCRIPT_BIN", "Rscript")
    rscript = Path(shutil.which(executable) or executable).resolve()
    return {"grid": grid, "flags": flags, "targets": targets, "tuning": tuning,
            "variants": variants, "reference": reference, "pilot": bool(pilot),
            "input_root": str(Path(env.get("INPUT_ROOT", "runs/compositionality/regressions/inputs_snapshot")).resolve()),
            "out": str(out), "rscript": str(rscript)}


def _task_id(task):
    # Lock identity matches output filenames, not requested tuning/recipe. Two
    # incompatible studies must not concurrently write the same named outputs.
    keys = ["tagging", "vocab", "weight", "setting", "timespan", "cutoff",
            "frequency_family", "productivity_family", "train_variant", "test_variant"]
    return _digest({key: task[key] for key in keys})[:24]


def tasks_for(config):
    tasks = []
    g = config["grid"]
    axes = [g[name] for name in ("TAGGINGS", "VOCABS", "WEIGHTINGS", "SETTINGS")]
    for tag, vocab, weight, setting in itertools.product(*axes):
        reference = config["reference"] == dict(vocab=vocab, weight=weight, setting=setting)
        sets = list(BASE_SETS)
        if config["flags"]["RUN_TEMPORAL"]:
            sets += ["all_static", "temporal"]
        if setting != "Agnostic":
            sets += ["with_setting"]
        if config["flags"]["RUN_FAMILIES"] and reference:
            sets += FAMILY_SETS
        recipes = [("freq", "prod", g["TIMESPANS"], sets)]
        if config["flags"]["RUN_ENCODINGS"] and reference:
            recipes += [("log_freq", "prod", [10000], ["all_no_setting"]),
                        ("tf", "prod", [10000], ["all_no_setting"]),
                        ("freq", "family_size", [10000], ["all_no_setting"])]
        for frequency, productivity, widths, feature_sets in recipes:
            for width, cutoff in itertools.product(widths, g["CUTOFFS"]):
                task = {"tagging": tag, "vocab": vocab, "weight": weight, "setting": setting,
                        "timespan": width, "cutoff": cutoff, "frequency_family": frequency,
                        "productivity_family": productivity, "feature_sets": ",".join(feature_sets),
                        "targets": ",".join(config["targets"]), "input_root": config["input_root"],
                        "out": config["out"], "context_mode": "local5",
                        **config["tuning"], **config["variants"]}
                task["task_id"] = _task_id(task)
                tasks.append(task)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task/output identities")
    return tasks


def _input_names(tasks):
    names = set()
    for task in tasks:
        for family in (["features"] if task["timespan"] == 10000 else ["features", "temporal"]):
            names.add(feature_filename(family, task["setting"], task["weight"], task["tagging"],
                                       task["vocab"], task["timespan"], task["cutoff"]))
    return sorted(names)


def checkpoint_status(tasks, rscript):
    with tempfile.TemporaryDirectory(prefix="google-regression-status-") as temporary:
        path = Path(temporary) / "tasks.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TASK_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows({key: task[key] for key in TASK_FIELDS} for task in tasks)
        checked = subprocess.run([rscript, str(STATUS_SCRIPT), "--tasks-tsv", str(path)],
                                 text=True, capture_output=True, check=True)
    rows = list(csv.DictReader(io.StringIO(checked.stdout), delimiter="\t"))
    expected = {task["task_id"] for task in tasks}
    if len(rows) != len(tasks) or {row["task_id"] for row in rows} != expected:
        raise RuntimeError("Checkpoint verifier returned wrong/duplicate task identities")
    if any(row["status"] not in ("completed", "pending", "incompatible") for row in rows):
        raise RuntimeError("Unknown checkpoint status")
    return {row["task_id"]: row for row in rows}


def _compatible(statuses):
    bad = [row for row in statuses.values() if row["status"] == "incompatible"]
    if bad:
        details = "\n".join(row["task_id"] + ": " + row["detail"] for row in bad[:8])
        raise ValueError(f"{len(bad)} incompatible checkpoint(s); completed fits must not be overwritten:\n{details}")


def _code_records():
    paths = [Path(__file__), STATUS_SCRIPT, FITTER, R_DIRECTORY / "coha_v3_regression_data.R",
             DIRECTORY / "regression_inputs.py"]
    return {str(path.resolve()): _hash(path) for path in paths}


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def stage_ranges(tasks):
    """Contiguous immutable-plan ranges, with aggregated work first.

    These ranges are separate Slurm arrays chained by afterok. Merely placing
    aggregated indices earlier in one independent array is not a finish barrier.
    """
    stages = []
    cursor = 0
    previous_priority = None
    for timespan, group in itertools.groupby(tasks, key=lambda task: task["timespan"]):
        timespan = _integer(timespan, "stage timespan")
        priority = (timespan != 10000, timespan)
        if previous_priority is not None and priority <= previous_priority:
            raise ValueError("Task time stages must be contiguous and ordered: 10000, 10, 20, 50, 100")
        count = sum(1 for _ in group)
        stages.append({"timespan": timespan, "task_start": cursor,
                       "task_end": cursor + count - 1, "task_count": count})
        cursor += count
        previous_priority = priority
    return stages


def create_plan(config, plan_root=None, dry_run=False):
    rscript = Path(config["rscript"])
    if not rscript.is_file() or not os.access(rscript, os.X_OK):
        raise ValueError(f"Rscript is not executable: {rscript}")
    tasks = tasks_for(config)
    with contextlib.redirect_stdout(sys.stderr):
        snapshot = verify_snapshot(config["input_root"], _input_names(tasks))
    statuses = checkpoint_status(tasks, config["rscript"])
    _compatible(statuses)
    pending = [task for task in tasks if statuses[task["task_id"]]["status"] != "completed"]
    # Finish all aggregated work before advancing through the disjoint widths.
    # Tagged/high-vocabulary priority applies only within a time stage. Task
    # identities and all scientific model settings remain unchanged.
    pending.sort(key=lambda task: (task["timespan"] != 10000, task["timespan"],
                                  task["tagging"] != "Tagged",
                                  -task["vocab"], task["cutoff"], task["task_id"]))
    snapshot_root = Path(config["input_root"])
    identity = {"schema_version": PLAN_VERSION, "config": config,
                "snapshot_fingerprint": snapshot["fingerprint"],
                "snapshot_sha256": _hash(snapshot_root / "snapshot.json"),
                "ready_sha256": _hash(snapshot_root / "READY"),
                "code_sha256": _code_records(), "tasks": pending}
    fingerprint = _digest(identity)
    root = Path(plan_root or (Path(config["out"]) / ".batch_plans")).resolve() / fingerprint[:24]
    path = root / "plan.json"
    completed_patterns = sum(len([word for word in row["completed_patterns"].split(",") if word])
                             for row in statuses.values())
    result = {"plan_path": str(path), "task_count": len(pending), "total_cells": len(tasks),
              "completed_cells": len(tasks) - len(pending),
              "completed_patterns": completed_patterns,
              "total_patterns": sum(2 if task["tagging"] == "Tagged" else 1 for task in tasks),
              "stages": stage_ranges(pending),
              "fingerprint": fingerprint, "dry_run": dry_run}
    plan = {"fingerprint": fingerprint, "identity": identity, "summary": result}
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".plan.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["fingerprint"] != fingerprint or previous["identity"] != identity:
                    raise ValueError("Existing immutable plan differs")
            else:
                _atomic_json(path, plan)
                with (root / "tasks.tsv").open("w", newline="", encoding="utf-8") as table:
                    writer = csv.DictWriter(table, fieldnames=["task_index", *TASK_FIELDS], delimiter="\t")
                    writer.writeheader()
                    writer.writerows({"task_index": index, **task} for index, task in enumerate(pending))
    return result


def _load_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    identity = plan["identity"]
    if identity.get("schema_version") != PLAN_VERSION or _digest(identity) != plan.get("fingerprint"):
        raise ValueError("Batch plan changed or has an unsupported schema")
    tasks = identity["tasks"]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate batch tasks")
    for name, expected in identity["code_sha256"].items():
        if _hash(name) != expected:
            raise ValueError(f"Code changed after planning: {name}; create a fresh plan")
    root = Path(identity["config"]["input_root"])
    if (_hash(root / "snapshot.json") != identity["snapshot_sha256"]
            or _hash(root / "READY") != identity["ready_sha256"]):
        raise ValueError("Frozen snapshot changed after planning")
    for task in tasks:
        if task["task_id"] != _task_id(task):
            raise ValueError("Invalid task/output identity")
    return path, plan


def lock_paths(task):
    root = Path(task["out"]) / "logs"
    coarse = root / (f"local5_ctx{task['vocab']}_{task['weight']}_{task['setting']}_"
                     f"{task['frequency_family']}_{task['productivity_family']}_{task['tagging']}_"
                     f"train{task['train_variant']}_test{task['test_variant']}.lock")
    return coarse, root / ".batch_locks" / (task["task_id"] + ".lock")


@contextlib.contextmanager
def task_locks(tasks):
    paths = {}
    for task in tasks:
        coarse, cell = lock_paths(task)
        paths[coarse] = fcntl.LOCK_SH
        paths[cell] = fcntl.LOCK_EX
    with contextlib.ExitStack() as stack:
        for path, mode in sorted(paths.items(), key=lambda item: str(item[0])):
            path.parent.mkdir(parents=True, exist_ok=True)
            # NFS implements flock via fcntl: SH needs read access and EX
            # needs write access. Append-only ('a') descriptors fail with
            # EBADF for SH on NFS even though they work on a local filesystem.
            handle = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle, mode | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"An overlapping regression job is still using {path}; "
                                   "stop it and wait for it to exit before submitting the replacement") from None
        yield


@contextlib.contextmanager
def idle_locks(tasks):
    # Cell workers hold shared configuration locks, and configuration workers
    # hold exclusive locks. An EX preflight therefore detects both without opening thousands of
    # cell lock files or exhausting the usual 1024-file descriptor limit.
    paths = sorted({lock_paths(task)[0] for task in tasks})
    with contextlib.ExitStack() as stack:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"An overlapping regression job is still using {path}; "
                                   "stop it and wait for it to exit before submitting the replacement") from None
        yield


def verify_plan(path, require_idle=False):
    _, plan = _load_plan(path)
    tasks = plan["identity"]["tasks"]
    config = plan["identity"]["config"]
    if tasks:
        with contextlib.redirect_stdout(sys.stderr):
            verify_snapshot(config["input_root"], _input_names(tasks))
        statuses = checkpoint_status(tasks, config["rscript"])
        _compatible(statuses)
        if require_idle:
            with idle_locks(tasks):
                pass
    return {"plan_path": str(Path(path).resolve()), "task_count": len(tasks),
            "fingerprint": plan["fingerprint"], "verified": True, "require_idle": require_idle}


def command_for(task, config):
    flags = [("--input-root", task["input_root"]), ("--output-dir", task["out"]),
             ("--context-mode", "local5"), ("--context-size", task["vocab"]),
             ("--ppmi", task["weight"]), ("--setting", task["setting"]),
             ("--tagging", task["tagging"]), ("--timespans", task["timespan"]),
             ("--cutoffs", task["cutoff"]), ("--targets", task["targets"]),
             ("--feature-sets", task["feature_sets"]),
             ("--frequency-family", task["frequency_family"]),
             ("--productivity-family", task["productivity_family"]),
             ("--train-variant", task["train_variant"]), ("--test-variant", task["test_variant"]),
             ("--cores", 1), ("--folds", task["folds"]), ("--repeats", task["repeats"]),
             ("--alpha-count", task["alpha_count"]), ("--lambda-count", task["lambda_count"]),
             ("--permutations", task["permutations"])]
    return [config["rscript"], str(FITTER), *[str(value) for pair in flags for value in pair]]


def _fit(command):
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    process = subprocess.Popen(command, env=env, start_new_session=True)
    previous = {}

    def terminate(signum, _frame):
        if process.poll() is None:
            os.killpg(process.pid, signum)
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, terminate)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_task(path, task_index):
    if _integer(os.environ.get("SLURM_CPUS_PER_TASK", "1"), "SLURM_CPUS_PER_TASK") != 1:
        raise ValueError("Each batch task requires exactly one allocated CPU")
    plan_path, plan = _load_plan(path)
    tasks, config = plan["identity"]["tasks"], plan["identity"]["config"]
    index = _integer(task_index, "task index", 0)
    if index >= len(tasks):
        raise ValueError(f"Task index must be 0 through {len(tasks) - 1}")
    task = tasks[index]
    with task_locks([task]):
        with contextlib.redirect_stdout(sys.stderr):
            verify_snapshot(config["input_root"], _input_names([task]))
        statuses = checkpoint_status([task], config["rscript"])
        _compatible(statuses)
        reused = statuses[task["task_id"]]["status"] == "completed"
        if reused:
            print(f"Reused completed task {index}: {task['task_id']}", flush=True)
        else:
            print(f"Fitting task {index}: {task['tagging']} ctx{task['vocab']} {task['weight']} "
                  f"{task['setting']} width={task['timespan']} cutoff={task['cutoff']}; "
                  f"already completed patterns: {statuses[task['task_id']]['completed_patterns'] or 'none'}", flush=True)
            _fit(command_for(task, config))
            after = checkpoint_status([task], config["rscript"])
            _compatible(after)
            if after[task["task_id"]]["status"] != "completed":
                raise RuntimeError("R exited without complete outputs for every requested pattern: "
                                   + after[task["task_id"]]["detail"])
        receipt_root = plan_path.parent / "task_receipts"
        receipt_root.mkdir(exist_ok=True)
        _atomic_json(receipt_root / (task["task_id"] + ".json"),
                     {"plan_fingerprint": plan["fingerprint"], "task_index": index,
                      "task_id": task["task_id"], "status": "complete", "reused": reused})
    return {"task_index": index, "task_id": task["task_id"], "status": "complete", "reused": reused}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="action", required=True)
    plan = children.add_parser("plan")
    plan.add_argument("--plan-root")
    plan.add_argument("--dry-run", action="store_true")
    verify = children.add_parser("verify")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--require-idle", action="store_true")
    run = children.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--task-index", required=True, type=int)
    args = parser.parse_args(argv)
    if args.action == "plan":
        result = create_plan(configuration(), args.plan_root, args.dry_run)
    elif args.action == "verify":
        result = verify_plan(args.plan, args.require_idle)
    else:
        result = run_task(args.plan, args.task_index)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr, flush=True)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr, flush=True)
        sys.exit(2)
