"""Fine-grained regression scheduling, without Slurm or production fitting.

All generated test data are temporary. Local runner commands are inspected using
its read-only plan action; checkpoint integration uses base R and synthetic
CSV/RDS receipts, not caret models.
"""

import contextlib
import csv
import errno
import fcntl
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


GOOGLE_DIRECTORY = Path(__file__).resolve().parents[1]
if str(GOOGLE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GOOGLE_DIRECTORY))
try:
    import regression_batches as batches
except ImportError:
    batches = None


def statuses_for(tasks, status="pending", completed_patterns=""):
    return {task["task_id"]: {"task_id": task["task_id"], "status": status,
                             "patterns": "nn,an" if task["tagging"] == "Tagged" else "all",
                             "completed_patterns": completed_patterns,
                             "detail": "synthetic checkpoint status"} for task in tasks}


def flag_map(command):
    flags = command[2:]
    return dict(zip(flags[::2], flags[1::2]))


@unittest.skipIf(batches is None, "Google regression adapter dependencies are unavailable")
class RegressionBatchConfigurationTests(unittest.TestCase):
    def test_default_grid_cells_patterns_and_ordered_targets(self):
        config = batches.configuration({})
        tasks = batches.tasks_for(config)
        self.assertEqual(len(tasks), 2160)
        self.assertEqual(sum(2 if task["tagging"] == "Tagged" else 1 for task in tasks), 3240)
        self.assertEqual(len({task["task_id"] for task in tasks}), 2160)
        self.assertEqual({task["targets"] for task in tasks}, {"compound,head,modifier"})
        self.assertEqual(config["tuning"], dict(folds=5, repeats=10, alpha_count=10,
                                               lambda_count=200, permutations=5))

    def test_local_runner_shell_plan_has_identical_ordered_feature_recipes(self):
        for setting in ("Agnostic", "Aware", "Pooled"):
            requested = dict(TAGGINGS="Tagged", VOCABS="5000", WEIGHTINGS="PPMI",
                             SETTINGS=setting, TIMESPANS="50", CUTOFFS="100",
                             REFERENCE_SETTING=setting)
            config = batches.configuration(requested)
            task = batches.tasks_for(config)[0]
            env = {**os.environ, **requested, "TARGETS": "compound head modifier",
                   "INPUT_ROOT": config["input_root"], "OUT": config["out"],
                   "RSCRIPT_BIN": config["rscript"], "RUN_TEMPORAL": "1",
                   "RUN_FAMILIES": "1", "RUN_ENCODINGS": "0", "PILOT": "0", "FORCE": "0",
                   "CORES": "1", "JOBS": "1", "FOLDS": "5", "REPEATS": "10",
                   "ALPHA_COUNT": "10", "LAMBDA_COUNT": "200", "PERMUTATIONS": "5",
                   "TRAIN_VARIANT": "modified", "TEST_VARIANT": "modified"}
            env.pop("BATCH_INDEX", None)
            env.pop("SLURM_CPUS_PER_TASK", None)
            planned = subprocess.run(["bash", str(GOOGLE_DIRECTORY / "run_regressions.sh"),
                                      "plan"], env=env, capture_output=True, text=True, check=True)
            local_runner = shlex.split(planned.stdout.strip())
            current = batches.command_for(task, config)
            self.assertEqual(current[:2], local_runner[:2])
            self.assertEqual(flag_map(current), flag_map(local_runner), setting)

    def test_reference_families_encodings_and_temporal_flags_match_local_runner(self):
        config = batches.configuration(dict(TAGGINGS="Tagged", VOCABS="5000", WEIGHTINGS="PPMI",
                                            SETTINGS="Agnostic", TIMESPANS="10 10000", CUTOFFS="0 10",
                                            RUN_ENCODINGS="1"))
        tasks = batches.tasks_for(config)
        self.assertEqual(len(tasks), 10)  # Four ordinary cells + three encodings x two cutoffs.
        ordinary = [task for task in tasks if task["feature_sets"] != "all_no_setting"]
        self.assertEqual(len(ordinary), 4)
        self.assertEqual(ordinary[0]["feature_sets"].split(",")[-8:], batches.FAMILY_SETS)
        encoded = [task for task in tasks if task["feature_sets"] == "all_no_setting"]
        self.assertEqual({(task["frequency_family"], task["productivity_family"])
                          for task in encoded}, {("log_freq", "prod"), ("tf", "prod"),
                                                ("freq", "family_size")})
        self.assertEqual({task["timespan"] for task in encoded}, {10000})
        no_temporal = batches.configuration(dict(RUN_TEMPORAL="0", RUN_FAMILIES="0"))
        self.assertEqual(len(batches.tasks_for(no_temporal)), 432)
        self.assertEqual({task["timespan"] for task in batches.tasks_for(no_temporal)}, {10000})
        self.assertTrue(all("temporal" not in task["feature_sets"]
                            for task in batches.tasks_for(no_temporal)))

    def test_pilot_tuning_is_explicit_and_uses_separate_output(self):
        config = batches.configuration(dict(PILOT="1", FOLDS="7", REPEATS="2", ALPHA_COUNT="4"))
        self.assertEqual(config["tuning"], dict(folds=3, repeats=1, alpha_count=3,
                                               lambda_count=12, permutations=1))
        self.assertTrue(config["out"].endswith("results_pilot"))
        command = batches.command_for(batches.tasks_for(config)[0], config)
        self.assertNotIn("--pilot", command)
        self.assertNotIn("--force", command)
        self.assertEqual(flag_map(command)["--cores"], "1")
        self.assertEqual(flag_map(command)["--lambda-count"], "12")

    def test_words_are_normalized_without_reordering_targets(self):
        config = batches.configuration(dict(TAGGINGS="Tagged, Tagged", VOCABS="5000,5000",
                                            TARGETS="head, compound head modifier"))
        self.assertEqual(config["grid"]["TAGGINGS"], ["Tagged"])
        self.assertEqual(config["grid"]["VOCABS"], [5000])
        self.assertEqual(config["targets"], ["head", "compound", "modifier"])

    def test_invalid_environment_is_rejected(self):
        invalid = [dict(CORES="2"), dict(JOBS="2"), dict(FORCE="1"), dict(PILOT="2"),
                   dict(MODES="sentence"), dict(TAGGINGS="tagged"), dict(VOCABS="0"),
                   dict(TIMESPANS="-10"), dict(CUTOFFS="-1"), dict(WEIGHTINGS="OTHER"),
                   dict(SETTINGS="CompoundAware"), dict(TARGETS="other"), dict(FOLDS="1"),
                   dict(ALPHA_COUNT="1"), dict(LAMBDA_COUNT="1"), dict(PERMUTATIONS="0"),
                   dict(RUN_TEMPORAL="yes"), dict(RUN_ENCODINGS="2"), dict(TRAIN_VARIANT="other"),
                   dict(REFERENCE_WEIGHTING="other"), dict(VOCABS=""), dict(TARGETS="")]
        for env in invalid:
            with self.subTest(env=env), self.assertRaises(ValueError):
                batches.configuration(env)

    def test_output_identity_ignores_tuning_but_distinguishes_cells(self):
        original = batches.tasks_for(batches.configuration({}))[0]
        changed = dict(original, folds=3, targets="head,compound,modifier", feature_sets="all")
        self.assertEqual(batches._task_id(original), batches._task_id(changed))
        self.assertNotEqual(batches._task_id(original), batches._task_id(dict(original, cutoff=10)))


@unittest.skipIf(batches is None, "Google regression adapter dependencies are unavailable")
class RegressionBatchPlanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="google-batch-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "snapshot.json").write_text('{"test_data": true}\n')
        (self.snapshot / "READY").write_text("test_data-ready\n")
        self.code = self.root / "identity-code.py"
        self.code.write_text("# test data code identity\n")
        self.config = self.make_config()
        verify = mock.patch.object(batches, "verify_snapshot", return_value={"fingerprint": "test_data"})
        self.verify = verify.start()
        self.addCleanup(verify.stop)
        records = mock.patch.object(batches, "_code_records", side_effect=lambda: {
            str(self.code): batches._hash(self.code)})
        records.start()
        self.addCleanup(records.stop)

    def make_config(self, **extra):
        env = dict(INPUT_ROOT=str(self.snapshot), OUT=str(self.root / "results"),
                   RSCRIPT_BIN=sys.executable, TAGGINGS="UnTagged", VOCABS="1000",
                   WEIGHTINGS="PPMI", SETTINGS="Agnostic", TIMESPANS="10", CUTOFFS="0")
        env.update(extra)
        return batches.configuration(env)

    def create(self, statuses=None, dry_run=False, config=None):
        config = config or self.config
        with mock.patch.object(batches, "checkpoint_status", side_effect=statuses or statuses_for):
            return batches.create_plan(config, self.root / "plans", dry_run)

    def test_plan_is_immutable_reproducible_and_writes_inspectable_tsv(self):
        first = self.create()
        path = Path(first["plan_path"])
        before = path.read_bytes()
        second = self.create()
        self.assertEqual(first, second)
        self.assertEqual(path.read_bytes(), before)
        with (path.parent / "tasks.tsv").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_index"], "0")
        self.assertEqual(rows[0]["timespan"], "10")
        with mock.patch.object(batches, "checkpoint_status", side_effect=statuses_for):
            self.assertTrue(batches.verify_plan(path)["verified"])

    def test_dry_run_does_not_create_plans(self):
        result = self.create(dry_run=True)
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["stages"], [dict(timespan=10, task_start=0,
                                               task_end=0, task_count=1)])
        self.assertFalse((self.root / "plans").exists())

    def test_completed_cells_excluded_and_partial_tagged_retained(self):
        config = self.make_config(TAGGINGS="UnTagged Tagged")
        def mixed(tasks, *_):
            result = statuses_for(tasks)
            for task in tasks:
                row = result[task["task_id"]]
                if task["tagging"] == "UnTagged":
                    row.update(status="completed", completed_patterns="all")
                else:
                    row["completed_patterns"] = "nn"
            return result
        result = self.create(mixed, config=config)
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["completed_cells"], 1)
        self.assertEqual(result["completed_patterns"], 2)
        self.assertEqual(result["total_patterns"], 3)
        plan = json.loads(Path(result["plan_path"]).read_text())
        self.assertEqual(plan["identity"]["tasks"][0]["tagging"], "Tagged")

    def test_all_completed_produces_zero_task_plan(self):
        result = self.create(lambda tasks, *_: statuses_for(tasks, "completed", "all"))
        self.assertEqual(result["task_count"], 0)
        self.assertEqual(result["stages"], [])
        self.assertTrue(batches.verify_plan(result["plan_path"], require_idle=True)["verified"])

    def test_incompatible_checkpoint_refuses_plan_without_writes(self):
        with self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
            self.create(lambda tasks, *_: statuses_for(tasks, "incompatible"))
        self.assertFalse((self.root / "plans").exists())

    def test_aggregated_then_increasing_width_dominates_tagging_and_vocab_priority(self):
        config = self.make_config(TAGGINGS="UnTagged Tagged", VOCABS="1000 50000",
                                  TIMESPANS="50 10 100 10000 20", CUTOFFS="0 1000")
        result = self.create(config=config)
        tasks = json.loads(Path(result["plan_path"]).read_text())["identity"]["tasks"]
        self.assertEqual((tasks[0]["tagging"], tasks[0]["timespan"], tasks[0]["vocab"], tasks[0]["cutoff"]),
                         ("Tagged", 10000, 50000, 0))
        expected = [dict(timespan=width, task_start=index * 8,
                         task_end=index * 8 + 7, task_count=8)
                    for index, width in enumerate((10000, 10, 20, 50, 100))]
        self.assertEqual(result["stages"], expected)
        self.assertEqual(batches.stage_ranges(tasks), expected)
        for stage in expected:
            group = tasks[stage["task_start"]:stage["task_end"] + 1]
            self.assertEqual({task["timespan"] for task in group}, {stage["timespan"]})
            self.assertEqual([task["tagging"] for task in group], ["Tagged"] * 4 + ["UnTagged"] * 4)
            self.assertEqual([task["vocab"] for task in group[:4]], [50000, 50000, 1000, 1000])

    def test_completed_widths_are_omitted_and_remaining_indices_are_contiguous(self):
        config = self.make_config(TIMESPANS="10000 10 20 50")
        def completed_widths(tasks, *_):
            result = statuses_for(tasks)
            for task in tasks:
                if task["timespan"] in (10000, 10):
                    result[task["task_id"]].update(status="completed", completed_patterns="all")
            return result
        result = self.create(completed_widths, config=config)
        self.assertEqual(result["completed_cells"], 2)
        self.assertEqual(result["stages"], [dict(timespan=20, task_start=0, task_end=0, task_count=1),
                                            dict(timespan=50, task_start=1, task_end=1, task_count=1)])

    def test_stage_ranges_rejects_repeated_or_out_of_order_widths(self):
        self.assertEqual(batches.stage_ranges([]), [])
        for widths in ((10, 10000), (20, 10), (10000, 10, 10000), (10, 20, 10)):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                batches.stage_ranges([dict(timespan=width) for width in widths])

    def test_changed_code_snapshot_and_tampered_plan_are_rejected(self):
        result = self.create()
        path = Path(result["plan_path"])
        original = path.read_bytes()
        plan = json.loads(original)
        plan["identity"]["tasks"][0]["folds"] = 99
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Batch plan changed"):
            batches.verify_plan(path)
        path.write_bytes(original)
        self.code.write_text("# code changed\n")
        with self.assertRaisesRegex(ValueError, "Code changed"):
            batches.verify_plan(path)
        self.code.write_text("# test data code identity\n")
        (self.snapshot / "READY").write_text("changed-ready\n")
        with self.assertRaisesRegex(ValueError, "Frozen snapshot changed"):
            batches.verify_plan(path)

    def test_configuration_lock_blocks_cell_worker(self):
        task = batches.tasks_for(self.config)[0]
        coarse, _ = batches.lock_paths(task)
        coarse.parent.mkdir(parents=True)
        with coarse.open("a") as old:
            fcntl.flock(old, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with batches.task_locks([task]):
                    pass
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with batches.idle_locks([task]):
                    pass

    def test_disjoint_cells_share_coarse_lock_and_same_cell_is_blocked(self):
        task = batches.tasks_for(self.config)[0]
        other = dict(task, cutoff=1000)
        other["task_id"] = batches._task_id(other)
        self.assertEqual(batches.lock_paths(task)[0], batches.lock_paths(other)[0])
        with batches.task_locks([task]):
            with batches.task_locks([other]):
                pass
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with batches.task_locks([task]):
                    pass

    def test_shared_locks_have_read_access_required_by_nfs(self):
        native_flock = fcntl.flock
        seen = []

        def nfs_flock(handle, operation):
            descriptor = handle if isinstance(handle, int) else handle.fileno()
            access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
            if operation & fcntl.LOCK_SH and access == os.O_WRONLY:
                raise OSError(errno.EBADF, "NFS shared lock needs read access")
            seen.append((operation, access))
            return native_flock(handle, operation)

        task = batches.tasks_for(self.config)[0]
        other = dict(task, cutoff=1000)
        other["task_id"] = batches._task_id(other)
        coarse, _ = batches.lock_paths(task)
        coarse.parent.mkdir(parents=True)
        with coarse.open("a") as write_only:
            with self.assertRaises(OSError) as caught:
                nfs_flock(write_only, fcntl.LOCK_SH | fcntl.LOCK_NB)
            self.assertEqual(caught.exception.errno, errno.EBADF)
        with mock.patch.object(batches.fcntl, "flock", side_effect=nfs_flock):
            with batches.task_locks([task]):
                with batches.task_locks([other]):
                    pass
                with self.assertRaisesRegex(RuntimeError, "overlapping"):
                    with batches.task_locks([task]):
                        pass
                with self.assertRaisesRegex(RuntimeError, "overlapping"):
                    with batches.idle_locks([task]):
                        pass
        self.assertTrue(any(operation & fcntl.LOCK_SH for operation, _ in seen))
        self.assertTrue(all(access == os.O_RDWR for _, access in seen))

    def test_idle_preflight_detects_fine_shared_coarse_lock(self):
        task = batches.tasks_for(self.config)[0]
        with batches.task_locks([task]):
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                with batches.idle_locks([task]):
                    pass

    def test_full_grid_idle_preflight_opens_only_72_coarse_locks(self):
        config = batches.configuration(dict(OUT=str(self.root / "idle-results"),
                                            INPUT_ROOT=str(self.snapshot)))
        tasks = batches.tasks_for(config)
        with mock.patch.object(batches.fcntl, "flock", wraps=fcntl.flock) as locked:
            with batches.idle_locks(tasks):
                self.assertEqual(locked.call_count, 72)
                self.assertEqual({call.args[1] for call in locked.call_args_list},
                                 {fcntl.LOCK_EX | fcntl.LOCK_NB})
        self.assertFalse((Path(config["out"]) / "logs" / ".batch_locks").exists())

    def test_require_idle_rejects_overlapping_lock(self):
        result = self.create()
        task = batches.tasks_for(self.config)[0]
        with batches.task_locks([task]), mock.patch.object(batches, "checkpoint_status", side_effect=statuses_for):
            self.assertTrue(batches.verify_plan(result["plan_path"], require_idle=False)["verified"])
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                batches.verify_plan(result["plan_path"], require_idle=True)

    def test_finished_since_planning_skips_fit_and_writes_own_plan_receipt(self):
        result = self.create()
        with mock.patch.object(batches, "checkpoint_status", side_effect=lambda tasks, *_: statuses_for(tasks, "completed", "all")), \
                mock.patch.object(batches, "_fit") as fit, mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}):
            with contextlib.redirect_stdout(io.StringIO()):
                run = batches.run_task(result["plan_path"], 0)
        fit.assert_not_called()
        self.assertTrue(run["reused"])
        receipt = Path(result["plan_path"]).parent / "task_receipts" / (run["task_id"] + ".json")
        self.assertEqual(json.loads(receipt.read_text())["status"], "complete")

    def test_partial_tagged_runs_unchanged_fitter_recipe(self):
        config = self.make_config(TAGGINGS="Tagged")
        result = self.create(config=config)
        task = batches.tasks_for(config)[0]
        before = statuses_for([task], "pending", "nn")
        after = statuses_for([task], "completed", "nn,an")
        with mock.patch.object(batches, "checkpoint_status", side_effect=[before, after]), \
                mock.patch.object(batches, "_fit") as fit, mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}):
            with contextlib.redirect_stdout(io.StringIO()):
                run = batches.run_task(result["plan_path"], 0)
        self.assertFalse(run["reused"])
        fit.assert_called_once_with(batches.command_for(task, config))
        self.assertNotIn("--force", fit.call_args.args[0])
        self.assertNotIn("--pattern", fit.call_args.args[0])

    def test_fit_success_without_complete_outputs_is_error_and_no_receipt(self):
        result = self.create()
        with mock.patch.object(batches, "checkpoint_status", side_effect=statuses_for), \
                mock.patch.object(batches, "_fit"), mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "without complete outputs"):
                batches.run_task(result["plan_path"], 0)
        self.assertFalse((Path(result["plan_path"]).parent / "task_receipts").exists())

    def test_runtime_refuses_bad_cpu_count_index_and_incompatible_receipt(self):
        result = self.create()
        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}), self.assertRaisesRegex(ValueError, "exactly one"):
            batches.run_task(result["plan_path"], 0)
        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "1"}):
            with self.assertRaisesRegex(ValueError, "Task index"):
                batches.run_task(result["plan_path"], 1)
            with mock.patch.object(batches, "checkpoint_status", side_effect=lambda tasks, *_: statuses_for(tasks, "incompatible")), \
                    mock.patch.object(batches, "_fit") as fit, self.assertRaisesRegex(ValueError, "incompatible checkpoint"):
                batches.run_task(result["plan_path"], 0)
            fit.assert_not_called()

    def test_bad_checkpoint_status_protocol_rejected(self):
        tasks = batches.tasks_for(self.config)
        task_id = tasks[0]["task_id"]
        for stdout in ("task_id\tstatus\nwrong\tpending\n",
                       f"task_id\tstatus\n{task_id}\tunknown\n"):
            with mock.patch.object(batches.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout, "")), \
                    self.assertRaises(RuntimeError):
                batches.checkpoint_status(tasks, sys.executable)


@unittest.skipIf(batches is None, "Google regression adapter dependencies are unavailable")
class RegressionBatchRIntegrationTests(unittest.TestCase):
    def test_real_r_receipts_exact_hashes_and_partial_patterns(self):
        candidates = [Path("Rscript")]
        if shutil.which("Rscript"):
            candidates.append(Path(shutil.which("Rscript")))
        rscript = next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
        if rscript is None:
            self.skipTest("Rscript is unavailable")
        with tempfile.TemporaryDirectory(prefix="google-batch-r-test-") as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            (snapshot / "local5").mkdir(parents=True)
            (root / "results").mkdir()
            config = batches.configuration(dict(INPUT_ROOT=str(snapshot), OUT=str(root / "results"),
                                                RSCRIPT_BIN=rscript, TAGGINGS="Tagged", VOCABS="1000",
                                                WEIGHTINGS="PPMI", SETTINGS="Agnostic", TIMESPANS="10000",
                                                CUTOFFS="0"))
            tasks = batches.tasks_for(config)
            task = tasks[0]
            for name in batches._input_names(tasks):
                (snapshot / "local5" / name).write_text("synthetic input; hashing only\n")
            table = root / "tasks.tsv"
            with table.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=batches.TASK_FIELDS, delimiter="\t")
                writer.writeheader()
                writer.writerows(tasks)
            def write_receipts(patterns):
                script = f'''
source({json.dumps(str(batches.STATUS_SCRIPT))})
source({json.dumps(str(batches.R_DIRECTORY / "coha_v3_regression_data.R"))})
rows <- read.delim({json.dumps(str(table))}, colClasses="character", check.names=FALSE)
task <- google_v3_status_task(as.list(rows[1,,drop=FALSE]))
code_md5 <- unname(tools::md5sum(c({json.dumps(str(batches.FITTER))},
                                 {json.dumps(str(batches.R_DIRECTORY / "coha_v3_regression_data.R"))})))
config <- google_v3_status_config(task, unname(tools::md5sum(google_v3_status_input_paths(task))), code_md5)
for (pattern in c({','.join(json.dumps(pattern) for pattern in patterns)})) {{
  suffix <- google_v3_status_suffix(task, pattern)
  output_types <- c("rsquared","coefimp","permimp","familyimp","predictions","cvgrid","cvfolds")
  invisible(file.create(file.path(task$out, paste0(output_types,"_",suffix,".csv"))))
  saveRDS(config, file.path(task$out,paste0("complete_",suffix,".rds")))
}}
'''
                subprocess.run([rscript, "-"], input=script, capture_output=True, text=True, check=True)
            pending = batches.checkpoint_status(tasks, rscript)[task["task_id"]]
            self.assertEqual(pending["status"], "pending")
            write_receipts(["nn"])
            partial = batches.checkpoint_status(tasks, rscript)[task["task_id"]]
            self.assertEqual(partial["status"], "pending")
            self.assertEqual(partial["completed_patterns"], "nn")
            write_receipts(["an"])
            complete = batches.checkpoint_status(tasks, rscript)[task["task_id"]]
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(complete["completed_patterns"], "nn,an")
            paths = list((snapshot / "local5").iterdir())
            paths[0].write_text("changed input\n")
            stale = batches.checkpoint_status(tasks, rscript)[task["task_id"]]
            self.assertEqual(stale["status"], "incompatible")
            self.assertIn("input_md5", stale["detail"])


if __name__ == "__main__":
    unittest.main()
