"""Fine-grained launcher specification tests; sbatch and the planner are spies."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


GOOGLE = Path(__file__).resolve().parents[1]
WORKER = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
if args and args[0] == "-c":
    os.execv(sys.executable, [sys.executable, *args])
kind = pathlib.Path(sys.argv[0]).name
with open(os.environ["BATCH_TEST_EVENTS"], "a") as stream:
    stream.write(json.dumps({"kind": kind, "args": args,
        "plan": os.environ.get("BATCH_PLAN_PATH"), "out": os.environ.get("OUT"),
        "cores": os.environ.get("CORES"), "jobs": os.environ.get("JOBS"),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "blas": os.environ.get("OPENBLAS_NUM_THREADS")}) + "\n")
if kind == "sbatch":
    submissions = [json.loads(line) for line in pathlib.Path(os.environ["BATCH_TEST_EVENTS"]).read_text().splitlines()
                   if json.loads(line)["kind"] == "sbatch"]
    number = len(submissions)
    if str(number) == os.environ.get("BATCH_TEST_FAIL_SUBMISSION"):
        print("synthetic scheduler rejection", file=sys.stderr)
        sys.exit(17)
    bad = os.environ.get("BATCH_TEST_BAD_JOB_ID") or str(number) == os.environ.get("BATCH_TEST_BAD_JOB_ID_SUBMISSION")
    print("bad-response" if bad else f"{776 + number};test-cluster")
    sys.exit(0)
action = args[1]
if action == "plan":
    sys.exit(7) if os.environ.get("BATCH_TEST_PLAN_FAIL") else None
    root = pathlib.Path(args[args.index("--plan-root") + 1])
    plan = root / "test-plan" / "plan.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("{}\n")
    result = {"plan_path": str(plan.resolve()),
              "task_count": int(os.environ.get("BATCH_TEST_TASK_COUNT", "12")),
              "total_cells": 120, "completed_cells": 108}
    count = result["task_count"]
    result["completed_cells"] = result["total_cells"] - count
    result["stages"] = json.loads(os.environ["BATCH_TEST_STAGES"]) if "BATCH_TEST_STAGES" in os.environ else (
        [{"timespan": 10, "task_start": 0, "task_end": count - 1, "task_count": count}] if count else [])
    if os.environ.get("BATCH_TEST_INVALID_PLAN"):
        result["task_count"] = -1
    print(json.dumps(result))
elif action == "verify":
    sys.exit(int(os.environ.get("BATCH_TEST_IDLE_STATUS", "0")))
elif action == "run":
    sys.exit(int(os.environ.get("BATCH_TEST_RUN_STATUS", "0")))
'''


@unittest.skipUnless(
    all((GOOGLE / "slurm" / name).is_file() for name in (
        "run_regressions_batched.sh", "run_regressions_batched.slurm",
        "run_regressions_submit.sh", "run_regressions.slurm")),
    "Optional local batch helpers are not distributed")
class BatchedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="google-batched-launchers-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for name in ("python-spy", "python-preferred", "sbatch"):
            worker = self.bin_dir / name
            worker.write_text(f"#!{sys.executable}\n" + WORKER)
            worker.chmod(0o755)
        self.events_path = self.root / "events.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin_dir}:{os.environ['PATH']}",
                        SCRIPT_DIR=str(GOOGLE), OUT=str(self.root / "out"),
                        INPUT_ROOT=str(self.root / "snapshot"),
                        LOG_ROOT=str(self.root / "logs"),
                        PYTHON_BIN=str(self.bin_dir / "python-spy"),
                        BATCH_TEST_EVENTS=str(self.events_path))
        for variable in ("BATCH_INDEX", "BATCH_PLAN_PATH", "BATCH_PLAN_ROOT",
                         "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "SLURM_NTASKS",
                         "TAGGINGS", "VOCABS", "WEIGHTINGS", "SETTINGS", "TIMESPANS", "CUTOFFS",
                         "TARGETS", "JOBS", "CORES", "PILOT", "RUN_ENCODINGS", "RUN_TEMPORAL",
                         "PREPARE_JOB_ID", "FIT_JOB_ID", "FIT_ARRAY_CONCURRENCY",
                         "NNCPYTHON_BIN", "FIT_MEMORY", "FIT_TIME"):
            self.env.pop(variable, None)

    def run_helper(self, action=None, options=(), **overrides):
        command = ["bash", str(GOOGLE / "slurm" / "run_regressions_batched.sh")]
        if action is not None:
            command.append(action)
        return subprocess.run(command + list(options), env=dict(self.env, **overrides),
                              text=True, capture_output=True, timeout=20)

    def run_slurm(self, **overrides):
        return subprocess.run(["bash", str(GOOGLE / "slurm" / "run_regressions_batched.slurm")],
                              env=dict(self.env, **overrides), text=True,
                              capture_output=True, timeout=20)

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines()]

    def make_plan(self):
        plan = self.root / "plan.json"
        plan.write_text("{}\n")
        return str(plan)

    def test_default_action_plans_without_submitting_or_fitting(self):
        result = self.run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("108 already complete / 120 total; 12 remaining tasks", result.stdout)
        self.assertIn("No jobs submitted", result.stdout)
        events = self.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["args"][1], "plan")
        self.assertEqual(events[0]["args"][-1], str(self.root / "out" / ".batch_plans"))
        self.assertFalse((self.root / "logs").exists())

    def test_fit_verifies_idle_then_submits_remaining_one_cpu_tasks(self):
        result = self.run_helper("fit")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan, verify, submission = self.events()
        self.assertEqual(plan["args"][1], "plan")
        self.assertEqual(verify["args"][1], "verify")
        self.assertIn("--require-idle", verify["args"])
        args = submission["args"]
        self.assertEqual(submission["kind"], "sbatch")
        self.assertIn("--array=0-11%64", args)
        self.assertIn("--cpus-per-task=1", args)
        self.assertIn("--ntasks=1", args)
        self.assertIn("--nodes=1", args)
        self.assertIn("--mem=8G", args)
        self.assertIn("--time=3-00:00:00", args)
        self.assertIn("--export=ALL", args)
        self.assertTrue(submission["plan"].endswith("/test-plan/plan.json"))
        self.assertEqual(submission["cores"], "1")
        self.assertEqual(submission["jobs"], "1")
        self.assertTrue(args[-1].endswith("run_regressions_batched.slurm"))
        self.assertIn("777", result.stdout)
        self.assertIn("FIT_JOB_ID=777", result.stdout)

    def test_width_stages_use_success_dependencies_and_global_task_indices(self):
        stages = [dict(timespan=10000, task_start=0, task_end=3, task_count=4),
                  dict(timespan=10, task_start=4, task_end=5, task_count=2),
                  dict(timespan=20, task_start=6, task_end=11, task_count=6)]
        result = self.run_helper("fit", BATCH_TEST_STAGES=json.dumps(stages))
        self.assertEqual(result.returncode, 0, result.stderr)
        plan, verify, *submissions = self.events()
        self.assertEqual(plan["args"][1], "plan")
        self.assertEqual(verify["args"][1], "verify")
        self.assertEqual(len(submissions), 3)
        for event, stage in zip(submissions, stages):
            self.assertEqual(event["kind"], "sbatch")
            self.assertIn(f'--array={stage["task_start"]}-{stage["task_end"]}%64', event["args"])
            self.assertEqual(event["plan"], submissions[0]["plan"])
        self.assertFalse(any(arg.startswith("--dependency") for arg in submissions[0]["args"]))
        self.assertIn("--dependency=afterok:777", submissions[1]["args"])
        self.assertIn("--dependency=afterok:778", submissions[2]["args"])
        self.assertIn("FIT_JOB_ID=779", result.stdout)
        for job_id in ("777", "778", "779"):
            self.assertIn(job_id, result.stdout)

    def test_completed_stages_absent_from_plan_do_not_get_submitted(self):
        stages = [dict(timespan=20, task_start=0, task_end=11, task_count=12)]
        result = self.run_helper("fit", BATCH_TEST_STAGES=json.dumps(stages))
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = [event for event in self.events() if event["kind"] == "sbatch"]
        self.assertEqual(len(submissions), 1)
        self.assertIn("--array=0-11%64", submissions[0]["args"])
        self.assertFalse(any(arg.startswith("--dependency") for arg in submissions[0]["args"]))

    def test_custom_concurrency_resources_and_plan_location(self):
        result = self.run_helper("fit", options=("--mem=16G", "--time=12:00:00", "--partition=normal"),
                                 FIT_ARRAY_CONCURRENCY="32", FIT_MEMORY="12G", FIT_TIME="1-00:00:00",
                                 BATCH_PLAN_ROOT=str(self.root / "custom plans"))
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual(events[0]["args"][-1], str(self.root / "custom plans"))
        args = events[-1]["args"]
        self.assertIn("--array=0-11%32", args)
        self.assertLess(args.index("--mem=12G"), args.index("--mem=16G"))
        self.assertLess(args.index("--time=1-00:00:00"), args.index("--time=12:00:00"))

    def test_completed_grid_does_not_submit_empty_array(self):
        result = self.run_helper("fit", BATCH_TEST_TASK_COUNT="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No remaining work", result.stdout)
        self.assertEqual(len(self.events()), 1)

    def test_busy_existing_jobs_stop_before_submission(self):
        result = self.run_helper("fit", BATCH_TEST_IDLE_STATUS="9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual([event["args"][1] for event in self.events()], ["plan", "verify"])
        self.assertFalse((self.root / "logs").exists())

    def test_planner_failure_does_not_submit(self):
        result = self.run_helper("fit", BATCH_TEST_PLAN_FAIL="1")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(self.events()), 1)

    def test_invalid_planner_counts_do_not_submit(self):
        result = self.run_helper("fit", BATCH_TEST_INVALID_PLAN="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid configuration counts", result.stderr)
        self.assertEqual(len(self.events()), 1)

    def test_bad_scheduler_response_is_reported(self):
        result = self.run_helper("fit", BATCH_TEST_BAD_JOB_ID="1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Unexpected sbatch job ID", result.stderr)
        self.assertEqual(len(self.events()), 3)

    def test_malformed_stages_rejected_before_verify_or_submission(self):
        good = dict(timespan=10000, task_start=0, task_end=11, task_count=12)
        invalid = [None, {}, [], [dict(good, timespan="10000")], [dict(good, timespan=True)],
                   [dict(good, task_start=True)], [dict(good, task_count="12")],
                   [dict(good, timespan=0)], [dict(good, task_start=1)],
                   [dict(good, task_end=12)], [dict(good, task_end=-1)],
                   [dict(good, task_count=11)], [dict(good, task_count=0)],
                   [dict(good, task_start=-1)],
                   [dict(timespan=10, task_start=0, task_end=5, task_count=6),
                    dict(timespan=10000, task_start=6, task_end=11, task_count=6)],
                   [dict(timespan=10000, task_start=0, task_end=5, task_count=6),
                    dict(timespan=10000, task_start=6, task_end=11, task_count=6)],
                   [dict(timespan=10000, task_start=0, task_end=5, task_count=6),
                    dict(timespan=20, task_start=5, task_end=10, task_count=6)],
                   [dict(timespan=10000, task_start=0, task_end=4, task_count=5),
                    dict(timespan=20, task_start=6, task_end=11, task_count=6)]]
        for stages in invalid:
            with self.subTest(stages=stages):
                before = len(self.events())
                result = self.run_helper("fit", BATCH_TEST_STAGES=json.dumps(stages))
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertEqual([event["args"][1] for event in self.events()[before:]], ["plan"])

    def test_first_submission_failure_stops_without_later_stage(self):
        stages = [dict(timespan=10000, task_start=0, task_end=5, task_count=6),
                  dict(timespan=20, task_start=6, task_end=11, task_count=6)]
        result = self.run_helper("fit", BATCH_TEST_STAGES=json.dumps(stages), BATCH_TEST_FAIL_SUBMISSION="1")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([event for event in self.events() if event["kind"] == "sbatch"]), 1)

    def test_later_submission_failure_reports_accepted_jobs_and_stops(self):
        stages = [dict(timespan=10000, task_start=0, task_end=3, task_count=4),
                  dict(timespan=10, task_start=4, task_end=7, task_count=4),
                  dict(timespan=20, task_start=8, task_end=11, task_count=4)]
        result = self.run_helper("fit", BATCH_TEST_STAGES=json.dumps(stages), BATCH_TEST_FAIL_SUBMISSION="2")
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len([event for event in self.events() if event["kind"] == "sbatch"]), 2)
        self.assertIn("777", result.stdout + result.stderr)

    def test_later_bad_job_id_reports_accepted_jobs_and_stops(self):
        stages = [dict(timespan=10000, task_start=0, task_end=3, task_count=4),
                  dict(timespan=10, task_start=4, task_end=7, task_count=4),
                  dict(timespan=20, task_start=8, task_end=11, task_count=4)]
        result = self.run_helper("fit", BATCH_TEST_STAGES=json.dumps(stages),
                                 BATCH_TEST_BAD_JOB_ID_SUBMISSION="2")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len([event for event in self.events() if event["kind"] == "sbatch"]), 2)
        self.assertIn("Unexpected sbatch job ID", result.stderr)
        self.assertIn("777", result.stdout + result.stderr)

    def test_invalid_parallelism_and_stale_batch_selection_rejected(self):
        for overrides in ({"CORES": "2"}, {"JOBS": "2"}, {"FIT_ARRAY_CONCURRENCY": "65"},
                          {"FIT_ARRAY_CONCURRENCY": "0"}, {"FIT_ARRAY_CONCURRENCY": "-1"},
                          {"BATCH_INDEX": "0"}, {"BATCH_PLAN_PATH": "/tmp/stale-plan.json"},
                          {"PREPARE_JOB_ID": "123"}):
            result = self.run_helper("fit", **overrides)
            self.assertEqual(result.returncode, 2, (overrides, result.stderr))
        self.assertEqual(self.events(), [])

    def test_user_array_and_multicore_overrides_rejected(self):
        for options in (("--array=0-3",), ("-a0-3",), ("--cpus-per-task=2",),
                        ("--cpus-per-task", "2"), ("-c2",), ("-c", "2"),
                        ("--ntasks=2",), ("-n2",), ("--nodes=2",),
                        ("--ntasks-per-node=2",), ("--wrap=echo hi",)):
            result = self.run_helper("fit", options=options)
            self.assertEqual(result.returncode, 2, (options, result.stderr))
        self.assertEqual(self.events(), [])

    def test_user_dependencies_cannot_override_stage_chain(self):
        for options in (("--dependency=afterany:123",), ("--dependency", "afterok:123"),
                        ("-dafterok:123",), ("-d", "afterok:123")):
            with self.subTest(options=options):
                result = self.run_helper("fit", options=options)
                self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.events(), [])

    def test_explicit_single_cpu_options_are_accepted(self):
        result = self.run_helper("fit", options=("--cpus-per-task", "1", "-n1", "--nodes=1"))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preferred_python_override(self):
        result = self.run_helper("plan", NNCPYTHON_BIN=str(self.bin_dir / "python-preferred"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events()[0]["kind"], "python-preferred")

    def test_pilot_defaults_separate_output_directory(self):
        result = self.run_helper("plan", OUT="", PILOT="1", BATCH_PLAN_ROOT=str(self.root / "pilot-plans"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.events()[0]["out"], "runs/compositionality/regressions/results_pilot")

    def test_report_delegates_existing_dependency_and_complete_grid_launcher(self):
        result = self.run_helper("report", FIT_JOB_ID="12345")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual(len(events), 1)
        args = events[0]["args"]
        self.assertIn("--dependency=afterok:12345", args)
        self.assertIn("--export=ALL,MODE=report", args)
        self.assertIn("--mem=64G", args)
        self.assertTrue(args[-1].endswith("run_regressions.slurm"))
        self.assertFalse(any(arg.startswith("--array") for arg in args))

    def test_slurm_executes_exact_immutable_task_and_limits_threads(self):
        plan = self.make_plan()
        result = self.run_slurm(BATCH_PLAN_PATH=plan, SLURM_ARRAY_TASK_ID="7", SLURM_CPUS_PER_TASK="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertEqual(len(events), 1)
        args = events[0]["args"]
        self.assertEqual(args[1:], ["run", "--plan", plan, "--task-index", "7"])
        self.assertEqual(events[0]["omp"], "1")
        self.assertEqual(events[0]["blas"], "1")

    def test_slurm_rejects_missing_plan_and_invalid_allocation(self):
        plan = self.make_plan()
        for overrides in ({}, {"BATCH_PLAN_PATH": "relative.json", "SLURM_ARRAY_TASK_ID": "0"},
                          {"BATCH_PLAN_PATH": plan},
                          {"BATCH_PLAN_PATH": plan, "SLURM_ARRAY_TASK_ID": "-1"},
                          {"BATCH_PLAN_PATH": plan, "SLURM_ARRAY_TASK_ID": "0", "SLURM_CPUS_PER_TASK": "2"},
                          {"BATCH_PLAN_PATH": plan, "SLURM_ARRAY_TASK_ID": "0", "SLURM_NTASKS": "2"},
                          {"BATCH_PLAN_PATH": plan, "SLURM_ARRAY_TASK_ID": "0", "CORES": "2"}):
            result = self.run_slurm(**overrides)
            self.assertEqual(result.returncode, 2, (overrides, result.stderr))
        self.assertEqual(self.events(), [])

    def test_slurm_propagates_runner_failure(self):
        result = self.run_slurm(BATCH_PLAN_PATH=self.make_plan(), SLURM_ARRAY_TASK_ID="0",
                                BATCH_TEST_RUN_STATUS="17")
        self.assertEqual(result.returncode, 17, result.stderr)


if __name__ == "__main__":
    unittest.main()
