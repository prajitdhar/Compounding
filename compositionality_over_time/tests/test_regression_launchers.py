"""Exercise regression scheduling with spies; never submit real cluster jobs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


GOOGLE = Path(__file__).resolve().parents[1]
LOCAL_SUBMISSION = all((GOOGLE / "slurm" / name).is_file()
                       for name in ("run_regressions_submit.sh", "run_regressions.slurm"))
LOCAL_TEMPLATE = (GOOGLE / "slurm" / "run_regressions.slurm").is_file()
WORKER = r'''
import fcntl, json, os, sys, time
args = sys.argv[1:]
kind = os.path.basename(sys.argv[0])
with open(os.environ["REGRESSION_TEST_EVENTS"], "a+") as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    stream.seek(0)
    count = len(stream.readlines())
    stream.write(json.dumps({"kind": kind, "args": args,
                            "omp": os.environ.get("OMP_NUM_THREADS"),
                            "blas": os.environ.get("OPENBLAS_NUM_THREADS")}) + "\n")
if kind == "sbatch":
    print("bad-response" if os.environ.get("REGRESSION_TEST_BAD_JOB_ID") else 500 + count)
elif kind == "python-spy":
    sys.exit(int(os.environ.get("REGRESSION_TEST_VERIFY_STATUS", "0")))
elif "--context-size" in args and "--validate-only" not in args:
    time.sleep(float(os.environ.get("REGRESSION_TEST_DELAY", ".01")))
    if args[args.index("--context-size") + 1] == os.environ.get("REGRESSION_TEST_FAIL_VOCAB"):
        sys.exit(7)
    print("mock fit complete")
'''


class RegressionLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="google-regression-launchers-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("Rscript", "python-spy", "sbatch"):
            worker = bin_dir / name
            worker.write_text(f"#!{sys.executable}\n" + WORKER)
            worker.chmod(0o755)
        self.events_path = self.root / "events.jsonl"
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
                        SCRIPT_DIR=str(GOOGLE), OUT=str(self.root / "out"),
                        INPUT_ROOT=str(self.root / "snapshot"),
                        LOG_ROOT=str(self.root / "logs"),
                        RSCRIPT_BIN=str(bin_dir / "Rscript"),
                        PYTHON_BIN=str(bin_dir / "python-spy"),
                        REGRESSION_TEST_EVENTS=str(self.events_path))
        for variable in ("BATCH_INDEX", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "MODE",
                         "TAGGINGS", "VOCABS", "WEIGHTINGS", "SETTINGS", "TIMESPANS", "CUTOFFS",
                         "TARGETS", "JOBS", "CORES", "PILOT", "RUN_ENCODINGS", "RUN_TEMPORAL",
                         "PREPARE_JOB_ID", "FIT_JOB_ID"):
            self.env.pop(variable, None)

    def run_script(self, script, action=None, **overrides):
        directory = GOOGLE / "slurm" if script.endswith(".slurm") or script.endswith("_submit.sh") else GOOGLE
        command = ["bash", str(directory / script)]
        if action is not None:
            command.append(action)
        return subprocess.run(command, env=dict(self.env, **overrides), text=True,
                              capture_output=True, timeout=30)

    def run_grid(self, action, **overrides):
        return self.run_script("run_regressions.sh", action, **overrides)

    def events(self):
        return [json.loads(line) for line in self.events_path.read_text().splitlines()]

    def test_coha_sentence_plan_and_snapshot_verification_use_the_same_mode(self):
        options = dict(CORPUS="coha", MODES="sentence", TAGGINGS="UnTagged",
                       VOCABS="1000", WEIGHTINGS="RAW", SETTINGS="Agnostic")
        plan = self.run_grid("plan", **options)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        command = shlex.split(plan.stdout.strip())
        self.assertEqual(command[command.index("--corpus") + 1], "coha")
        self.assertEqual(command[command.index("--context-mode") + 1], "sentence")
        checked = self.run_grid("check", **options)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        verify, fit = self.events()
        self.assertEqual(verify["args"][verify["args"].index("--corpus") + 1], "coha")
        self.assertEqual(verify["args"][verify["args"].index("--context-mode") + 1], "sentence")
        self.assertEqual(fit["args"][fit["args"].index("--context-mode") + 1], "sentence")
        self.assertEqual(fit["args"][fit["args"].index("--corpus") + 1], "coha")

    def test_default_plan_is_complete_and_read_only(self):
        result = self.run_grid("plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(commands), 72)
        for args in commands:
            values = lambda flag: args[args.index(flag) + 1]
            self.assertEqual(values("--context-mode"), "local5")
            self.assertEqual(values("--timespans"), "10000,10,20,50,100")
            self.assertEqual(values("--cutoffs"), "0,10,50,100,500,1000")
            self.assertEqual(values("--cores"), "1")
            self.assertEqual(values("--train-variant"), "modified")
            self.assertEqual(values("--test-variant"), "modified")
        self.assertFalse((self.root / "out").exists())
        self.assertFalse(self.events_path.exists())

    def test_index_order_and_subset_verification(self):
        result = self.run_grid("check", BATCH_INDEX="36")
        self.assertEqual(result.returncode, 0, result.stderr)
        verify, check = self.events()
        self.assertEqual(verify["kind"], "python-spy")
        for flag, value in (("--taggings", "Tagged"), ("--vocabs", "1000"),
                            ("--weightings", "RAW"), ("--settings", "Agnostic")):
            self.assertEqual(verify["args"][verify["args"].index(flag) + 1], value)
        self.assertIn("--validate-only", check["args"])
        self.assertEqual(check["args"][check["args"].index("--tagging") + 1], "Tagged")

    def test_duplicate_grid_values_do_not_create_phantom_indices(self):
        result = self.run_grid("plan", TAGGINGS="UnTagged,UnTagged", VOCABS="1000 1000 3000",
                               WEIGHTINGS="RAW RAW", SETTINGS="Aware,Aware,Pooled")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 4)
        invalid = self.run_grid("plan", TAGGINGS="UnTagged", VOCABS="1000", WEIGHTINGS="RAW",
                                SETTINGS="Aware", BATCH_INDEX="1")
        self.assertEqual(invalid.returncode, 2)

    def test_allocation_and_google_mode_validation(self):
        for options in ({"SLURM_CPUS_PER_TASK": "2", "JOBS": "3"},
                        {"MODES": "sentence"}, {"BATCH_INDEX": "72"},
                        {"JOBS": "0"}, {"CORES": "-1"}, {"TIMESPANS": "0"},
                        {"CUTOFFS": "-1"}, {"FOLDS": "1"}, {"ALPHA_COUNT": "1"},
                        {"LAMBDA_COUNT": "1"}, {"PERMUTATIONS": "0"}):
            result = self.run_grid("plan", **options)
            self.assertEqual(result.returncode, 2, (options, result.stderr))

    def test_unverified_snapshot_stops_before_fitting(self):
        result = self.run_grid("fit", BATCH_INDEX="0", REGRESSION_TEST_VERIFY_STATUS="9")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual([item["kind"] for item in self.events()], ["python-spy"])
        self.assertFalse((self.root / "out").exists())

    def test_report_cannot_be_sliced_and_requires_pooled_period(self):
        for overrides in ({"BATCH_INDEX": "0"}, {"TIMESPANS": "10 20"}, {"CUTOFFS": "10"}):
            result = self.run_grid("report", **overrides)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.events_path.exists())

    def test_parallel_fits_preserve_parameters_and_reports_follow_them(self):
        result = self.run_grid("all", TAGGINGS="UnTagged", VOCABS="1000 3000",
                               WEIGHTINGS="RAW", SETTINGS="Agnostic Aware", JOBS="2",
                               SLURM_CPUS_PER_TASK="2", PILOT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        checks = [item for item in events if "--validate-only" in item["args"]]
        fits = [item for item in events if item["kind"] == "Rscript" and
                "--context-size" in item["args"] and "--validate-only" not in item["args"]]
        reports = [item for item in events if item["kind"] == "Rscript" and
                   "--context-size" not in item["args"]]
        self.assertEqual((len(checks), len(fits), len(reports)), (4, 4, 2))
        self.assertEqual(events[-2:], reports)
        self.assertTrue(all(item["omp"] == item["blas"] == "1" for item in events))
        self.assertTrue(all("--pilot" in item["args"] for item in fits))
        self.assertEqual(len(list((self.root / "out" / "logs").glob("run.*/*.log"))), 4)

    def test_parallel_failure_skips_report(self):
        result = self.run_grid("all", TAGGINGS="UnTagged", VOCABS="1000 3000 5000",
                               WEIGHTINGS="RAW", SETTINGS="Agnostic", JOBS="2",
                               REGRESSION_TEST_FAIL_VOCAB="1000")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertFalse(any(item["kind"] == "Rscript" and "--context-size" not in item["args"]
                             for item in self.events()))

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_submission_stages_then_arrays_then_reports_with_dependencies(self):
        result = self.run_script("run_regressions_submit.sh", "submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        submissions = self.events()
        self.assertEqual(len(submissions), 3)
        self.assertIn("--export=ALL,MODE=prepare", submissions[0]["args"])
        self.assertIn("--array=0-71%4", submissions[1]["args"])
        self.assertIn("--dependency=afterok:500", submissions[1]["args"])
        self.assertIn("--dependency=afterok:501", submissions[2]["args"])
        self.assertIn("--mem=64G", submissions[2]["args"])
        self.assertIn("--cpus-per-task=1", submissions[2]["args"])
        self.assertFalse(any(arg.startswith("--mem=") for item in submissions[:2] for arg in item["args"]))
        self.assertFalse(any("--array" in arg for arg in submissions[0]["args"] + submissions[2]["args"]))
        self.assertFalse(any("gpu" in arg.lower() for item in submissions for arg in item["args"]))

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_report_resource_overrides_and_explicit_sbatch_options(self):
        result = subprocess.run(["bash", str(GOOGLE / "slurm" / "run_regressions_submit.sh"),
                                 "report", "--mem=96G", "--cpus-per-task=2"],
                                env=dict(self.env, REPORT_MEMORY="80G", REPORT_CPUS="3", REPORT_TIME="12:00:00"),
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.events()[0]["args"]
        self.assertLess(args.index("--mem=80G"), args.index("--mem=96G"))
        self.assertLess(args.index("--cpus-per-task=3"), args.index("--cpus-per-task=2"))
        self.assertIn("--time=12:00:00", args)

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_submission_custom_grid_and_fit_dependency(self):
        result = self.run_script("run_regressions_submit.sh", "fit", TAGGINGS="Tagged",
                                 VOCABS="1000,1000,3000", WEIGHTINGS="RAW", SETTINGS="Aware Pooled",
                                 PREPARE_JOB_ID="123", FEATURE_ARRAY_CONCURRENCY="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.events()[0]["args"]
        self.assertIn("--array=0-3%2", args)
        self.assertIn("--dependency=afterok:123", args)

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_submission_rejects_unowned_arrays_and_invalid_dependency_ids(self):
        for option in ("--array=0-3", "-a0-3"):
            result = subprocess.run(["bash", str(GOOGLE / "slurm" / "run_regressions_submit.sh"),
                                     "submit", option], env=self.env, text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)
        invalid = self.run_script("run_regressions_submit.sh", "fit", PREPARE_JOB_ID="abc")
        self.assertEqual(invalid.returncode, 2, invalid.stderr)
        self.assertFalse(self.events_path.exists())

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_bad_scheduler_response_does_not_launch_dependents(self):
        result = self.run_script("run_regressions_submit.sh", "submit",
                                 REGRESSION_TEST_BAD_JOB_ID="1")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(self.events()), 1)

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_submission_rejects_unreportable_grid_before_submitting(self):
        result = self.run_script("run_regressions_submit.sh", "submit", CUTOFFS="10")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.events_path.exists())

    def test_default_pilot_is_separated_but_explicit_output_is_respected(self):
        default = self.run_grid("plan", OUT="", PILOT="1", BATCH_INDEX="0")
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn("--output-dir runs/compositionality/regressions/results_pilot", default.stdout)
        custom = self.run_grid("plan", PILOT="1", BATCH_INDEX="0")
        self.assertIn(f"--output-dir {self.root / 'out'}", custom.stdout)
        if LOCAL_TEMPLATE:
            slurm = self.run_script("run_regressions.slurm", MODE="plan", OUT="", PILOT="1",
                                    SLURM_ARRAY_TASK_ID="0")
            self.assertIn("--output-dir runs/compositionality/regressions/results_pilot", slurm.stdout)

    @unittest.skipUnless(LOCAL_TEMPLATE, "Optional local batch helpers are not distributed")
    def test_slurm_routes_array_and_refuses_array_preparation_or_report(self):
        result = self.run_script("run_regressions.slurm", MODE="plan", SLURM_ARRAY_TASK_ID="36")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertIn("--tagging Tagged", result.stdout)
        for mode in ("prepare", "report"):
            result = self.run_script("run_regressions.slurm", MODE=mode, SLURM_ARRAY_TASK_ID="0")
            self.assertEqual(result.returncode, 2, result.stderr)
        prepared = self.run_script("run_regressions.slurm", MODE="prepare")
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        args = self.events()[-1]["args"]
        self.assertIn("snapshot", args)
        self.assertIn("--features-root", args)


if __name__ == "__main__":
    unittest.main()
