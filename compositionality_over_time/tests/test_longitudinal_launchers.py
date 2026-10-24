"""Verify longitudinal dispatch and dependencies without invoking real Slurm."""
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
                       for name in ("run_longitudinal_submit.sh", "run_longitudinal.slurm"))
LOCAL_TEMPLATE = (GOOGLE / "slurm" / "run_longitudinal.slurm").is_file()
WORKER = r'''
import json, os, sys
kind = os.path.basename(sys.argv[0])
path = os.environ["LONGITUDINAL_TEST_EVENTS"]
try:
    with open(path) as stream:
        earlier = [json.loads(line) for line in stream]
except FileNotFoundError:
    earlier = []
args = sys.argv[1:]
with open(path, "a") as stream:
    stream.write(json.dumps({"kind": kind, "args": args,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "blas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "study": {key: os.environ.get(key) for key in (
            "CONTEXT_SIZE", "CUTOFF", "TIME_STEP", "SETTING", "WEIGHTING", "TAGGING", "TEMPORAL_FEATURES",
            "LONGITUDINAL_STUDY_ID", "LONGITUDINAL_ROOT", "WORK_ROOT", "CUMULATIVE_ROOT",
            "TEMP_DIRECTORY", "RESULTS_ROOT", "LONGITUDINAL_LOG_ROOT")}}) + "\n")
if kind == "sbatch":
    if os.environ.get("LONGITUDINAL_TEST_BAD_JOB_ID"):
        print("invalid-job-id")
    else:
        print(str(700 + sum(item["kind"] == "sbatch" for item in earlier)) + ";test_data")
elif kind == "python-spy":
    print("Read-only input plan" if "plan" in args else "Mock Python stage complete")
    sys.exit(int(os.environ.get("LONGITUDINAL_TEST_PYTHON_STATUS", "0")))
elif "--check-environment" in args:
    sys.exit(int(os.environ.get("LONGITUDINAL_TEST_ENVIRONMENT_STATUS", "0")))
'''


class LongitudinalLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="google-longitudinal-launchers-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        for name in ("python-spy", "Rscript", "sbatch"):
            worker = bin_dir / name
            worker.write_text(f"#!{sys.executable}\n" + WORKER)
            worker.chmod(0o755)
        self.events_path = self.root / "events.jsonl"
        self.env = dict(os.environ)
        for name in ("MODE", "R_ENV", "CONTEXT_SIZE", "CUTOFF", "TIME_STEP", "END_DECADES",
                     "SETTING", "WEIGHTING", "TAGGING", "TEMPORAL_FEATURES", "LONGITUDINAL_STUDY_ID", "LONGITUDINAL_ROOT",
                     "FEATURE_SET", "TRAIN_VARIANT", "TEST_VARIANT", "DUCKDB_MEMORY_LIMIT",
                     "FEATURES_DUCKDB_MEMORY_LIMIT", "MAX_TEMP_DIRECTORY_SIZE", "FOLDS", "REPEATS",
                     "ALPHA_COUNT", "LAMBDA_COUNT", "PILOT", "FORCE", "PREPARE_JOB_ID", "FEATURES_JOB_ID",
                     "PREPARE_MEMORY", "PREPARE_TIME", "FEATURES_MEMORY", "FEATURES_TIME", "FIT_MEMORY",
                     "FIT_TIME", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK"):
            self.env.pop(name, None)
        self.env.update(PATH=f"{bin_dir}:{self.env['PATH']}", PROJECT_ROOT=str(GOOGLE),
                        LONGITUDINAL_SCRIPT_DIR=str(GOOGLE), PYTHON_BIN=str(bin_dir / "python-spy"),
                        RSCRIPT_BIN=str(bin_dir / "Rscript"), PREPARED_ROOT=str(self.root / "existing-prepared"),
                        MODERN_ROOT=str(self.root / "existing-modern"), WORK_ROOT=str(self.root / "work"),
                        CUMULATIVE_ROOT=str(self.root / "cumulative"), TEMP_DIRECTORY=str(self.root / "spill"),
                        RESULTS_ROOT=str(self.root / "results"), LONGITUDINAL_LOG_ROOT=str(self.root / "logs"),
                        LONGITUDINAL_TEST_EVENTS=str(self.events_path))

    def run_script(self, script="run_longitudinal.sh", action=None, options=(), **overrides):
        directory = GOOGLE / "slurm" if script.endswith(".slurm") or script.endswith("_submit.sh") else GOOGLE
        command = ["bash", str(directory / script)]
        if action is not None:
            command.append(action)
        environment = {key: value for key, value in dict(self.env, **overrides).items()
                       if value is not None}
        return subprocess.run(command + list(options), env=environment,
                              text=True, capture_output=True, timeout=30)

    def events(self):
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines()]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_plan_uses_requested_study_and_creates_no_output_directories(self):
        scripts = [name for name in ("run_longitudinal.sh", "run_longitudinal_submit.sh")
                   if ((GOOGLE / "slurm" if name.endswith("_submit.sh") else GOOGLE) / name).is_file()]
        for script in scripts:
            result = self.run_script(script)
            self.assert_success(result)
            printed = shlex.split(result.stdout.split("Frozen-model fit command: ", 1)[1])
            for flag, value in (("--context-size", "30000"), ("--setting", "Aware"), ("--ppmi", "PPMI"),
                                ("--tagging", "UnTagged"), ("--target", "compound"), ("--cutoff", "0"),
                                ("--feature-set", "all"), ("--train-variant", "modified"),
                                ("--test-variant", "modified"), ("--time-step", "10")):
                self.assertEqual(printed[printed.index(flag) + 1], value)
            self.assertTrue(printed[1].endswith("longitudinal_temporal_cutoff0.R"))
        self.assertEqual([item["kind"] for item in self.events()], ["python-spy"] * len(scripts))
        for item in self.events():
            args = item["args"]
            self.assertTrue(args[1].endswith("cumulative_temporal_cutoff0.py"))
            self.assertIn("plan", args)
            self.assertEqual(args[args.index("--threads") + 1], "1")
            self.assertEqual(args[args.index("--condition") + 1], "CompoundAware")
            self.assertEqual(args[args.index("--weight") + 1], "PPMI")
            endpoints = args[args.index("--end-decades") + 1:args.index("--condition")]
            self.assertEqual(endpoints, [str(year) for year in range(1820, 2020, 10)])
        for name in ("work", "cumulative", "spill", "results", "logs"):
            self.assertFalse((self.root / name).exists())

    def test_study_specific_default_paths_do_not_reuse_original_outputs(self):
        for script in ("run_longitudinal.sh", "run_longitudinal.slurm"):
            if not ((GOOGLE / "slurm" if script.endswith(".slurm") or script.endswith("_submit.sh") else GOOGLE) / script).is_file():
                continue
            result = self.run_script(script, MODE="plan", WORK_ROOT=None,
                                     CUMULATIVE_ROOT=None, TEMP_DIRECTORY=None,
                                     RESULTS_ROOT=None, LONGITUDINAL_LOG_ROOT=None)
            self.assert_success(result)
            event = self.events()[-1]
            study_id = "ctx30000_UnTagged_PPMI_Aware_10_0_cumulative_lag10"
            expected = {
                "--work-root": "work/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_0",
                "--temp-directory": f"work/compositionality/longitudinal_tmp/{study_id}",
                "--output-root": f"runs/compositionality/longitudinal/{study_id}/features_cumulative",
            }
            for flag, value in expected.items():
                self.assertEqual(event["args"][event["args"].index(flag) + 1], value)
            fit_command = shlex.split(result.stdout.split("Frozen-model fit command: ", 1)[1])
            self.assertEqual(fit_command[fit_command.index("--output-dir") + 1],
                             f"runs/compositionality/longitudinal/{study_id}/results")
        for name in ("work", "cumulative", "spill", "results", "logs"):
            self.assertFalse((self.root / name).exists())

    def test_setting_and_weighting_overrides_reach_both_programs(self):
        for setting in ("Aware", "Pooled", "Agnostic"):
            result = self.run_script(action="plan", SETTING=setting, WEIGHTING="RAW",
                                     CONTEXT_SIZE="5000", CUTOFF="0")
            self.assert_success(result)
            python = self.events()[-1]["args"]
            self.assertEqual(python[python.index("--condition") + 1], f"Compound{setting}")
            self.assertEqual(python[python.index("--weight") + 1], "RAW")
            self.assertEqual(python[python.index("--context-size") + 1], "5000")
            self.assertEqual(python[python.index("--cutoff") + 1], "0")
            fit_command = shlex.split(result.stdout.split("Frozen-model fit command: ", 1)[1])
            for flag, value in (("--setting", setting), ("--ppmi", "RAW"),
                                ("--context-size", "5000"), ("--cutoff", "0")):
                self.assertEqual(fit_command[fit_command.index(flag) + 1], value)
            self.assertTrue(python[1].endswith("cumulative_features.py"))
            self.assertTrue(fit_command[1].endswith("longitudinal.R"))

    def test_temporal_opt_out_preserves_static_paths_and_recipe(self):
        result = self.run_script(action="plan", TEMPORAL_FEATURES="0", WORK_ROOT=None,
                                 CUMULATIVE_ROOT=None, RESULTS_ROOT=None)
        self.assert_success(result)
        python = self.events()[-1]["args"]
        printed = shlex.split(result.stdout.split("Frozen-model fit command: ", 1)[1])
        self.assertTrue(python[1].endswith("cumulative_features.py"))
        self.assertTrue(printed[1].endswith("longitudinal.R"))
        self.assertEqual(printed[printed.index("--feature-set") + 1], "all_static")
        self.assertEqual(printed[printed.index("--output-dir") + 1],
                         "runs/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_0/results")

    def test_cutoff_500_comparison_uses_original_drivers_and_separate_paths(self):
        result = self.run_script(action="plan", CUTOFF="500", WORK_ROOT=None,
                                 CUMULATIVE_ROOT=None, RESULTS_ROOT=None)
        self.assert_success(result)
        python = self.events()[-1]["args"]
        printed = shlex.split(result.stdout.split("Frozen-model fit command: ", 1)[1])
        self.assertTrue(python[1].endswith("cumulative_temporal.py"))
        self.assertTrue(printed[1].endswith("longitudinal_temporal.R"))
        self.assertEqual(printed[printed.index("--cutoff") + 1], "500")
        self.assertEqual(printed[printed.index("--output-dir") + 1],
                         "runs/compositionality/longitudinal/ctx30000_UnTagged_PPMI_Aware_10_500_cumulative_lag10/results")
        self.assertEqual(self.events()[-1]["study"]["TEMPORAL_FEATURES"], "1")

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_cutoff_500_submission_keeps_original_environment_check(self):
        result = self.run_script("run_longitudinal_submit.sh", "fit", CUTOFF="500")
        self.assert_success(result)
        python, rscript, sbatch = self.events()
        self.assertTrue(python["args"][1].endswith("cumulative_temporal.py"))
        self.assertTrue(rscript["args"][0].endswith("longitudinal_temporal.R"))
        self.assertEqual(sbatch["study"]["CUTOFF"], "500")

    def test_temporal_scope_guard_and_feature_selection_fail_before_work(self):
        for overrides in ({"TEMPORAL_FEATURES": "yes"},
                          {"TEMPORAL_FEATURES": "1", "CONTEXT_SIZE": "5000"},
                          {"TEMPORAL_FEATURES": "1", "SETTING": "Pooled"},
                          {"TEMPORAL_FEATURES": "1", "WEIGHTING": "RAW"},
                          {"TEMPORAL_FEATURES": "1", "CUTOFF": "10"},
                          {"TEMPORAL_FEATURES": "1", "TRAIN_VARIANT": "original"},
                          {"TEMPORAL_FEATURES": "1", "TEST_VARIANT": "original"},
                          {"TEMPORAL_FEATURES": "0", "FEATURE_SET": "all"}):
            for script, action in (("run_longitudinal.sh", "plan"),
                                   ("run_longitudinal_submit.sh", "submit")):
                if not ((GOOGLE / "slurm" if script.endswith(".slurm") or script.endswith("_submit.sh") else GOOGLE) / script).is_file():
                    continue
                result = self.run_script(script, action, **overrides)
                self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.events(), [])
        for name in ("work", "cumulative", "spill", "results", "logs"):
            self.assertFalse((self.root / name).exists())

    def test_temporal_sensitivity_recipes_are_explicit(self):
        for recipe in ("all", "all_static", "representation_only", "temporal"):
            result = self.run_script(action="plan", FEATURE_SET=recipe)
            self.assert_success(result)
            printed = shlex.split(result.stdout.split("Frozen-model fit command: ", 1)[1])
            self.assertEqual(printed[printed.index("--feature-set") + 1], recipe)

    def test_invalid_study_configuration_is_rejected_before_side_effects(self):
        for overrides, expected in (({"CONTEXT_SIZE": "0"}, "CONTEXT_SIZE"),
                                    ({"CUTOFF": "-1"}, "CUTOFF"),
                                    ({"SETTING": "Unknown"}, "SETTING"),
                                    ({"WEIGHTING": "Unknown"}, "WEIGHTING"),
                                    ({"TAGGING": "Tagged"}, "TAGGING"),
                                    ({"TIME_STEP": "20"}, "TIME_STEP")):
            for script, action in (("run_longitudinal.sh", "plan"),
                                   ("run_longitudinal_submit.sh", "submit"),
                                   ("run_longitudinal.slurm", None)):
                if not ((GOOGLE / "slurm" if script.endswith(".slurm") or script.endswith("_submit.sh") else GOOGLE) / script).is_file():
                    continue
                result = self.run_script(script, action, **overrides)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(expected, result.stderr)
        self.assertEqual(self.events(), [])
        for name in ("work", "cumulative", "spill", "results", "logs"):
            self.assertFalse((self.root / name).exists())

    def test_python_stages_receive_separate_memory_limits_and_force(self):
        self.assert_success(self.run_script(action="prepare", DUCKDB_MEMORY_LIMIT="32GB"))
        self.assert_success(self.run_script(action="FEATURES", FEATURES_DUCKDB_MEMORY_LIMIT="8GB", FORCE="1"))
        prepare, features = self.events()
        self.assertEqual(prepare["args"][prepare["args"].index("--memory-limit") + 1], "32GB")
        self.assertEqual(features["args"][features["args"].index("--memory-limit") + 1], "8GB")
        self.assertNotIn("--force", prepare["args"])
        self.assertIn("--force", features["args"])
        self.assertEqual((features["omp"], features["blas"]), ("1", "1"))

    def test_checks_validate_both_inputs_without_force_or_fit(self):
        self.assert_success(self.run_script(action="check", FORCE="1"))
        python, rscript = self.events()
        self.assertIn("check", python["args"])
        self.assertIn("--check", rscript["args"])
        self.assertNotIn("--force", python["args"] + rscript["args"])

    def test_sensitivity_pilot_and_force_reach_only_requested_fit(self):
        self.assert_success(self.run_script(action="fit", FEATURE_SET="representation_only",
                                           PILOT="1", FORCE="1"))
        event, = self.events()
        self.assertEqual(event["kind"], "Rscript")
        self.assertEqual(event["args"][event["args"].index("--feature-set") + 1], "representation_only")
        self.assertIn("--pilot", event["args"])
        self.assertIn("--force", event["args"])

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_submission_checks_dependencies_then_queues_serial_stages(self):
        self.assert_success(self.run_script("run_longitudinal_submit.sh", "submit"))
        events = self.events()
        self.assertEqual([item["kind"] for item in events], ["python-spy", "Rscript", "sbatch", "sbatch", "sbatch"])
        self.assertIn("--check-environment", events[1]["args"])
        prepare, features, fit = events[2:]
        for stage, name, memory in ((prepare, "prepare", "64G"), (features, "features", "16G"), (fit, "fit", "16G")):
            self.assertIn(f"--export=ALL,MODE={name}", stage["args"])
            self.assertIn(f"--mem={memory}", stage["args"])
            self.assertIn("--cpus-per-task=1", stage["args"])
            self.assertFalse(any(value.startswith("--array") for value in stage["args"]))
            for variable, value in (("CONTEXT_SIZE", "30000"), ("CUTOFF", "0"),
                                    ("TIME_STEP", "10"), ("SETTING", "Aware"),
                                    ("WEIGHTING", "PPMI"), ("TAGGING", "UnTagged"),
                                    ("TEMPORAL_FEATURES", "1")):
                self.assertEqual(stage["study"][variable], value)
            for variable in ("WORK_ROOT", "CUMULATIVE_ROOT", "TEMP_DIRECTORY",
                             "RESULTS_ROOT", "LONGITUDINAL_LOG_ROOT"):
                self.assertEqual(stage["study"][variable], self.env[variable])
        self.assertIn("--dependency=afterok:700", features["args"])
        self.assertIn("--dependency=afterok:701", fit["args"])
        self.assertTrue(events[1]["args"][0].endswith("longitudinal_temporal_cutoff0.R"))
        self.assertTrue((self.root / "logs").is_dir())

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_submission_exports_overrides_and_custom_longitudinal_root(self):
        root = self.root / "custom-study"
        self.assert_success(self.run_script("run_longitudinal_submit.sh", "prepare",
                                           CONTEXT_SIZE="5000", CUTOFF="0", SETTING="Pooled",
                                           WEIGHTING="RAW", LONGITUDINAL_ROOT=str(root),
                                           CUMULATIVE_ROOT=None, RESULTS_ROOT=None,
                                           LONGITUDINAL_LOG_ROOT=None))
        python, sbatch = self.events()
        self.assertEqual(python["args"][python["args"].index("--output-root") + 1],
                         str(root / "features_cumulative"))
        for variable, value in (("CONTEXT_SIZE", "5000"), ("CUTOFF", "0"),
                                ("SETTING", "Pooled"), ("WEIGHTING", "RAW"),
                                ("LONGITUDINAL_ROOT", str(root)),
                                ("CUMULATIVE_ROOT", str(root / "features_cumulative")),
                                ("RESULTS_ROOT", str(root / "results")),
                                ("LONGITUDINAL_LOG_ROOT", str(root / "logs"))):
            self.assertEqual(sbatch["study"][variable], value)
        self.assertTrue((root / "logs").is_dir())

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_stage_submission_inherits_prerequisite_and_resource_overrides(self):
        self.assert_success(self.run_script("run_longitudinal_submit.sh", "features",
            options=("--partition=test_data", "--mem=24G"), PREPARE_JOB_ID="123", FEATURES_MEMORY="20G"))
        events = self.events()
        self.assertEqual([item["kind"] for item in events], ["python-spy", "sbatch"])
        args = events[-1]["args"]
        self.assertIn("--dependency=afterok:123", args)
        self.assertLess(args.index("--mem=20G"), args.index("--mem=24G"))
        self.assertIn("--partition=test_data", args)

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_missing_r_dependencies_or_failed_plan_prevent_submission(self):
        failed = self.run_script("run_longitudinal_submit.sh", "submit",
                                 LONGITUDINAL_TEST_ENVIRONMENT_STATUS="7")
        self.assertEqual(failed.returncode, 7)
        self.assertFalse(any(item["kind"] == "sbatch" for item in self.events()))
        self.assertFalse((self.root / "logs").exists())
        failed_plan = self.run_script("run_longitudinal_submit.sh", "prepare",
                                     LONGITUDINAL_TEST_PYTHON_STATUS="9")
        self.assertEqual(failed_plan.returncode, 9)
        self.assertFalse(any(item["kind"] == "sbatch" for item in self.events()))

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_invalid_array_or_dependency_is_rejected_before_input_checks(self):
        for options in (("--array=0-19",), ("--dependency=afterok:1",), ("--wrap=echo",)):
            result = self.run_script("run_longitudinal_submit.sh", "submit", options=options)
            self.assertEqual(result.returncode, 2)
        invalid = self.run_script("run_longitudinal_submit.sh", "features", PREPARE_JOB_ID="invalid")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(self.events(), [])

    @unittest.skipUnless(LOCAL_SUBMISSION, "Optional local batch helpers are not distributed")
    def test_bad_scheduler_response_stops_before_dependent_jobs(self):
        result = self.run_script("run_longitudinal_submit.sh", "submit", LONGITUDINAL_TEST_BAD_JOB_ID="1")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sum(item["kind"] == "sbatch" for item in self.events()), 1)

    @unittest.skipUnless(LOCAL_TEMPLATE, "Optional local batch helpers are not distributed")
    def test_slurm_routes_modes_and_refuses_arrays(self):
        self.assert_success(self.run_script("run_longitudinal.slurm", MODE="PLAN"))
        self.assertIn("plan", self.events()[0]["args"])
        invalid = self.run_script("run_longitudinal.slurm", MODE="features", SLURM_ARRAY_TASK_ID="0")
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(len(self.events()), 1)

    def test_fit_rejects_input_overlap_before_creating_results(self):
        for output in (self.env["MODERN_ROOT"], self.env["CUMULATIVE_ROOT"],
                       str(self.root / "existing-modern" / "nested"), str(self.root)):
            result = self.run_script(action="fit", RESULTS_ROOT=output)
            self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.events(), [])
        self.assertFalse((self.root / "existing-modern").exists())
        self.assertFalse((self.root / "cumulative").exists())

    def test_existing_fit_lock_prevents_a_second_fit_and_releases_cleanly(self):
        import fcntl
        output = self.root / "results"
        output.mkdir()
        with (output / ".fit.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            busy = self.run_script(action="fit")
            self.assertEqual(busy.returncode, 1, busy.stderr)
            self.assertEqual(self.events(), [])
        self.assert_success(self.run_script(action="fit"))
        self.assertEqual([item["kind"] for item in self.events()], ["Rscript"])
        self.assertTrue((output / ".fit.lock").is_file())


if __name__ == "__main__":
    unittest.main()
