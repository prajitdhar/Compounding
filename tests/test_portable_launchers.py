"""Run copied Slurm templates with spies, never jobs, corpora or model code."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[1]
COMPOSITIONALITY = REPOSITORY / "compositionality_over_time"
PREPROCESSING = REPOSITORY / "google_preprocessing"
NOVEL = REPOSITORY / "novel_compound_predictor"
LOCAL_TEMPLATES = (
    COMPOSITIONALITY / "slurm" / "run_features.slurm",
    COMPOSITIONALITY / "slurm" / "run_longitudinal.slurm",
    PREPROCESSING / "slurm" / "run_google_v3_ner.slurm",
    NOVEL / "slurm" / "run_representations.slurm",
)


@unittest.skipUnless(all(path.is_file() for path in LOCAL_TEMPLATES),
                     "Optional local batch templates are not distributed")
@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Local batch templates require Bash")
class PortableLauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="portable-launcher-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.spool = self.root / "spool"
        self.working = self.root / "unrelated-working-directory"
        self.binary = self.root / "bin"
        for directory in [self.spool, self.working, self.binary]:
            directory.mkdir()
        self.trace = self.root / "interpreter-calls.jsonl"
        self.scheduler_trace = self.root / "scheduler-must-not-run"
        self.interpreter = self.binary / "interpreter-spy"
        self.interpreter.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['LAUNCH_TRACE'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        self.interpreter.chmod(0o755)
        scheduler = self.binary / "sbatch"
        scheduler.write_text(
            "#!/bin/sh\n"
            ": > \"$SCHEDULER_TRACE\"\n"
            "echo 'A portability test must never submit a job' >&2\n"
            "exit 99\n")
        scheduler.chmod(0o755)

    def environment(self, submit_directory, **overrides):
        # Do not inherit scheduler state, project overrides, local configuration
        # or user-supplied startup files from the invoking research environment.
        environment = {
            "PATH": str(self.binary) + os.pathsep + os.defpath,
            "LANG": "C.UTF-8", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHON_BIN": str(self.interpreter), "RSCRIPT_BIN": str(self.interpreter),
            "LAUNCH_TRACE": str(self.trace), "SCHEDULER_TRACE": str(self.scheduler_trace),
            "SLURM_SUBMIT_DIR": str(submit_directory), "SLURM_CPUS_PER_TASK": "1",
            "MODE": "prepare", "NER_DEVICE": "cpu", "TARGET_TAGGED": "0",
            "DATA_ROOT": str(self.root / "inputs"),
            "FEATURE_INPUT": str(self.root / "inputs"),
            "FEATURE_OUTPUT": str(self.root / "feature-output"),
            "RATINGS_DIR": str(self.root / "ratings"),
            "CONTEXT_VOCABULARY": str(self.root / "contexts.tsv"),
            "CACHE_DIRECTORY": str(self.root / "cache"),
            "TEMP_DIRECTORY": str(self.root / "temporary-work"),
            "OUTPUT_ROOT": str(self.root / "output"),
            "WORK_DIRECTORY": str(self.root / "scratch-work"),
            "PREPARED_ROOT": str(self.root / "prepared"),
            "MODERN_ROOT": str(self.root / "modern"),
            "LONGITUDINAL_ROOT": str(self.root / "longitudinal"),
            "WORK_ROOT": str(self.root / "cumulative-work"),
            "CUMULATIVE_ROOT": str(self.root / "cumulative-features"),
            "RESULTS_ROOT": str(self.root / "growth-results"),
            "LONGITUDINAL_LOG_ROOT": str(self.root / "growth-logs"),
        }
        environment.update(overrides)
        for forbidden in ["PROJECT_ROOT", "SCRIPT_DIR", "PROJECT_DIRECTORY", "COMPOUNDING_CONFIG", "LOCAL_CONFIG"]:
            self.assertNotIn(forbidden, environment)
        return environment

    def run_spool(self, source, submit_directory, **overrides):
        copied = self.spool / "slurm-script"
        shutil.copyfile(source, copied)
        self.trace.unlink(missing_ok=True)
        result = subprocess.run(["bash", str(copied)], cwd=self.working,
            env=self.environment(submit_directory, **overrides), capture_output=True,
            text=True, timeout=15)
        self.assertFalse(self.scheduler_trace.exists(), "A copied job template invoked sbatch")
        calls = ([json.loads(line) for line in self.trace.read_text().splitlines()]
                 if self.trace.exists() else [])
        return result, calls

    def assert_interpreter_call(self, source, submit_directory, expected_program, action, **overrides):
        result, calls = self.run_spool(source, submit_directory, **overrides)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 1, calls)
        arguments = calls[0]
        self.assertEqual(arguments[:3], ["-u", str(expected_program), action])
        self.assertNotIn(str(self.spool / "slurm-script"), arguments)
        self.assertNotIn(str(self.working), arguments)
        return arguments

    def test_analysis_features_resolve_checkout_from_submission_directories(self):
        for submit_directory in [REPOSITORY, COMPOSITIONALITY, COMPOSITIONALITY / "slurm"]:
            with self.subTest(submit_directory=submit_directory.name):
                arguments = self.assert_interpreter_call(
                    COMPOSITIONALITY / "slurm" / "run_features.slurm", submit_directory,
                    COMPOSITIONALITY / "features.py", "prepare")
                self.assertIn("--untagged", arguments)
                self.assertEqual(arguments[arguments.index("--inputdir") + 1], str(self.root / "inputs"))

    def test_google_ner_resolves_checkout_without_site_or_gpu_configuration(self):
        for submit_directory in [REPOSITORY, PREPROCESSING, PREPROCESSING / "slurm"]:
            with self.subTest(submit_directory=submit_directory.name):
                arguments = self.assert_interpreter_call(
                    PREPROCESSING / "slurm" / "run_google_v3_ner.slurm", submit_directory,
                    PREPROCESSING / "google_v3_annotations.py", "prepare")
                self.assertEqual(arguments[arguments.index("--device") + 1], "cpu")

    def test_longitudinal_spool_finds_wrapper_configuration_and_scoped_program(self):
        for submit_directory in [REPOSITORY, COMPOSITIONALITY, COMPOSITIONALITY / "slurm"]:
            for cutoff, program in [("0", "cumulative_temporal_cutoff0.py"),
                                    ("500", "cumulative_temporal.py")]:
                with self.subTest(submit_directory=submit_directory.name, cutoff=cutoff):
                    arguments = self.assert_interpreter_call(
                        COMPOSITIONALITY / "slurm" / "run_longitudinal.slurm", submit_directory,
                        COMPOSITIONALITY / program, "prepare", CUTOFF=cutoff)
                    self.assertEqual(arguments[arguments.index("--cutoff") + 1], cutoff)
                    self.assertEqual(arguments[arguments.index("--prepared-root") + 1],
                                     str(self.root / "prepared"))

    def test_novel_spool_resolves_repository_and_project_submission_directories(self):
        for submit_directory in [REPOSITORY, NOVEL, NOVEL / "slurm"]:
            for mode in ["pooled", "temporal"]:
                with self.subTest(submit_directory=submit_directory.name, mode=mode):
                    result, calls = self.run_spool(NOVEL / "slurm" / "run_representations.slurm",
                                                   submit_directory, MODE=mode)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(len(calls), 1, calls)
                    self.assertEqual(calls[0][:2], ["-u", str(NOVEL / "build_representations.py")])
                    self.assertEqual("--temporal" in calls[0], mode == "temporal")
                    self.assertEqual(calls[0][calls[0].index("--pair-contexts") + 1],
                                     str(self.root / "inputs" / "bigrams.parquet"))

    def test_unknown_submission_directory_cannot_fall_back_to_unrelated_code(self):
        for source in [COMPOSITIONALITY / "slurm" / "run_features.slurm",
                       PREPROCESSING / "slurm" / "run_google_v3_ner.slurm",
                       NOVEL / "slurm" / "run_representations.slurm"]:
            with self.subTest(source=source.name):
                mode = "pooled" if source.parent.parent == NOVEL else "prepare"
                result, calls = self.run_spool(source, self.working, MODE=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
