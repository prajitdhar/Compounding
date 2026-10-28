"""Portable launcher dispatch checks; never submit jobs or invoke a real build."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT / "slurm" / "run_representations.slurm"


@unittest.skipUnless(TEMPLATE.is_file(),
                     "Optional local batch template is not distributed")
class LauncherTests(unittest.TestCase):
    def test_slurm_spool_script_finds_project_from_root_or_project_submit_directory(self):
        for submit_directory in (PROJECT.parent, PROJECT, PROJECT / "slurm"):
            with self.subTest(submit_directory=submit_directory), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                spool = temporary / "spool"
                spool.mkdir()
                script = spool / "slurm_script"
                shutil.copyfile(TEMPLATE, script)
                environment = os.environ.copy()
                for name in ("PROJECT_DIRECTORY", "LOCAL_CONFIG", "INPUT_DIRECTORY", "WORK_DIRECTORY"):
                    environment.pop(name, None)
                environment.update(DATA_ROOT=str(temporary / "inputs"), PYTHON_BIN="echo",
                    SCRATCH_ROOT=str(temporary / "work"), OUTPUT_ROOT=str(temporary / "outputs"),
                    SLURM_SUBMIT_DIR=str(submit_directory), MODE="temporal")
                result = subprocess.run(["bash", str(script)], env=environment,
                    cwd=temporary, text=True, capture_output=True, check=True)
                self.assertIn(str(PROJECT / "build_representations.py"), result.stdout)
                self.assertNotIn(str(spool / "build_representations.py"), result.stdout)
                self.assertIn("--temporal", result.stdout)

    def test_missing_input_location_fails_before_creating_work(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            for name in ("DATA_ROOT", "INPUT_DIRECTORY", "LOCAL_CONFIG"):
                environment.pop(name, None)
            work = Path(directory) / "work"
            environment.update(WORK_DIRECTORY=str(work), PYTHON_BIN="echo")
            result = subprocess.run(["bash", str(TEMPLATE)],
                env=environment, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Set DATA_ROOT or INPUT_DIRECTORY", result.stderr)
            self.assertFalse(work.exists())


if __name__ == "__main__":
    unittest.main()
