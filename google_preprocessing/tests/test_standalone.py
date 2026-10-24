"""Guard the reusable preprocessing stage against downstream dependencies."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

PREPROCESSING = Path(__file__).resolve().parents[1]
ROOT = PREPROCESSING.parent


class StandalonePreprocessingTests(unittest.TestCase):
    def test_production_imports_do_not_reach_analysis(self):
        forbidden = {"compositionality_over_time", "coha_features", "coha_feature_pipeline",
                     "features", "regression_inputs", "regression_batches"}
        for path in PREPROCESSING.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [name.name for name in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    self.assertNotIn(module.split(".")[0], forbidden, path.name)

    def test_pipeline_and_annotations_import_without_scipy_or_downstream_code(self):
        code = """
import builtins, importlib, sys
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
forbidden = {'scipy', 'compositionality_over_time', 'coha_features', 'coha_feature_pipeline'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('Preprocessing attempted a downstream import')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
for name in ['google_v3_pipeline', 'google_v3_annotations']:
    importlib.import_module(name)
"""
        environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code, str(PREPROCESSING)],
            env=environment, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
