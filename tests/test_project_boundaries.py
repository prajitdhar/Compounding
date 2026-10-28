"""Structural specifications for independent preprocessing and analysis projects."""

import ast
import importlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "src"
PREPROCESSING = ROOT / "google_preprocessing"
COHA = ROOT / "coha_preprocessing"
ANALYSIS = ROOT / "compositionality_over_time"
NOVEL = ROOT / "novel_compound_predictor"


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class ProjectBoundaryTests(unittest.TestCase):
    def test_shared_library_is_a_flat_importable_package(self):
        self.assertFalse((SHARED / "compounding").exists())
        self.assertEqual([p.name for p in SHARED.iterdir()
                          if p.is_dir() and p.name != "__pycache__"], [])
        for name in ["io", "progress", "source_info", "partitioned_parquet",
                     "count_store", "context_vocabulary"]:
            module = importlib.import_module(f"src.{name}")
            self.assertEqual(Path(module.__file__).resolve(), SHARED / f"{name}.py")
        self.assertEqual(importlib.import_module("src.io").atomic_path.__module__, "src.io")

    def test_count_storage_has_one_shared_implementation(self):
        module = importlib.import_module("src.count_store")
        self.assertEqual(module.CountStore.__module__, "src.count_store")
        self.assertFalse((PREPROCESSING / "count_store.py").exists())
        self.assertFalse((COHA / "count_store.py").exists())
        self.assertIn("src.count_store", set(imported_modules(COHA / "coha_pipeline.py")))

    def test_installer_names_the_analysis_and_preprocessing_extras(self):
        configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(configuration, r"(?m)^preprocessing\s*=\s*\[")
        self.assertRegex(configuration, r"(?m)^coha\s*=\s*\[")
        self.assertRegex(configuration, r"(?m)^compositionality\s*=\s*\[")
        self.assertNotRegex(configuration, r"(?m)^google\s*=")
        self.assertIn('[tool.setuptools]', configuration)
        self.assertRegex(configuration, r'(?m)^packages\s*=\s*\["src"\]')
        self.assertNotIn("[tool.setuptools.packages.find]", configuration)

    def test_projects_have_separate_source_boundaries(self):
        for project in [PREPROCESSING, COHA, ANALYSIS, NOVEL]:
            with self.subTest(project=project.name):
                self.assertTrue((project / "README.md").is_file())
        self.assertTrue((PREPROCESSING / "google_v3_pipeline.py").is_file())
        self.assertTrue((COHA / "coha_pipeline.py").is_file())
        self.assertTrue((COHA / "vocabulary.py").is_file())
        self.assertTrue((ANALYSIS / "features.py").is_file())
        self.assertTrue((ANALYSIS / "regression").is_dir())
        self.assertFalse((ANALYSIS / "google").exists())
        self.assertFalse((ANALYSIS / "regression" / "paper").exists())
        self.assertFalse((PREPROCESSING / "regression").exists())
        self.assertFalse((PREPROCESSING / "features.py").exists())
        self.assertFalse((ANALYSIS / "coha_preprocessing").exists())
        self.assertFalse((COHA / "regression").exists())
        self.assertFalse((COHA / "features.py").exists())

    def test_preprocessing_does_not_import_analysis_or_prediction(self):
        forbidden = {"compositionality_over_time", "novel_compound_predictor",
                     "coha_features", "coha_feature_pipeline", "feature_math",
                     "features", "regression_inputs", "regression_batches"}
        for project in [PREPROCESSING, COHA]:
            for path in project.glob("*.py"):
                with self.subTest(module=str(path.relative_to(ROOT))):
                    self.assertFalse({name.split(".")[0] for name in imported_modules(path)} & forbidden)

    def test_corpus_preprocessors_are_independent(self):
        for project, forbidden in [
            (PREPROCESSING, {"coha_preprocessing", "coha_pipeline"}),
            (COHA, {"google_preprocessing", "google_v3_pipeline", "google_v3_annotations"}),
        ]:
            for path in project.glob("*.py"):
                with self.subTest(module=str(path.relative_to(ROOT))):
                    self.assertFalse({name.split(".")[0] for name in imported_modules(path)} & forbidden)

    def test_analysis_and_prediction_do_not_import_raw_preprocessing(self):
        forbidden = {"google_preprocessing", "google_v3_pipeline", "google_v3_annotations",
                     "coha_preprocessing", "coha_pipeline", "count_store"}
        paths = list(ANALYSIS.glob("*.py")) + list((ANALYSIS / "feature_helpers").glob("*.py"))
        paths += list(NOVEL.glob("*.py"))
        for path in paths:
            with self.subTest(module=str(path.relative_to(ROOT))):
                self.assertFalse({name.split(".")[0] for name in imported_modules(path)} & forbidden)

    def test_documentation_identifies_the_extracted_table_handoff(self):
        analysis_docs = "\n".join(path.read_text(encoding="utf-8")
                                   for path in [ANALYSIS / "README.md", ANALYSIS / "FEATURES.md"])
        self.assertIn("google_preprocessing", analysis_docs)
        self.assertIn("coha_preprocessing", analysis_docs)
        self.assertIn("feature_inputs/sentence/", analysis_docs)
        self.assertIn("bigram_counts.parquet", analysis_docs)
        novel_docs = (NOVEL / "README.md").read_text(encoding="utf-8")
        self.assertIn("google_preprocessing", novel_docs)
        self.assertNotIn("../compositionality_over_time/google", novel_docs)

    def test_batch_templates_are_ignored_and_public_docs_use_local_commands(self):
        patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("*.slurm", patterns)
        platform_reference = re.compile(r"\b(?:slurm|sbatch|squeue|sacct)\b|\.slurm\b", re.IGNORECASE)
        for path in ROOT.rglob("*.md"):
            if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.relative_to(ROOT).parts):
                continue
            with self.subTest(document=str(path.relative_to(ROOT))):
                self.assertIsNone(platform_reference.search(path.read_text(encoding="utf-8")),
                                  "Public documentation must use distributed command-line programs")

    def test_local_batch_assets_have_project_specific_folders(self):
        patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("**/slurm/", patterns)
        for project in [PREPROCESSING, COHA, ANALYSIS, NOVEL]:
            for path in project.rglob("*.slurm"):
                with self.subTest(template=str(path.relative_to(ROOT))):
                    self.assertEqual(path.parent, project / "slurm")
        for project, names in [(PREPROCESSING, ["run_google_v3_base.sh", "run_google_v3_ner.sh"]),
                               (ANALYSIS, ["run_features.sh", "run_regressions_submit.sh",
                                           "run_regressions_batched.sh", "run_longitudinal_submit.sh"])]:
            for name in names:
                self.assertFalse((project / name).exists())

    def test_runtime_code_uses_the_corpus_neutral_manifest_name(self):
        old_name = "coha_" + "manifest.json"
        paths = [*SHARED.glob("*.py")]
        for project in [PREPROCESSING, COHA, ANALYSIS, NOVEL]:
            paths.extend(path for path in project.rglob("*.py")
                         if "tests" not in path.relative_to(project).parts)
        for path in paths:
            with self.subTest(source=str(path.relative_to(ROOT))):
                self.assertNotIn(old_name, path.read_text(encoding="utf-8"))
        self.assertIn("corpus_manifest.json", (PREPROCESSING / "google_v3_pipeline.py").read_text())
        self.assertIn("corpus_manifest.json", (COHA / "coha_pipeline.py").read_text())
        self.assertIn("corpus_manifest.json", (ANALYSIS / "features.py").read_text())

    def test_current_workflow_has_supported_entrypoints_and_guides(self):
        supported = [PREPROCESSING / "google_v3_pipeline.py",
                     COHA / "coha_pipeline.py", COHA / "vocabulary.py",
                     COHA / "SOURCE_INFO.md", COHA / "examples" / "README.md",
                     ANALYSIS / "features.py", ANALYSIS / "regression_inputs.py",
                     ANALYSIS / "cumulative_features.py", NOVEL / "build_representations.py",
                     NOVEL / "predict_novel_compounds.py", ROOT / "TESTING.md",
                     NOVEL / "WORKFLOW.md"]
        for path in supported:
            with self.subTest(entrypoint=str(path.relative_to(ROOT))):
                self.assertTrue(path.is_file())

    def test_user_guides_describe_the_current_workflow_without_repository_history(self):
        origin_story = re.compile(r"\blegacy\b|pre-cleanup|historical reconstruction|"
                                  r"deleted historical|older workflow|recovered.{0,20}coordinates", re.IGNORECASE)
        for path in ROOT.rglob("*.md"):
            if any(part in {".git", ".venv", "venv", "__pycache__", "slurm"}
                   for part in path.relative_to(ROOT).parts):
                continue
            with self.subTest(document=str(path.relative_to(ROOT))):
                self.assertIsNone(origin_story.search(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
