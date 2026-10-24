"""Conservative public-tree checks; not a comprehensive security certification.

The scanner reports relative filenames and issue categories, never matched
values. Git history, private archives, and external data are outside its scope.
"""

import os
from pathlib import Path
import re
import tempfile
import unittest

PUBLIC_ROOT = Path(__file__).resolve().parents[2]
MAX_PUBLIC_BYTES = 5 * 1024 * 1024
SKIP_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv"}
BLOCKED_SUFFIXES = {".ipynb", ".pkl", ".pickle", ".rds", ".rdata", ".pt", ".pth",
                    ".rda", ".npz", ".npy", ".zip", ".gz", ".xz", ".7z", ".tar",
                    ".out", ".err"}
SAMPLE_ONLY_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".png", ".pdf", ".log"}
BLOCKED_ROOT_NAMES = {".codex", ".vscode", "Compounding", "temp.md", "latest.md", "coha.zip",
                      "social-media-data"}
TEXT_SUFFIXES = {".py", ".r", ".md", ".txt", ".json", ".toml", ".cfg", ".ini",
                 ".sh", ".slurm", ".yml", ".yaml", ".csv", ".tsv", ".example", ".log"}
PERSONAL_PATH = re.compile(r"/(?:home|Users)/[^/\s\"']+|/mnt/(?:[^/\s]+/)*home/[^/\s\"']+|"
                           r"/scratch(?:[_-][^/\s\"']+)?/[^/\s\"']+|"
                           r"/mount/[^/\s\"']+/users/[^/\s\"']+")
EMAIL = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
INFRASTRUCTURE = re.compile(
    r"^\s*#SBATCH\s+(?:--nodelist|-w|--account|-A)(?:=|\s+)(?!\$|YOUR_|<)[^\s]+|"
    r"\b(?:login|compute|gpu|memory)[_-]?node[0-9]{1,3}\b",
    re.IGNORECASE | re.MULTILINE)
CREDENTIAL = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b|"
                        r"\bgithub_pat_[A-Za-z0-9_]{30,}\b|\bhf_[A-Za-z0-9]{30,}\b|"
                        r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b|"
                        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def scan_public_tree(root):
    """Return (relative filename, category) issues without disclosing values."""
    root = Path(root)
    issues = []
    for directory, subdirectories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in sorted(subdirectories):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                issues.append((str(relative), "symlink"))
            if relative.parts[0] in BLOCKED_ROOT_NAMES:
                issues.append((str(relative), "private or unrelated root entry"))
        subdirectories[:] = sorted(name for name in subdirectories
            if name not in SKIP_DIRECTORIES and not name.endswith(".egg-info")
            and not (parent / name).is_symlink())
        for name in sorted(filenames):
            path = parent / name
            relative = path.relative_to(root)
            if path.is_symlink():
                issues.append((str(relative), "symlink"))
                continue
            suffix = path.suffix.lower()
            sample_location = ("examples" in relative.parts[:-1] or
                any(relative.parts[index:index + 2] == ("tests", "test_data")
                    for index in range(len(relative.parts) - 1)))
            if suffix in BLOCKED_SUFFIXES or name.lower() in {"codex.md", ".rdata", ".rhistory"}:
                issues.append((str(relative), "notebook, research output, archive or operational note"))
            if suffix in SAMPLE_ONLY_SUFFIXES and not sample_location:
                issues.append((str(relative), "data or generated artifact outside examples/test_data"))
            if relative.parts[0] in BLOCKED_ROOT_NAMES:
                issues.append((str(relative), "private or unrelated root entry"))
            size = path.stat().st_size
            if size > MAX_PUBLIC_BYTES:
                issues.append((str(relative), "oversized public file"))
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and name not in {"README", "LICENSE", ".gitignore", "Makefile"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError:
                issues.append((str(relative), "non-UTF-8 text file"))
                continue
            for category, pattern in [("personal absolute path", PERSONAL_PATH),
                                      ("email address", EMAIL),
                                      ("private infrastructure identifier", INFRASTRUCTURE),
                                      ("credential-shaped literal", CREDENTIAL)]:
                if pattern.search(text):
                    issues.append((str(relative), category))
    return issues


class PublicTreeTests(unittest.TestCase):
    def test_public_source_tree(self):
        issues = scan_public_tree(PUBLIC_ROOT)
        self.assertEqual(issues, [], "\n".join(f"{path}: {category}" for path, category in issues))

    def test_scanner_rejects_private_paths_notebooks_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            private = root / "private.py"
            private.write_text("/" + "home" + "/example-owner/data")
            (root / ("experiment" + ".ipynb")).write_text("{}")
            (root / "link").symlink_to(private)
            issues = scan_public_tree(root)
            self.assertIn(("private.py", "personal absolute path"), issues)
            self.assertIn(("experiment.ipynb", "notebook, research output, archive or operational note"), issues)
            self.assertIn(("link", "symlink"), issues)
            # Reports must not expose the matched path or its owner.
            self.assertNotIn("example-owner", repr(issues))

    def test_scanner_checks_outputs_credentials_email_and_giant_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "temp.md").write_text("private operational note")
            (root / "settings.txt").write_text("A" + "KIA" + "0" * 16 + "\n" + "user" + "@" + "example.org")
            with (root / "large.csv").open("wb") as stream:
                stream.truncate(MAX_PUBLIC_BYTES + 1)
            issues = scan_public_tree(root)
            self.assertIn(("settings.txt", "credential-shaped literal"), issues)
            self.assertIn(("settings.txt", "email address"), issues)
            self.assertIn(("large.csv", "oversized public file"), issues)
            self.assertIn(("temp.md", "private or unrelated root entry"), issues)

    def test_public_urls_attribution_and_parameterized_paths_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(
                "https://example.org/docs/\nAuthor attribution\n"
                "${DATA_ROOT}/corpus\nPath.home()\n/data/example\n")
            self.assertEqual(scan_public_tree(root), [])

    def test_small_generated_files_are_not_silently_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["figure.png", "figure.pdf", "table.parquet", "table.csv", "run.log"]:
                (root / name).write_text("tiny generated artifact")
            for name in [".RData", ".Rhistory", "workspace.rda"]:
                (root / name).write_text("private session")
            issues = scan_public_tree(root)
            self.assertEqual(sum(category == "data or generated artifact outside examples/test_data"
                                 for _, category in issues), 5)
            self.assertEqual(sum(category == "notebook, research output, archive or operational note"
                                 for _, category in issues), 3)

    def test_small_explicit_synthetic_test_data_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for location in [root / "project" / "examples", root / "project" / "tests" / "test_data"]:
                location.mkdir(parents=True)
                (location / "counts.csv").write_text("word,count\nexample_NOUN,3\n")
                (location / "table.parquet").write_bytes(b"synthetic placeholder")
            self.assertEqual(scan_public_tree(root), [])


if __name__ == "__main__":
    unittest.main()
