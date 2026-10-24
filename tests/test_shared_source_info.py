"""Corpus-free specifications for the shared artifact and table utilities."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src import __version__
from src.io import atomic_path, write_parquet
from src.source_info import (
    FEATURE_INPUT_SCHEMA_VERSION, artifact, canonical_time, external_artifact,
    feature_table, source_info, sha256_file, software_record,
    source_identity_paths, validate_feature_input_manifest, write_json,
)


class SharedSourceInfoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_package_version_and_authoritative_identities(self):
        self.assertEqual(__version__, "0.4.0")
        paths = source_identity_paths()
        self.assertIsInstance(paths, tuple)
        self.assertEqual([p.name for p in paths],
                         ["source_info.py", "io.py", "progress.py", "partitioned_parquet.py"])
        self.assertTrue(all(p.is_absolute() and p.is_file() for p in paths))
        record = software_record(paths, ["a-package-that-does-not-exist"])
        self.assertEqual(set(record["code_sha256"]), {p.name for p in paths})
        self.assertIsNone(record["packages"]["a-package-that-does-not-exist"])
        self.assertEqual(software_record([])["code_sha256"], record["code_sha256"])

    def test_hash_and_artifact_preserve_explicit_identity(self):
        path = self.root / "input.bin"
        content = bytes(range(256)) * 8193
        path.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(sha256_file(path), expected)
        self.assertEqual(artifact(path), {"name": "input.bin", "path": str(path.resolve()),
                                        "bytes": len(content), "sha256": expected})
        explicit = artifact(path, name="logical_input", sha256="provided")
        self.assertEqual(explicit["name"], "logical_input")
        self.assertEqual(explicit["sha256"], "provided")
        external = external_artifact("remote", "https://example.org/corpus")
        self.assertIsNone(external["sha256"])
        self.assertIsNone(external["bytes"])

    def test_ambiguous_code_basename_is_rejected(self):
        left = self.root / "left" / "helper.py"
        right = self.root / "right" / "helper.py"
        left.parent.mkdir()
        right.parent.mkdir()
        left.write_text("left")
        right.write_text("right")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            software_record([left, right])
        self.assertIn("helper.py", software_record([left, left])["code_sha256"])

    def test_atomic_failure_preserves_existing_destination(self):
        destination = self.root / "value.txt"
        destination.write_text("original")
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            with atomic_path(destination) as temporary:
                self.assertEqual(temporary.parent, destination.parent)
                self.assertNotEqual(temporary, destination)
                temporary.write_text("partial replacement")
                raise RuntimeError("writer failed")
        self.assertEqual(destination.read_text(), "original")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_atomic_json_is_stable_and_invalid_json_preserves_target(self):
        destination = self.root / "nested" / "metadata.json"
        write_json(destination, {"z": 2, "a": 1})
        initial = destination.read_text()
        self.assertTrue(initial.endswith("\n"))
        self.assertLess(initial.index('"a"'), initial.index('"z"'))
        self.assertEqual(json.loads(initial), {"a": 1, "z": 2})
        with self.assertRaises(TypeError):
            write_json(destination, {"bad": object()})
        self.assertEqual(destination.read_text(), initial)
        self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_parquet_schema_and_write_failure(self):
        destination = self.root / "table.parquet"
        frame = pd.DataFrame({"word": ["stone_NOUN"], "time": [1900], "count": [7]})
        write_parquet(frame, destination)
        pd.testing.assert_frame_equal(pd.read_parquet(destination), frame)
        record = feature_table(destination, frame.columns, "match_count", required=True)
        self.assertEqual(record["columns"], list(frame.columns))
        self.assertEqual(record["format"], "parquet")
        self.assertEqual(record["count_unit"], "match_count")
        initial = destination.read_bytes()

        class BrokenFrame:
            def to_parquet(self, temporary, **kwargs):
                Path(temporary).write_bytes(b"broken")
                raise RuntimeError("parquet failed")

        with self.assertRaisesRegex(RuntimeError, "parquet failed"):
            write_parquet(BrokenFrame(), destination)
        self.assertEqual(destination.read_bytes(), initial)
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_source_info_envelope_status_and_defaults(self):
        value = source_info("extract", "example", [], {"cutoff": 0}, {"unit": "decade"}, {})
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["outputs"], [])
        self.assertEqual(value["statistics"], {})
        self.assertNotIn("timestamp", value)
        for status in ["planned", "partial", "complete"]:
            self.assertEqual(source_info("stage", "corpus", [], {}, {}, {}, status=status)["status"], status)
        with self.assertRaisesRegex(ValueError, "status"):
            source_info("stage", "corpus", [], {}, {}, {}, status="unknown")

    def test_feature_input_specification_rejects_incomplete_or_wrong_population(self):
        value = {"feature_input_schema_version": FEATURE_INPUT_SCHEMA_VERSION,
                 "format": "parquet", "word_target_pos": "all_lexical",
                 "bigram_target_pos": "all_lexical",
                 "table_files": dict.fromkeys(["unigrams", "unigram_counts", "bigrams", "bigram_counts"]),
                 "source_info": {"status": "complete", "stage": "feature_inputs"}}
        self.assertIs(validate_feature_input_manifest(value), value)
        for key, wrong in [("feature_input_schema_version", 2), ("format", "pickle"),
                           ("word_target_pos", "nouns_only"), ("table_files", {}),
                           ("source_info", {"status": "partial", "stage": "feature_inputs"})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_feature_input_manifest({**value, key: wrong})

    def test_canonical_time_does_not_mutate_or_merge_conflicting_clocks(self):
        frame = pd.DataFrame({"year": [1900, 1910], "count": [1, 2]})
        actual = canonical_time(frame)
        self.assertIn("year", frame)
        self.assertNotIn("year", actual)
        self.assertEqual(list(actual.time), [1900, 1910])
        same = pd.DataFrame({"time": [1900], "year": [1900]})
        self.assertEqual(list(canonical_time(same).columns), ["time"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            canonical_time(pd.DataFrame({"time": [1900], "year": [1910]}))
        with self.assertRaisesRegex(ValueError, "Missing"):
            canonical_time(pd.DataFrame({"count": [1]}))


if __name__ == "__main__":
    unittest.main()
