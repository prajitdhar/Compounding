"""Completed table reuse accepts only the current checkpoint format."""

import json
from pathlib import Path
import tempfile
import unittest

from src import partitioned_parquet as partitions


class PublishedCheckpointFormatTests(unittest.TestCase):
    def make_test_data(self, root, version):
        destination = root / "result.parquet"
        destination.write_bytes(b"completed-output-test_data")
        work_root = root / "work"
        candidate = work_root / "result-existing"
        candidate.mkdir(parents=True)
        workspace = work_root / "result-requested"
        shared = {"duckdb_version": "test", "pyarrow_version": "test",
                  "sources": [{"path": "input", "bytes": 1}],
                  "identity": {"dataset": "same"}, "keys": ["word"],
                  "query": "SELECT word, count FROM partition_input",
                  "input_columns": ["word", "count"],
                  "required_identifiers": ["word"], "preserve_count_mass": True}
        previous = {**shared, "algorithm_version": version,
                    "implementation_sha256": "same-implementation", "partitions": 2}
        current = {**shared, "algorithm_version": partitions.ALGORITHM_VERSION,
                   "implementation_sha256": "same-implementation", "partitions": 4}
        (candidate / "specification.json").write_text(json.dumps(previous))
        (candidate / "published.json").write_text(json.dumps({
            "fingerprint": partitions._digest(previous),
            "statistics": {"rows": 1, "count_sum": 7},
            "files": [partitions._file_record(destination, root)]}))
        return work_root, workspace, destination, current

    def test_current_published_table_reuses_matching_logical_specification(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_test_data(Path(directory), partitions.ALGORITHM_VERSION)
            result = partitions._reuse_published(*args, partitions._digest(args[-1]))
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["count_sum"], 7)
            self.assertEqual(result["reused_algorithm_version"], partitions.ALGORITHM_VERSION)
            self.assertTrue((args[1] / "published.json").is_file())
            self.assertEqual(json.loads((args[1] / "specification.json").read_text()), args[-1])

    def test_unsupported_published_formats_are_not_adopted(self):
        for version in [None, 1, partitions.ALGORITHM_VERSION + 1]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                args = self.make_test_data(Path(directory), version)
                result = partitions._reuse_published(*args, partitions._digest(args[-1]))
                self.assertIsNone(result)
                self.assertFalse(args[1].exists())


if __name__ == "__main__":
    unittest.main()
