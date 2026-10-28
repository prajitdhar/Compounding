import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "export_visualization_dataset.py"
spec = importlib.util.spec_from_file_location("export_visualization", MODULE_PATH)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class VisualizationExportTests(unittest.TestCase):
    def make_test_data(self, directory, extension="csv", dimensions=3, missing=False, temporal=False):
        d = Path(directory)
        dims = [f"dim_{i}" for i in range(dimensions)]
        frames = {
            "heads": pd.DataFrame([["boat_NOUN", *range(3, dimensions + 3)],
                                    ["cycle_NOUN", *([2] * dimensions)]], columns=["head", *dims]),
            "modifiers": pd.DataFrame([["glass_NOUN", *range(4, dimensions + 4)],
                                        ["water_NOUN", *([1] * dimensions)]], columns=["modifier", *dims]),
            "compounds": pd.DataFrame([["water_NOUN", "cycle_NOUN", *range(7, dimensions + 7)],
                                        ["water_NOUN", "cycle_NOUN", *range(7, dimensions + 7)]],
                                       columns=["modifier", "head", *dims]),
        }
        args = []
        for role, frame in frames.items():
            if temporal:
                frame["decade"] = 1990
            path = d / f"{role}.{extension}"
            if extension == "parquet":
                frame.to_parquet(path, index=False)
            else:
                frame.to_csv(path, index=False)
            args.extend(["--" + role, str(path)])
        pairs = [("glass_NOUN", "boat_NOUN"), ("water_NOUN", "cycle_NOUN"),
                 ("glass_NOUN", "boat_NOUN")]
        if missing:
            pairs.append(("absent_NOUN", "boat_NOUN"))
        pd.DataFrame(pairs, columns=["modifier", "head"]).to_csv(d / "novel_compounds.csv", index=False)
        return args + ["--novel-compounds", str(d / "novel_compounds.csv"),
                       "--batch-size", "1", "--dimensions", str(dimensions)]

    def test_tagged_lookup_observed_priority_addition_and_streaming(self):
        for extension in ["csv", "parquet"]:
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                d = Path(directory)
                args = self.make_test_data(directory, extension=extension)
                output = d / ("out." + extension)
                manifest = export.main(args + ["--output", str(output)])
                result = pd.read_parquet(output) if extension == "parquet" else pd.read_csv(output, sep="\t")
                self.assertEqual(result.common.tolist(), ["boat", "cycle", "water cycle", "glass boat"])
                self.assertEqual(result.novel.tolist(), [0, 0, 0, 1])
                self.assertEqual(result.columns.tolist(), ["common", "novel", "0", "1", "2"])
                np.testing.assert_allclose(result.loc[result.common.eq("glass boat"), ["0", "1", "2"]], [[7, 9, 11]])
                np.testing.assert_allclose(result.loc[result.common.eq("water cycle"), ["0", "1", "2"]], [[7, 8, 9]])
                self.assertEqual(manifest["counts"]["duplicate_observed_identifiers"], 1)
                self.assertEqual(manifest["counts"]["novel_observed_overlap"], 1)
                self.assertEqual(manifest["counts"]["duplicate_novel_pairs"], 1)
                self.assertEqual(manifest["status"], "complete")

    def test_missing_novel_requires_explicit_coverage_option(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            args = self.make_test_data(directory, missing=True)
            blocked_output = d / "blocked.csv"
            with self.assertRaisesRegex(ValueError, "allow-missing-novel"):
                export.main(args + ["--output", str(blocked_output)])
            self.assertFalse(blocked_output.exists())
            self.assertEqual(list(d.glob(".visualization-*")), [])
            report = pd.read_csv(d / "blocked.csv.missing.csv")
            self.assertEqual(report.modifier.tolist(), ["absent_NOUN"])
            self.assertEqual(report.missing_modifier.tolist(), [True])
            self.assertEqual(json.loads((d / "blocked.csv.json").read_text())["status"], "blocked_missing_novel")
            accepted_output = d / "accepted.csv"
            manifest = export.main(args + ["--output", str(accepted_output), "--allow-missing-novel"])
            self.assertTrue(accepted_output.exists())
            self.assertEqual(manifest["counts"]["missing_novel_rows"], 1)
            self.assertEqual(manifest["rows"], 4)

    def test_temporal_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_test_data(directory, temporal=True)
            output = Path(directory) / "out.csv"
            with self.assertRaisesRegex(ValueError, "pooled"):
                export.main(args + ["--output", str(output)])
            self.assertFalse(output.exists())

    def test_full_300_coordinates_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_test_data(directory, extension="parquet", dimensions=300)
            output = Path(directory) / "out.parquet"
            manifest = export.main(args + ["--output", str(output)])
            result = pd.read_parquet(output)
            self.assertEqual(result.shape, (4, 302))
            self.assertEqual(manifest["dimensions"], 300)
            self.assertEqual(result.loc[result.common.eq("glass boat"), "299"].iloc[0], 605)

    def test_insufficient_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_test_data(directory)
            output = Path(directory) / "out.csv"
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                export.main(args + ["--output", str(output), "--dimensions", "300"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
