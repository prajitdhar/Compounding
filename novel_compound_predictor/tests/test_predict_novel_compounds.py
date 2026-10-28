"""Run from predictor directory: python -m unittest discover -s tests -v."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict_novel_compounds as nnc


class CompoundPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.rng = np.random.default_rng(12)
        self.dimensions = 300
        self.pairs = [(f"m{i}", f"h{i}") for i in range(12)]
        for role, prefix in [("modifier", "m"), ("head", "h")]:
            rows = []
            for i in range(12):
                for decade in [1980, 1990, 2000]:
                    vector = self.rng.normal(0, 0.05, 300) if decade < 2000 else np.full(300, 9999)
                    rows.append([f"{prefix}{i}", decade, *vector])
            # A future-only constituent must never enter a vocabulary.
            rows.append([f"{prefix}_future", 2000, *np.ones(300)])
            pd.DataFrame(rows, columns=[role, "decade", *[f"dim_{i}" for i in range(300)]]).to_parquet(
                self.folder / f"{role}s.parquet", index=False)
        self.attested = self.pairs + [("m0", "h1"), ("m11", "h0"), ("m_future", "h_future")]
        self.write_pairs("attested.csv", self.attested)
        pd.DataFrame([(*pair, True) for pair in self.pairs + [("m_future", "h_future")]] +
                     [("m11", "h0", False)], columns=["modifier", "head", "novel"]).to_csv(
                         self.folder / "positives.csv", index=False)
        self.write_pairs("historical.csv", [("m11", "h0")])

    def tearDown(self):
        self.temporary.cleanup()

    def write_pairs(self, filename, pairs):
        pd.DataFrame(pairs, columns=["modifier", "head"]).to_csv(self.folder / filename, index=False)

    def training_arguments(self, encoder="pooled", output="model"):
        return nnc.parser().parse_args([
            "train", "--modifiers", str(self.folder / "modifiers.parquet"),
            "--heads", str(self.folder / "heads.parquet"),
            "--positives", str(self.folder / "positives.csv"), "--positive-flag", "novel",
            "--attested", str(self.folder / "attested.csv"), "--output-dir", str(self.folder / output),
            "--encoder", encoder, "--epochs", "2", "--batch-size", "5", "--hidden-size", "12",
            "--lstm-hidden-size", "12", "--device", "cpu",
        ])

    def fit(self, encoder="pooled", output="model"):
        with contextlib.redirect_stdout(io.StringIO()):
            return nnc.train(self.training_arguments(encoder, output))

    def test_architecture_preserves_configured_dimensions(self):
        model = nnc.CompoundNNM()
        self.assertEqual(model.fc1.in_features, 600)
        self.assertEqual(model.fc1.out_features, 300)
        self.assertEqual(model.fc2.out_features, 2)
        self.assertIsNone(model.fc2.bias)
        self.assertIsInstance(model.relu, torch.nn.ReLU)

    def test_future_rows_are_excluded_before_vocabulary_pooling_or_sequence(self):
        path = self.folder / "modifiers.parquet"
        pooled = nnc.load_embeddings(path, "modifier", 300, "pooled", 1800, 1999)
        temporal = nnc.load_embeddings(path, "modifier", 300, "lstm", 1800, 1999)
        self.assertNotIn("m_future", pooled.index)
        self.assertNotIn("m_future", temporal.index)
        self.assertEqual(temporal.decades, list(range(1800, 2000, 10)))
        self.assertTrue(torch.equal(temporal.values[:, :18], torch.zeros(12, 18, 300)))
        self.assertTrue(torch.allclose(pooled.values, temporal.values[:, 18:].mean(1)))
        self.assertLess(pooled.values.abs().max().item(), 1)

    def test_dimension_validation_and_duplicate_rows(self):
        path = self.folder / "modifiers.parquet"
        table = pd.read_parquet(path)
        table.drop(columns="dim_299").to_parquet(self.folder / "bad.parquet")
        with self.assertRaisesRegex(ValueError, "300 dimensions"):
            nnc.load_embeddings(self.folder / "bad.parquet", "modifier", 300, "pooled", 1800, 1999)
        pd.concat([table, table.iloc[[0]]]).to_parquet(self.folder / "duplicate.parquet")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            nnc.load_embeddings(self.folder / "duplicate.parquet", "modifier", 300, "pooled", 1800, 1999)
        pooled = table.loc[table.decade.eq(1980)].drop(columns="decade")
        pooled.to_csv(self.folder / "pooled.csv", index=False)
        with self.assertRaisesRegex(ValueError, "requires per-decade"):
            nnc.load_embeddings(self.folder / "pooled.csv", "modifier", 300, "lstm", 1800, 1999)

    def test_mid_decade_cutoff_rejects_indivisible_decade_aggregates(self):
        for encoder in ["pooled", "lstm"]:
            with self.subTest(encoder=encoder):
                with self.assertRaisesRegex(ValueError, "cutoff-year must end in 9"):
                    nnc.load_embeddings(self.folder / "modifiers.parquet", "modifier", 300,
                                        encoder, 1800, 1995)

    def test_corruption_excludes_every_attested_pair_and_splits_are_disjoint(self):
        splits = nnc.split_positives(self.pairs + self.pairs[:2], 1991, .1, .2)
        for role in ["head", "modifier"]:
            examples = nnc.corrupt_splits(splits, [f"m{i}" for i in range(12)],
                                          [f"h{i}" for i in range(12)], set(self.attested), role, 1991)
            self.assertFalse(examples.duplicated(["modifier", "head"]).any())
            negatives = examples.loc[examples.label.eq(0)]
            self.assertTrue(set(negatives[["modifier", "head"]].itertuples(index=False, name=None)).isdisjoint(
                self.attested))
            self.assertEqual(len(negatives), len(self.pairs))
            self.assertTrue((negatives.modifier == negatives.source_modifier).all() if role == "head" else
                            (negatives["head"] == negatives.source_head).all())
            source_splits = examples.groupby(["source_modifier", "source_head"]).split.nunique()
            self.assertTrue(source_splits.eq(1).all())
            self.assertEqual(examples.split.value_counts().to_dict(), {"train": 18, "test": 4, "validation": 2})

    def test_saturated_corruption_fails_instead_of_sampling_false_negatives(self):
        with self.assertRaisesRegex(ValueError, "Cannot generate"):
            nnc.corrupt_splits({"train": [("m", "h")]}, ["m"], ["h"], {("m", "h")}, "head", 1)

    def test_pooled_training_is_deterministic_and_checkpoint_scores_round_trip(self):
        first = self.fit(output="first")
        second = self.fit(output="second")
        self.assertEqual(first["history"], second["history"])
        self.assertEqual(first["test"], second["test"])
        self.assertEqual(first["positive_pairs_requested"], 13)
        self.assertEqual(first["positive_pairs_used"], 12)
        self.assertEqual(first["positive_pairs_missing_historical_constituents"], 1)
        self.assertEqual(first["test"]["examples"], 4)
        model, checkpoint = nnc.load_checkpoint(self.folder / "first/model.pt", torch.device("cpu"))
        self.assertEqual(checkpoint["model_config"]["dimensions"], 300)
        modifiers = nnc.load_embeddings(self.folder / "modifiers.parquet", "modifier", 300, "pooled", 1800, 1999)
        heads = nnc.load_embeddings(self.folder / "heads.parquet", "head", 300, "pooled", 1800, 1999)
        predictions = pd.read_csv(self.folder / "first/test_predictions.csv")
        loader = nnc.make_loader(predictions, modifiers, heads, 1)
        metrics, scores = nnc.evaluate(model, loader, modifiers, heads, torch.device("cpu"))
        np.testing.assert_allclose(scores, predictions.score, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(metrics["accuracy"], first["test"]["accuracy"])

    def test_temporal_training_and_batched_ranking_preserve_future_candidates(self):
        metrics = self.fit("lstm")
        self.assertIn("Separate role LSTMs", metrics["temporal_encoder_choices"])
        self.write_pairs("candidates.csv", self.pairs + [("m11", "h0"), ("missing", "h0")])
        def ranking(batch_size):
            args = nnc.parser().parse_args([
                "rank", "--checkpoint", str(self.folder / "model/model.pt"),
                "--modifiers", str(self.folder / "modifiers.parquet"),
                "--heads", str(self.folder / "heads.parquet"),
                "--attested", str(self.folder / "historical.csv"),
                "--candidates", str(self.folder / "candidates.csv"),
                "--top-k", "20", "--batch-size", str(batch_size), "--device", "cpu",
                "--output", str(self.folder / f"ranking{batch_size}.csv"),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                result = nnc.rank(args)
            return result, pd.read_csv(args.output)
        result, rows = ranking(3)
        other, other_rows = ranking(5)
        self.assertEqual(result["scored"], 12)
        self.assertEqual(result["excluded_attested"], 1)
        self.assertEqual(result["missing_constituents"], 1)
        self.assertEqual(set(rows[["modifier", "head"]].itertuples(index=False, name=None)), set(self.pairs))
        self.assertTrue(rows.score.is_monotonic_decreasing)
        self.assertEqual(rows[["modifier", "head"]].to_dict(), other_rows[["modifier", "head"]].to_dict())
        np.testing.assert_allclose(rows.score, other_rows.score, atol=1e-7, rtol=1e-5)

    def test_cartesian_ranking_search_guard_and_embedding_basis_check(self):
        self.fit()
        args = nnc.parser().parse_args([
            "rank", "--checkpoint", str(self.folder / "model/model.pt"),
            "--modifiers", str(self.folder / "modifiers.parquet"),
            "--heads", str(self.folder / "heads.parquet"),
            "--attested", str(self.folder / "historical.csv"),
            "--output", str(self.folder / "ranking.csv"), "--max-pairs", "100", "--device", "cpu",
        ])
        with self.assertRaisesRegex(ValueError, "Search has 144 pairs"):
            nnc.rank(args)
        table = pd.read_parquet(self.folder / "heads.parquet")
        table.loc[0, "dim_0"] += 1
        table.to_parquet(self.folder / "heads.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "differ from training"):
            nnc.rank(args)

    def test_attested_file_must_include_future_positive_pairs(self):
        arguments = self.training_arguments()
        arguments.attested = str(self.folder / "historical.csv")
        with self.assertRaisesRegex(ValueError, "must include all positive pairs"):
            nnc.train(arguments)

    def test_manifest_checks_svd_fit_dates_even_when_future_embedding_rows_are_dropped(self):
        manifest = {"schema_version": 1, "status": "complete",
                    "time_window": {"start_inclusive": 1800, "end_exclusive": 2010},
                    "parameters": {"dimensions": 300, "train_before": 2010}}
        path = self.folder / "representation_manifest.json"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "SVD training window includes future data"):
            self.fit()
        manifest["time_window"]["end_exclusive"] = 2000
        manifest["parameters"]["train_before"] = 2000
        path.write_text(json.dumps(manifest))
        metrics = self.fit()
        self.assertEqual(metrics["representation_manifests"]["heads"]["content"], manifest)
        self.assertIsNone(metrics["embedding_source_info_warning"])

    def test_checkpoint_records_shared_code_and_rejects_unsupported_or_changed_implementation(self):
        self.fit()
        checkpoint_path = self.folder / "model/model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(set(checkpoint["code_sha256"]),
                         {"predict_novel_compounds.py", "source_info.py", "io.py", "progress.py",
                          "partitioned_parquet.py"})
        self.assertEqual(checkpoint["code_sha256"], nnc.implementation_identity())
        checkpoint["format_version"] = 1
        unsupported = self.folder / "unsupported.pt"
        torch.save(checkpoint, unsupported)
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
            nnc.load_checkpoint(unsupported, torch.device("cpu"))
        checkpoint["format_version"] = 2
        checkpoint["code_sha256"]["source_info.py"] = "0" * 64
        changed = self.folder / "changed.pt"
        torch.save(checkpoint, changed)
        with self.assertRaisesRegex(ValueError, "implementation differs"):
            nnc.load_checkpoint(changed, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
