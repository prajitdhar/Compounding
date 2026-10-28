"""Numerical ranking policies and corpus-specific surface projection."""

import csv
from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.context_vocabulary import equal_decade_ranking_query


class SharedVocabularyTests(unittest.TestCase):
    def setUp(self):
        try:
            import duckdb
            import pandas as pd
        except ImportError:
            self.skipTest("DuckDB and pandas are required")
        self.pd = pd
        self.connection = duckdb.connect()
        self.addCleanup(self.connection.close)

    def test_equal_decade_scores_preserve_each_observed_period_policy(self):
        # A large first decade must not outweigh the smaller second decade.
        self.connection.register("counts", self.pd.DataFrame([
            ("a_NOUN", 1900, 900), ("b_NOUN", 1900, 100),
            ("a_NOUN", 1910, 1), ("c_ADV", 1910, 9)],
            columns=["context", "decade", "count"]))
        eligible = self.connection.execute(equal_decade_ranking_query(
            "counts", ["context"]) + " ORDER BY rank").fetchdf()
        all_observed = self.connection.execute(equal_decade_ranking_query(
            "counts", ["context"], observed_decades=3) + " ORDER BY rank").fetchdf()
        self.assertEqual(eligible.context.tolist(), ["a_NOUN", "c_ADV", "b_NOUN"])
        self.assertEqual(all_observed.context.tolist(), eligible.context.tolist())
        self.assertEqual(eligible.tokens.tolist(), [901, 9, 100])
        self.assertEqual(eligible.decades_observed.tolist(), [2, 1, 1])
        self.assertEqual(eligible["rank"].tolist(), [1, 2, 3])
        for score, expected in zip(eligible.score, [Fraction(1, 2), Fraction(9, 20), Fraction(1, 20)]):
            self.assertAlmostEqual(score, float(expected))
        for score, expected in zip(all_observed.score, [Fraction(1, 3), Fraction(3, 10), Fraction(1, 30)]):
            self.assertAlmostEqual(score, float(expected))

    def test_tuple_and_serialized_context_ties_keep_the_callers_order(self):
        self.connection.register("pairs", self.pd.DataFrame([
            ("a1", "NOUN", 1900, 1), ("a", "NOUN", 1900, 1)],
            columns=["lemma", "pos", "decade", "count"]))
        paired = self.connection.execute(equal_decade_ranking_query(
            "pairs", ["lemma", "pos"], total_count="lemma_match_count",
            ordered_sum=False) + " ORDER BY rank").fetchdf()
        self.assertEqual(paired.lemma.tolist(), ["a", "a1"])
        self.connection.execute("CREATE VIEW contexts AS SELECT "
            "lemma || '_' || pos AS context, decade, count FROM pairs")
        serialized = self.connection.execute(equal_decade_ranking_query(
            "contexts", ["context"]) + " ORDER BY rank").fetchdf()
        self.assertEqual(serialized.context.tolist(), ["a1_NOUN", "a_NOUN"])
        self.assertEqual(paired.lemma_match_count.tolist(), [1, 1])

    def test_integer_totals_and_empty_populations(self):
        self.connection.execute("CREATE TABLE counts(context VARCHAR, decade BIGINT, count BIGINT)")
        empty = self.connection.execute(equal_decade_ranking_query("counts", ["context"]))
        self.assertEqual(empty.fetchall(), [])
        self.connection.execute("INSERT INTO counts VALUES ('large_NOUN', 1900, ?), "
                                "('large_NOUN', 1910, 2)", [2 ** 53 + 1])
        rows = self.connection.execute(equal_decade_ranking_query("counts", ["context"])).fetchall()
        self.assertEqual(rows[0][3], 2 ** 53 + 3)
        self.assertEqual(rows[0][1], 1.0)

    def test_google_adapter_preserves_surface_projection_top_k_and_coverage(self):
        from google_preprocessing import google_v3_pipeline as google

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reduced = root / "reduced"
            reduced.mkdir()
            # Volume counts are deliberately extreme; ranking uses match counts.
            self.pd.DataFrame([
                ("Dogs", "NOUN", 1900, 60, 1), ("dog", "NOUN", 1900, 30, 1),
                ("cat", "NOUN", 1900, 10, 999999),
                ("dog", "NOUN", 1910, 2, 1), ("cats", "NOUN", 1910, 8, 999999),
                ("the", "DET", 1920, 1000, 999999),
                ("unmapped", "NOUN", 1930, 1000, 999999)],
                columns=["surface", "pos", "decade", "count", "volume_count"]
            ).to_parquet(reduced / "unigrams.parquet", index=False)
            self.pd.DataFrame([
                ("Dogs", "dogs", "NOUN", "dog"), ("dog", "dog", "NOUN", "dog"),
                ("cat", "cat", "NOUN", "cat"), ("cats", "cats", "NOUN", "cat"),
                ("the", "the", "DET", "the")],
                columns=["surface", "norm", "pos", "lemma"]
            ).to_parquet(reduced / "lexicon.parquet", index=False)
            destination = root / "contexts.tsv"
            args = SimpleNamespace(output=str(root), vocab_output=str(destination),
                coverage_output=None, lexicon=None, top_k=1, context_pos="NOUN", threads=1,
                memory_limit="128MB", temp_directory=str(root / "scratch"))
            google.make_vocabulary(args)
            with destination.open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["norm"] for row in rows], ["dogs", "dog"])
            self.assertEqual({row["lemma"] for row in rows}, {"dog"})
            self.assertEqual([int(row["surface_match_count"]) for row in rows], [60, 32])
            for row in rows:
                self.assertEqual(int(row["rank"]), 1)
                self.assertEqual(int(row["lemma_match_count"]), 92)
                self.assertEqual(int(row["decades_observed"]), 2)
                self.assertAlmostEqual(float(row["score"]), .55)
            coverage = self.pd.read_csv(root / "contexts_coverage.tsv", sep="\t")
            self.assertEqual(coverage.time.tolist(), [1900, 1910])
            self.assertEqual(coverage.eligible_context_tokens.tolist(), [100, 10])
            self.assertEqual(coverage.retained_context_tokens.tolist(), [90, 2])
            self.assertEqual(coverage.retained_context_types.tolist(), [1, 1])
            receipt = json.loads(Path(str(destination) + ".json").read_text())
            self.assertEqual(receipt["actual_size"], 1)
            self.assertIn("context_vocabulary.py", receipt["source_info"]["software"]["code_sha256"])
            # Google top-k zero means zero selected contexts; COHA's size zero
            # convention remains handled by its own adapter as "all".
            args.top_k = 0
            google.make_vocabulary(args)
            self.assertEqual(len(destination.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
