"""Scientific ranking from raw occurrences, including empty eligible decades."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocabulary import build_vocabulary
from src.source_info import (artifact, sha256_file, source_info,
                             validate_feature_input_manifest, write_json)


class VocabularyTests(unittest.TestCase):
    def arguments(self, root, size=0):
        return SimpleNamespace(input=root / 'inputs', output=root / 'ranked.tsv',
            size=size, threads=1, memory_limit='128MB',
            temp_directory=root / 'scratch', max_temp_directory_size='1GB')

    def input(self, root):
        directory = root / 'inputs'
        directory.mkdir()
        counts = directory / 'unigram_counts.parquet'
        # The 1920 decade has no eligible contexts, but still has equal weight.
        # Global token frequency would put b before c, unlike this policy.
        pd.DataFrame([('a_NOUN', 1900, 45), ('a_NOUN', 1901, 45),
                      ('b_NOUN', 1900, 10), ('a_NOUN', 1910, 1),
                      ('c_ADV', 1910, 9), ('we_PRON', 1920, 100)],
                     columns=['word', 'time', 'count']).to_parquet(counts, index=False)
        info = source_info('feature_inputs', {'name': 'COHA'}, [], {},
            {'column': 'time', 'unit': 'calendar_year'}, {}, outputs=[artifact(counts)])
        manifest = {'schema_version': 3, 'format': 'parquet',
            'word_target_pos': 'all_lexical', 'bigram_target_pos': 'all_lexical',
            'context_mode': 'sentence', 'table_files':
                {name: name + '.parquet' for name in
                 ['unigrams', 'unigram_counts', 'bigrams', 'bigram_counts']},
            'source_info': info}
        validate_feature_input_manifest(manifest)
        write_json(directory / 'corpus_manifest.json', manifest)

    def test_equal_decade_ranking_and_private_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.input(root)
            args = self.arguments(root)
            info = build_vocabulary(args)
            ranked = pd.read_csv(args.output, sep='\t')
            self.assertEqual(ranked.context.tolist(), ['a_NOUN', 'c_ADV', 'b_NOUN'])
            self.assertEqual(ranked['rank'].tolist(), [1, 2, 3])
            for actual, expected in zip(ranked.score, [1 / 3, .3, 1 / 30]):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(ranked.tokens.tolist(), [91, 9, 10])
            self.assertEqual(ranked.decades_observed.tolist(), [2, 1, 1])
            self.assertEqual(info['statistics']['observed_decades'], [1900, 1910, 1920])
            self.assertEqual(info['outputs'][0]['sha256'], sha256_file(args.output))
            self.assertIn('source_info.py', info['software']['code_sha256'])
            self.assertIn('context_vocabulary.py', info['software']['code_sha256'])
            self.assertEqual(json.loads(Path(str(args.output) + '.source_info.json').read_text())['source_info'], info)
            self.assertEqual(list(args.temp_directory.iterdir()), [])
            args.size = 2
            build_vocabulary(args)
            self.assertEqual(pd.read_csv(args.output, sep='\t').context.tolist(), ['a_NOUN', 'c_ADV'])

    def test_ties_are_lexical_and_mutated_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.input(root)
            args = self.arguments(root)
            counts = args.input / 'unigram_counts.parquet'
            manifest_path = args.input / 'corpus_manifest.json'
            pd.DataFrame([('z_PROPN', 2000, 1), ('a_NUM', 2000, 1)],
                columns=['word', 'time', 'count']).to_parquet(counts, index=False)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                build_vocabulary(args)
            manifest = json.loads(manifest_path.read_text())
            manifest['source_info']['outputs'] = [artifact(counts)]
            write_json(manifest_path, manifest)
            build_vocabulary(args)
            self.assertEqual(pd.read_csv(args.output, sep='\t').context.tolist(), ['a_NUM', 'z_PROPN'])

    def test_vocabulary_and_receipt_cannot_replace_input_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.input(root)
            args = self.arguments(root)
            counts = args.input / 'unigram_counts.parquet'
            original = counts.read_bytes()
            args.output = counts
            with self.assertRaisesRegex(ValueError, 'must not replace'):
                build_vocabulary(args)
            self.assertEqual(counts.read_bytes(), original)
            args.output = args.input / 'ranked.tsv'
            receipt_path = Path(str(args.output) + '.source_info.json')
            receipt_path.write_text('existing input artifact')
            manifest_path = args.input / 'corpus_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['table_files']['bigrams'] = receipt_path.name
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'must not replace'):
                build_vocabulary(args)
            self.assertEqual(receipt_path.read_text(), 'existing input artifact')
            self.assertFalse(args.output.exists())

    def test_relocated_inputs_keep_checksum_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.input(root)
            args = self.arguments(root)
            manifest_path = args.input / 'corpus_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['source_info']['outputs'][0]['name'] = 'sentence/unigram_counts.parquet'
            write_json(manifest_path, manifest)
            relocated = root / 'relocated'
            args.input.rename(relocated)
            args.input = relocated
            info = build_vocabulary(args)
            self.assertEqual(info['inputs'][1]['path'], str((relocated / 'unigram_counts.parquet').resolve()))
            self.assertEqual(pd.read_csv(args.output, sep='\t').context.tolist(), ['a_NOUN', 'c_ADV', 'b_NOUN'])
            original_output = args.output.read_bytes()
            counts = relocated / 'unigram_counts.parquet'
            frame = pd.read_parquet(counts)
            frame.loc[0, 'count'] += 1
            frame.to_parquet(counts, index=False)
            with self.assertRaisesRegex(ValueError, 'do not match'):
                build_vocabulary(args)
            self.assertEqual(args.output.read_bytes(), original_output)

    def test_count_identity_must_be_unique_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.input(root)
            args = self.arguments(root)
            manifest_path = args.input / 'corpus_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            identity = manifest['source_info']['outputs'][0]
            cases = [([], 'must identify'),
                     ([dict(identity, sha256=None)], 'checksum'),
                     ([identity, dict(identity, path='another/unigram_counts.parquet',
                                      name='local5/unigram_counts.parquet')], 'Ambiguous')]
            for records, message in cases:
                with self.subTest(message=message):
                    manifest['source_info']['outputs'] = records
                    write_json(manifest_path, manifest)
                    with self.assertRaisesRegex(ValueError, message):
                        build_vocabulary(args)
                    self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
