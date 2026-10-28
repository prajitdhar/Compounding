"""End-to-end checks for worker-side counting and deterministic shard merging."""
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coha_pipeline as pipeline
from src.count_store import CountStore


class ParallelExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.model = cls.root / 'model'
        nlp = spacy.blank('en')
        nlp.add_pipe('sentencizer')
        ruler = nlp.add_pipe('attribute_ruler')
        for word, pos in [('water', 'NOUN'), ('tank', 'NOUN'), ('maker', 'NOUN'),
                          ('stone', 'NOUN'), ('wall', 'NOUN'), ('old', 'ADJ'),
                          ('works', 'VERB'), ('well', 'ADV')]:
            ruler.add(patterns=[[{'LOWER': word}]], attrs={'POS': pos, 'LEMMA': word})
        nlp.to_disk(cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def prepare(self, root, text):
        source = root / 'text_fic_2000.txt'
        source.write_text(text)
        prepared = root / 'prepared'
        pipeline.prepare(SimpleNamespace(input=source, output=prepared,
                                         metadata=None, encoding='utf-8', domains=None))
        return prepared

    def arguments(self, root, prepared, workers):
        return SimpleNamespace(input=prepared, output=root / f'extracted-{workers}',
                               model=str(self.model), max_length=1000000,
                               workers=workers, batch_size=2, shard_size=2,
                               scratch_dir=root / 'scratch', proper_nouns='map', domains=None)

    def test_parallel_matches_serial_reference_and_feature_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self.prepare(root,
                '@@100 water tank maker works well. @@101 old stone wall works. '
                '@@102 water tank works @ @ @ stone wall works well. '
                '@@103 water tank maker works well. @@104 water tank. @@105 old stone wall works.')
            args = self.arguments(root, prepared, 1)
            pipeline.extract(args)
            serial_output = args.output
            args = self.arguments(root, prepared, 2)
            pipeline.extract(args)
            parallel_output = args.output
            first = json.loads((serial_output / 'extraction_manifest.json').read_text())
            second = json.loads((parallel_output / 'extraction_manifest.json').read_text())
            self.assertEqual(first['stats'], second['stats'])
            self.assertGreater(second['stats']['eligible_pair_occurrences'], 0)
            self.assertEqual(first['sentences_sha256'], second['sentences_sha256'])
            self.assertEqual(second['execution']['workers'], 2)
            self.assertEqual(second['execution']['scratch_dir'], str(args.scratch_dir.resolve()))
            self.assertEqual(second['execution']['sqlite_temp_store'], 'FILE')
            self.assertEqual(Path(second['execution']['sqlite_tmpdir']).parent, args.scratch_dir.resolve())
            for path in serial_output.rglob('*'):
                if path.suffix == '.gz' or path.suffix == '.parquet':
                    self.assertEqual(path.read_bytes(), (parallel_output / path.relative_to(serial_output)).read_bytes(),
                                     str(path.relative_to(serial_output)))

            # Independent pre-sharding reference: parse/count the entire stream
            # directly, with no worker tasks or database merge.
            reference = CountStore(root / 'reference.sqlite')
            try:
                for name, columns in pipeline.schemas().items():
                    reference.create(name, columns)
                nlp = spacy.load(self.model)
                sentences = io.StringIO()
                stats = Counter()
                for doc, record in nlp.pipe(pipeline.read_fragments(prepared / 'fragments.jsonl.gz'),
                                            as_tuples=True, batch_size=2):
                    stats.update(pipeline.add_document(reference, doc, record, sentences, 'map'))
                self.assertEqual(dict(stats), second['stats'])
                with gzip.open(parallel_output / 'sentences.jsonl.gz', 'rt') as stream:
                    self.assertEqual(sentences.getvalue(), stream.read())
                for table, columns in reference.schemas.items():
                    pooled_columns = [c for c in columns if c != 'domain']
                    fields = ', '.join(f'"{c}"' for c in pooled_columns)
                    expected = pd.read_sql(f'SELECT {fields}, SUM(count) AS count FROM "{table}" GROUP BY {fields} ORDER BY {fields}', reference.connection)
                    path = (parallel_output / 'feature_inputs/local5/unigram_counts.parquet'
                            if table == 'unigram_counts' else parallel_output / pipeline.audit_table_path(table))
                    actual = pd.read_parquet(path)
                    for c in ['time', 'distance', 'direct_dependency', 'sentence_complete', 'count']:
                        if c in expected:
                            expected[c] = expected[c].astype('int64')
                    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
                    self.assertNotIn('domain', actual)
            finally:
                reference.close()
            self.assertEqual(list(args.scratch_dir.iterdir()), [])
            inputs = parallel_output / 'feature_inputs/local5'
            self.assertEqual(list(parallel_output.rglob('*.pkl')), [])
            self.assertEqual(list(parallel_output.rglob('*.csv.gz')), [])
            self.assertFalse((parallel_output / 'feature_inputs/shared').exists())
            for table in ['unigram_counts', 'bigram_counts']:
                first_counts = pd.read_parquet(inputs / f'{table}.parquet')
                second_counts = pd.read_parquet(inputs.parent / 'sentence' / f'{table}.parquet')
                pd.testing.assert_frame_equal(first_counts, second_counts)
            self.assertEqual(set(p.name for p in (parallel_output / 'tables').iterdir()),
                             {'bigram_counts.parquet', 'bigram_document_counts.parquet', 'bigram_annotations.parquet'})
            self.assertIn('works_VERB', set(pd.read_parquet(inputs / 'unigrams.parquet').word))
            all_bigrams = pd.read_parquet(inputs / 'bigrams.parquet')
            self.assertTrue(all_bigrams.modifier.str.endswith('_VERB').any())
            self.assertEqual(len(second['source_info']['outputs']),
                             len({item['path'] for item in second['source_info']['outputs']}))
            for path in [prepared / 'prepared_manifest.json', parallel_output / 'extraction_manifest.json',
                         inputs / 'corpus_manifest.json']:
                envelope = json.loads(path.read_text())['source_info']
                self.assertEqual(envelope['status'], 'complete')
                self.assertEqual(envelope['time']['column'], 'time')
                for item in envelope['outputs']:
                    self.assertEqual(pipeline.digest_file(item['path']), item['sha256'])
                    self.assertEqual(Path(item['path']).stat().st_size, item['bytes'])
            self.assertTrue(second['model_sha256'])
            self.assertIn('source_info.py', second['source_info']['software']['code_sha256'])

    def test_worker_failure_does_not_publish_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self.prepare(root, '@@100 water tank works well.')
            args = self.arguments(root, prepared, 2)
            args.max_length = 5
            with self.assertRaisesRegex(RuntimeError, 'Extraction failed in shard 0'):
                pipeline.extract(args)
            self.assertFalse((args.output / 'extraction_manifest.json').exists())
            self.assertFalse((args.output / 'feature_inputs/local5/corpus_manifest.json').exists())
            self.assertEqual(list(args.scratch_dir.iterdir()), [])

    def test_export_disk_full_reports_paths_and_cleans_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self.prepare(root, '@@100 water tank works well.')
            args = self.arguments(root, prepared, 1)
            with patch.object(pipeline, 'canonical_inputs', side_effect=pd.errors.DatabaseError('database or disk is full')):
                with self.assertRaisesRegex(RuntimeError, 'SQLite ran out of space') as caught:
                    pipeline.extract(args)
            self.assertIn(str(args.scratch_dir), str(caught.exception))
            self.assertIn(str(args.output), str(caught.exception))
            self.assertFalse((args.output / 'extraction_manifest.json').exists())
            self.assertEqual(list(args.scratch_dir.iterdir()), [])

    def test_empty_input_produces_empty_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = self.prepare(root, '@@100 @ @ @')
            args = self.arguments(root, prepared, 2)
            pipeline.extract(args)
            self.assertEqual(json.loads((args.output / 'extraction_manifest.json').read_text())['stats'], {})
            self.assertTrue(pd.read_parquet(args.output / 'feature_inputs/local5/bigrams.parquet').empty)


class MergeTests(unittest.TestCase):
    def test_merge_is_additive_and_rolls_back_on_invalid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = CountStore(root / 'destination.sqlite')
            shard = CountStore(root / 'shard.sqlite')
            invalid = CountStore(root / 'invalid.sqlite')
            try:
                for store in [destination, shard, invalid]:
                    store.create('first', ['word'])
                    store.create('second', ['word' if store is not invalid else 'wrong_column'])
                destination.add('first', [(('water',), 5)])
                shard.add('first', [(('water',), 2), (('tank',), 3)])
                shard.connection.commit()
                invalid.add('first', [(('water',), 100)])
                invalid.connection.commit()
                with self.assertRaisesRegex(ValueError, 'Incompatible shard schema'):
                    destination.merge_from(root / 'invalid.sqlite')
                self.assertEqual(destination.connection.execute('SELECT * FROM first').fetchall(), [('water', 5)])
                destination.merge_from(root / 'shard.sqlite')
                self.assertEqual(destination.connection.execute('SELECT * FROM first ORDER BY word').fetchall(),
                                 [('tank', 3), ('water', 7)])
            finally:
                destination.close()
                shard.close()
                invalid.close()


if __name__ == '__main__':
    unittest.main()
