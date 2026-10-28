"""Exact Parquet schemas, domain pooling, and context-vs-occurrence semantics."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from spacy.tokens import Doc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coha_pipeline as pipeline
from src.count_store import CountStore


class ParquetTests(unittest.TestCase):
    def test_streamed_domain_pooling_keeps_exact_integers_and_empty_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CountStore(root / 'counts.sqlite')
            try:
                store.create('unigram_counts', ['word', 'time', 'domain'])
                huge = 2 ** 53 + 1
                store.add('unigram_counts', [(('a_NOUN', 1820, 'fic'), huge),
                    (('a_NOUN', 1820, 'news'), 3), (('b_VERB', 1820, 'fic'), 2),
                    (('c_DET', 1830, 'fic'), 1)])
                path = root / 'unigram_counts.parquet'
                with patch.object(pd, 'read_sql_query', side_effect=AssertionError('Must stream the cursor')):
                    stats = store.export_parquet('unigram_counts', path, columns=['word', 'time'], integer_columns={'time'}, batch_size=2)
                frame = pd.read_parquet(path)
                self.assertEqual(list(frame), ['word', 'time', 'count'])
                self.assertEqual(frame['count'].tolist(), [huge + 3, 2, 1])
                self.assertEqual(stats, {'rows': 3, 'count_sum': huge + 6})
                self.assertEqual(pq.ParquetFile(path).num_row_groups, 2)
                self.assertEqual(pq.read_schema(path).field('count').type, pa.int64())
                filtered = root / 'fic.parquet'
                store.export_parquet('unigram_counts', filtered, columns=['word', 'time'], integer_columns={'time'}, filters={'domain': ['fic']})
                self.assertEqual(pd.read_parquet(filtered)['count'].tolist(), [huge, 2, 1])
                empty = root / 'empty.parquet'
                store.export_parquet('unigram_counts', empty, columns=['word', 'time'], integer_columns={'time'}, filters={'domain': ['absent']})
                self.assertTrue(pd.read_parquet(empty).empty)
                self.assertEqual(pq.read_schema(empty), pq.read_schema(path))
            finally:
                store.close()

    def test_all_pos_modes_share_occurrences_but_not_context_events(self):
        vocab = spacy.blank('en').vocab
        words = ['we', 'make', 'water', 'tanks', 'work', 'well', 'outside', 'today']
        doc = Doc(vocab, words=words, lemmas=words,
                  pos=['PRON', 'VERB', 'NOUN', 'NOUN', 'VERB', 'ADV', 'ADV', 'ADV'],
                  sent_starts=[True] + [False] * 7)
        record = {'doc_id': '1', 'document_uid': 'source-1', 'fragment_id': 0,
                  'time': 2000, 'domain': 'fic', 'normalized_start': 0,
                  'left_gap': False, 'right_gap': False, 'left_gap_length': 0, 'right_gap_length': 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = CountStore(root / 'counts.sqlite')
            try:
                for name, columns in pipeline.schemas().items():
                    store.create(name, columns)
                pipeline.add_document(store, doc, record, io.StringIO(), 'map')
                cache = {}
                for mode in pipeline.MODES:
                    pipeline.canonical_inputs(store, root / mode, mode, {}, shared_outputs=cache)
                for table in ['unigram_counts', 'bigram_counts']:
                    pd.testing.assert_frame_equal(pd.read_parquet(root / 'local5' / f'{table}.parquet'),
                                                  pd.read_parquet(root / 'sentence' / f'{table}.parquet'))
                local = pd.read_parquet(root / 'local5/bigrams.parquet')
                sentence = pd.read_parquet(root / 'sentence/bigrams.parquet')
                # 'we make' is not an eligible compound, but now gets contexts.
                self.assertEqual(local.loc[local.modifier == 'we_PRON', 'count'].sum(), 3)
                self.assertEqual(sentence.loc[sentence.modifier == 'we_PRON', 'count'].sum(), 6)
                counts = pd.read_parquet(root / 'sentence/bigram_counts.parquet')
                self.assertEqual(counts.loc[counts.modifier == 'we_PRON', 'count'].item(), 1)
                self.assertEqual(list(local), ['modifier', 'head', 'time', 'context', 'count'])
                self.assertEqual(list(pd.read_parquet(root / 'local5/unigrams.parquet')), ['word', 'time', 'context', 'count'])
                annotations = pd.read_sql('SELECT * FROM bigram_annotations', store.connection)
                self.assertEqual(annotations['count'].sum(), 7)
                self.assertIn('PRON', set(annotations.modifier_pos))
                opportunity = pd.read_parquet(root / 'local5/bigram_opportunities_local5.parquet')
                self.assertEqual(opportunity['count'].sum(), 6 * counts['count'].sum())
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
