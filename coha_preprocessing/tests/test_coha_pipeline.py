"""Regression tests for raw documents, gaps, and the two COHA context modes."""
from collections import Counter, defaultdict
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

import pandas as pd
import spacy
from spacy.tokens import Doc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coha_pipeline as pipeline
from src.count_store import CountStore


class PreparationTests(unittest.TestCase):
    def test_stage_paths_reject_source_overlap_before_mutating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            marker = source / 'prepared_manifest.json'
            marker.write_text('original')
            for output in [source, source / 'generated', root]:
                args = SimpleNamespace(input=source, output=output)
                with self.assertRaisesRegex(ValueError, 'non-overlapping'):
                    pipeline.prepare(args)
                with self.assertRaisesRegex(ValueError, 'non-overlapping'):
                    pipeline.extract(args)
                self.assertEqual(marker.read_text(), 'original')
            with self.assertRaisesRegex(ValueError, 'Scratch directory'):
                pipeline.extract(SimpleNamespace(input=source, output=root / 'output',
                                                 scratch_dir=source / 'scratch'))
            self.assertFalse((source / 'generated').exists())
            self.assertFalse((source / 'scratch').exists())

    def test_legacy_prepared_records_are_read_as_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fragments.jsonl.gz'
            with pipeline.gzip_writer(path) as writer:
                writer.write(json.dumps({'text': 'water tank', 'year': 1820, 'year_precision': 'decade'}) + '\n')
            before = pipeline.digest_file(path)
            _, record = next(pipeline.read_fragments(path))
            self.assertEqual(record, {'text': 'water tank', 'time': 1820, 'time_precision': 'decade'})
            self.assertEqual(before, pipeline.digest_file(path))

    def test_marker_suffix_and_rating_are_not_prose_boundaries(self):
        docs = list(pipeline.documents('@@1562.txt  | First text. ##1/2 stars.\n@@200 Second text.', 'fallback'))
        self.assertEqual([doc_id for doc_id, _ in docs], ['1562', '200'])
        self.assertEqual(docs[0][1].strip(), 'First text. ##1/2 stars.')
        glued = list(pipeline.documents('@@962.txtCHAPTER ONE\n@@1335.txtp50Figure', 'fallback'))
        self.assertEqual(glued, [('962', 'CHAPTER ONE\n'), ('1335', 'p50Figure')])
        period = list(pipeline.documents('@@183069049. I AM accustomed. ##1.5 rating.', 'fallback'))
        self.assertEqual(period, [('183069049', ' I AM accustomed. ##1.5 rating.')])

    def test_reused_ids_have_distinct_identities_and_scoped_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'raw'
            source.mkdir()
            (source / 'text_acad_1950.txt').write_text('@@1562.txt | Academic text.')
            (source / 'text_fic_1850.txt').write_text('@@1562 Fiction text.')
            output = root / 'prepared'
            args = SimpleNamespace(input=source, output=output, metadata=None, encoding='utf-8', domains=None)
            pipeline.prepare(args)
            records = [record for _, record in pipeline.read_fragments(output / 'fragments.jsonl.gz')]
            self.assertEqual([r['doc_id'] for r in records], ['1562', '1562'])
            self.assertEqual(len({r['document_uid'] for r in records}), 2)
            manifest = json.loads((output / 'prepared_manifest.json').read_text())
            self.assertEqual(manifest['stats']['document_id_reuses_across_sources'], 1)
            metadata = root / 'metadata.tsv'
            metadata.write_text('doc_id\tyear\tdomain\n1562\t1951\tacad\n')
            args.metadata = metadata
            with self.assertRaisesRegex(ValueError, 'ambiguous.*source column'):
                pipeline.prepare(args)
            metadata.write_text('source\tdoc_id\ttime\tdomain\n'
                                'text_acad_1950.txt\t1562\t1951\tacad\n'
                                'text_fic_1850.txt\t1562\t1852\tfic\n')
            pipeline.prepare(args)
            scoped = [record for _, record in pipeline.read_fragments(output / 'fragments.jsonl.gz')]
            self.assertEqual([r['time'] for r in scoped], [1951, 1852])
            self.assertEqual([r['document_uid'] for r in scoped], [r['document_uid'] for r in records])
            args.metadata = None
            (source / 'text_acad_1950.txt').write_text('@@1562 First. @@1562 Second.')
            with self.assertRaisesRegex(ValueError, 'within source.*text_acad_1950.txt'):
                pipeline.prepare(args)

    def test_standalone_zip_and_tar_text_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = 'text_fic_1820.txt'
            payload = b'@@100 First document.'
            zipped = root / 'text.zip'
            with zipfile.ZipFile(zipped, 'w') as archive:
                archive.writestr(member, payload)
            self.assertEqual(list(pipeline.text_sources(zipped)), [(f'text.zip!{member}', payload)])
            for mode, filename in [('w', 'text.tar'), ('w:gz', 'text.tar.gz')]:
                source = root / filename
                with tarfile.open(source, mode) as archive:
                    info = tarfile.TarInfo(member)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                self.assertEqual(list(pipeline.text_sources(source)), [(f'{filename}!{member}', payload)])

    def test_tar_with_multiple_zips_prepares_every_member(self):
        # ZIP detection searches near EOF and accepts prepended bytes, so an
        # uncompressed TAR can also appear to be its final embedded ZIP.
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'coha-text.tar'
                texts = {'text_fic_1820': '@@100 First document.',
                         'text_tvm_2010': '@@101 Last document.'}
                with tarfile.open(source, 'w') as archive:
                    for stem, text in texts.items():
                        zipped = io.BytesIO()
                        with zipfile.ZipFile(zipped, 'w', compression=compression) as inner:
                            inner.writestr(stem + '.txt', text)
                        payload = zipped.getvalue()
                        info = tarfile.TarInfo(stem + '.zip')
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                self.assertTrue(tarfile.is_tarfile(source))
                self.assertTrue(zipfile.is_zipfile(source))
                output = root / 'prepared'
                pipeline.prepare(SimpleNamespace(input=source, output=output, metadata=None,
                                                 encoding='utf-8', domains=None))
                manifest = json.loads((output / 'prepared_manifest.json').read_text())
                self.assertEqual(manifest['stats']['source_members'], 2)
                self.assertEqual(manifest['documents_by_decade_domain'],
                                 {'1820:fic': 1, '2010:tvm': 1})
                self.assertEqual([s['member'] for s in manifest['sources']],
                                 [f'coha-text.tar!{stem}.zip!{stem}.txt' for stem in texts])
                records = [record for _, record in pipeline.read_fragments(output / 'fragments.jsonl.gz')]
                self.assertEqual([r['doc_id'] for r in records], ['100', '101'])

    def test_conservative_normalization(self):
        text = 'café don’t Department.As &amp; well-\nknown <p>next /q/hello/q/'
        result = pipeline.normalize(text)
        self.assertIn("café don't Department.As & well-\nknown", result)
        self.assertIn('"hello"', result)
        self.assertNotIn('<p>', result)

    def test_ids_are_boundaries_and_gaps_are_not_ids(self):
        text = '@@100 water tank @@101 stone wall @@@@@@@@@@1820'
        docs = list(pipeline.documents(text, 'fallback'))
        self.assertEqual([doc_id for doc_id, _ in docs], ['100', '101'])
        parts = list(pipeline.fragments(docs[1][1]))
        self.assertEqual(parts[0]['text'], 'stone wall')
        self.assertEqual(parts[1]['text'], '1820')
        self.assertTrue(parts[0]['right_gap'])
        self.assertEqual(parts[1]['left_gap_length'], 10)

    def test_nested_archive_metadata_and_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zipped = io.BytesIO()
            with zipfile.ZipFile(zipped, 'w') as archive:
                archive.writestr('text_fic_1820.txt', '@@100 café water tank. @ @ @ @@101 New document.')
            source = root / 'coha.tar'
            with tarfile.open(source, 'w') as archive:
                payload = zipped.getvalue()
                info = tarfile.TarInfo('text_fic_1820.zip')
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            metadata = root / 'metadata.tsv'
            metadata.write_text('doc_id\tyear\tdomain\n100\t1821\tfic\n101\t1822\tfic\n')
            output = root / 'prepared'
            pipeline.prepare(SimpleNamespace(input=source, output=output, metadata=metadata,
                                               encoding='utf-8', domains=None))
            records = [record for _, record in pipeline.read_fragments(output / 'fragments.jsonl.gz')]
            self.assertEqual([r['doc_id'] for r in records], ['100', '101'])
            self.assertEqual([r['time'] for r in records], [1821, 1822])
            self.assertIn('café', records[0]['text'])
            self.assertTrue(records[0]['right_gap'])
            self.assertFalse(records[1]['left_gap'])
            manifest = json.loads((output / 'prepared_manifest.json').read_text())
            self.assertEqual(manifest['stats']['missing_token_markers'], 3)
            self.assertEqual(manifest['language_filter'], 'none')
            before = pipeline.digest_file(output / 'fragments.jsonl.gz')
            pipeline.prepare(SimpleNamespace(input=source, output=output, metadata=metadata,
                                               encoding='utf-8', domains=None))
            self.assertEqual(before, pipeline.digest_file(output / 'fragments.jsonl.gz'))


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        vocab = spacy.blank('en').vocab
        self.doc = Doc(vocab, words=['water', 'tank', 'maker', 'works', 'well', 'outside', 'today'],
                       lemmas=['water', 'tank', 'maker', 'work', 'well', 'outside', 'today'],
                       pos=['NOUN', 'NOUN', 'NOUN', 'VERB', 'ADV', 'ADV', 'ADV'],
                       heads=[1, 2, 3, 3, 3, 3, 3],
                       deps=['compound', 'compound', 'nsubj', 'ROOT', 'advmod', 'advmod', 'advmod'])
        self.record = {'doc_id': '100', 'fragment_id': 0, 'source': 'fic_2000_100.txt',
                       'time': 2000, 'domain': 'fic', 'text': self.doc.text, 'normalized_start': 0,
                       'left_gap': False, 'right_gap': False, 'left_gap_length': 0, 'right_gap_length': 0}

    def store_doc(self, store, record=None, doc=None):
        for name, columns in pipeline.schemas().items():
            store.create(name, columns)
        writer = io.StringIO()
        pipeline.add_document(store, doc or self.doc, record or self.record, writer, 'map')
        return [json.loads(line) for line in writer.getvalue().splitlines()]

    def test_document_counts_use_source_scoped_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CountStore(Path(directory) / 'counts.sqlite')
            try:
                first = dict(self.record, document_uid='source-one-100')
                second = dict(self.record, document_uid='source-two-100')
                self.store_doc(store, first)
                pipeline.add_document(store, self.doc, second, io.StringIO(), 'map')
                rows = pd.read_sql('SELECT * FROM bigram_document_counts', store.connection)
                self.assertEqual(set(rows.document_uid), {'source-one-100', 'source-two-100'})
                self.assertEqual(len(rows), 12)  # Six all-POS bigrams per document.
                self.assertTrue((rows['count'] == 1).all())
            finally:
                store.close()

    def test_local5_pair_boundaries_sentence_support_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CountStore(Path(directory) / 'counts.sqlite')
            try:
                self.store_doc(store)
                pairs = pd.read_sql('SELECT * FROM bigram_counts', store.connection)
                self.assertEqual(pairs['count'].sum(), 6)  # All POS bigrams, including verbs/adverbs.
                words = pd.read_sql('SELECT * FROM unigram_counts', store.connection)
                self.assertEqual(words['count'].sum(), sum(pipeline.lexical(t) for t in self.doc))
                local = pd.read_sql('SELECT * FROM bigrams_local5', store.connection)
                contexts = set(local.loc[local['head'] == 'tank_NOUN', 'context'])
                self.assertEqual(contexts, {'maker_NOUN', 'work_VERB', 'well_ADV'})
                full = pd.read_sql('SELECT * FROM bigrams_sentence', store.connection)
                self.assertIn('today_ADV', set(full.context))
                for mode in pipeline.MODES:
                    coverage = pd.read_sql(f'SELECT * FROM bigram_coverage_{mode}', store.connection)
                    self.assertEqual(coverage['count'].sum(), 6)
                for mode in pipeline.MODES:
                    pipeline.canonical_inputs(store, Path(directory) / mode, mode, {})
                for mode in pipeline.MODES:
                    self.assertTrue((Path(directory) / mode / 'bigram_counts.parquet').exists())
                    self.assertFalse((Path(directory) / mode / 'compounds.pkl').exists())
            finally:
                store.close()

    def test_all_pos_targets_without_punctuation_gap_or_sentence_bridging(self):
        doc = Doc(self.doc.vocab,
                  words=['pepper', 'the', 'soup', ',', 'pepper', '@', 'pot', '.', 'salt'],
                  lemmas=['pepper', 'the', 'soup', ',', 'pepper', '@', 'pot', '.', 'salt'],
                  pos=['VERB', 'DET', 'NOUN', 'PUNCT', 'NOUN', 'SYM', 'NOUN', 'PUNCT', 'NOUN'],
                  sent_starts=[True, False, False, False, False, False, False, False, True])
        with tempfile.TemporaryDirectory() as directory:
            store = CountStore(Path(directory) / 'counts.sqlite')
            try:
                self.store_doc(store, doc=doc)
                unigrams = pd.read_sql('SELECT * FROM unigram_counts', store.connection)
                self.assertEqual(set(unigrams.word), {'pepper_VERB', 'the_DET', 'soup_NOUN', 'pepper_NOUN', 'pot_NOUN', 'salt_NOUN'})
                bigrams = pd.read_sql('SELECT * FROM bigram_counts', store.connection)
                self.assertEqual(set(zip(bigrams.modifier, bigrams['head'])),
                                 {('pepper_VERB', 'the_DET'), ('the_DET', 'soup_NOUN')})
                words = pd.read_sql('SELECT * FROM unigrams_local5', store.connection)
                self.assertIn('pepper_VERB', set(words.word))
                self.assertIn('the_DET', set(words.word))
                self.assertNotIn('the_DET', set(words.context))
                self.assertNotIn('@_SYM', set(words.word) | set(words.context))
                self.assertNotIn('word_counts', store.schemas)
                self.assertNotIn('pair_counts', store.schemas)
            finally:
                store.close()

    def test_local_events_equal_shortest_3_4_5gram_extraction(self):
        expected = Counter()
        tokens = list(self.doc)
        for width in [3, 4, 5]:
            for start in range(len(tokens) - width + 1):
                window = tokens[start:start + width]
                for first, second, context, side in [(window[0], window[1], window[-1], 'R'),
                                                     (window[-2], window[-1], window[0], 'L')]:
                    if first.pos_ in {'ADJ', 'NOUN'} and second.pos_ == 'NOUN' and context.pos_ in pipeline.CONTEXT_POS:
                        expected[(pipeline.lemma(first, 'map'), pipeline.lemma(second, 'map'),
                                  pipeline.lemma(context, 'map'), side, width - 2)] += 1
        actual = Counter()
        for first, second in zip(tokens, tokens[1:]):
            if first.pos_ in {'ADJ', 'NOUN'} and second.pos_ == 'NOUN':
                for context, side, distance in pipeline.context_events(self.doc[:], first.i, second.i + 1, 'local5', 'map'):
                    actual[(pipeline.lemma(first, 'map'), pipeline.lemma(second, 'map'), context, side, distance)] += 1
        self.assertEqual(actual, expected)

    def test_gap_incomplete_sentence_keeps_counts_and_local_context(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CountStore(Path(directory) / 'counts.sqlite')
            try:
                saved = self.store_doc(store, dict(self.record, right_gap=True, right_gap_length=10))
                self.assertFalse(saved[0]['sentence_complete'])
                self.assertEqual(store.connection.execute('SELECT SUM(count) FROM bigram_counts').fetchone()[0], 6)
                self.assertEqual(store.connection.execute('SELECT COUNT(*) FROM bigrams_sentence').fetchone()[0], 0)
                self.assertGreater(store.connection.execute('SELECT COUNT(*) FROM bigrams_local5').fetchone()[0], 0)
                self.assertEqual(store.connection.execute("SELECT SUM(count) FROM bigram_coverage_sentence WHERE status='gap_incomplete'").fetchone()[0], 6)
                total = store.connection.execute('SELECT SUM(count) FROM bigram_opportunities_local5').fetchone()[0]
                self.assertEqual(total, 36)
            finally:
                store.close()

    def test_explicit_sentence_end_before_gap_is_usable(self):
        vocab = self.doc.vocab
        doc = Doc(vocab, words=['water', 'tank', 'works', '.'], lemmas=['water','tank','work','.'],
                  pos=['NOUN','NOUN','VERB','PUNCT'], heads=[1,2,2,2], deps=['compound','nsubj','ROOT','punct'])
        with tempfile.TemporaryDirectory() as directory:
            store = CountStore(Path(directory) / 'counts.sqlite')
            try:
                saved = self.store_doc(store, dict(self.record, right_gap=True), doc)
                self.assertTrue(saved[0]['sentence_complete'])
                self.assertEqual(store.connection.execute('SELECT SUM(count) FROM bigrams_sentence').fetchone()[0], 2)
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
