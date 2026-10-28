"""Small, corpus-free regression tests for the representation builder."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np
import pyarrow.parquet as pq

SPEC = importlib.util.spec_from_file_location('build_representations',Path(__file__).resolve().parents[1]/'build_representations.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def write_csv(path,columns,rows):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.contexts=self.root/'contexts.csv'
        self.counts=self.root/'counts.csv'
        write_csv(self.contexts,['modifier','head','decade','context','count'],[
            ['oil_n','field_n',1980,'land_NOUN',3],
            ['oil_n','field_n',1990,'land_NOUN',2],
            ['oil_n','field_n',1990,'energy_NOUN',4],
            ['gas_n','plant_n',1990,'energy_NOUN',5],
            ['gas_n','plant_n',1990,'green_ADJ',1],
            ['gas_n','field_n',1990,'land_NOUN',4],
            ['gas_n','field_n',1990,'green_ADJ',3],
            ['oil_n','plant_n',2000,'FUTURE_ONLY_NOUN',99999],
        ])
        write_csv(self.counts,['modifier','head','decade','count'],[
            ['oil_n','field_n',1980,1],['oil_n','field_n',1990,1],
            ['oil_n','field_n',2000,100], # early rare occurrence prevents novelty
            ['gas_n','plant_n',1990,5],['gas_n','field_n',1990,5],
            ['oil_n','plant_n',2000,3],
            ['new_n','item_n',2000,2], # not enough novel-period occurrences
            ['old_n','item_n',1750,1],['old_n','item_n',2000,5],
        ])

    def tearDown(self):
        self.directory.cleanup()

    def args(self,*extra):
        return builder.parser().parse_args([
            '--pair-contexts',str(self.contexts),'--pair-counts',str(self.counts),
            '--output',str(self.root/'output'),'--dimensions','2',
            '--batch-size','2','--threads','1','--aggregation-partitions','4',
            '--aggregation-batch-size','20',*extra])

    def run_builder(self,args):
        with contextlib.redirect_stdout(io.StringIO()):
            return builder.run(args)

    def test_source_record_identifies_the_corpus_manifest(self):
        manifest = self.root / 'corpus_manifest.json'
        manifest.write_text('{}')
        record = builder.source_record(self.contexts)
        self.assertEqual(record['directory_manifest'], str(manifest.resolve()))
        self.assertEqual(record['path'], str(self.contexts.resolve()))
        self.assertEqual(record['bytes'], self.contexts.stat().st_size)

    def test_shared_basis_future_exclusion_and_true_occurrence_novelty(self):
        manifest=self.run_builder(self.args())
        output=self.root/'output'
        with (output/'contexts.csv').open() as stream:
            contexts=list(csv.DictReader(stream))
        self.assertNotIn('FUTURE_ONLY_NOUN',{r['context'] for r in contexts})
        with (output/'novel_compounds.csv').open() as stream:
            self.assertEqual(list(csv.DictReader(stream)),[{'modifier':'oil_NOUN','head':'plant_NOUN'}])
        with (output/'compound_metadata.csv').open() as stream:
            metadata={(r['modifier'],r['head']):r for r in csv.DictReader(stream)}
        self.assertEqual(metadata['oil_NOUN','plant_NOUN']['has_embedding'],'false')
        self.assertEqual(metadata['oil_NOUN','field_NOUN']['novel'],'false')
        self.assertEqual(metadata['old_NOUN','item_NOUN']['novel'],'false')
        with (output/'historical_compounds.csv').open() as stream:
            historical={(r['modifier'],r['head']) for r in csv.DictReader(stream)}
        self.assertIn(('old_NOUN','item_NOUN'),historical)
        self.assertFalse((output/'compounds.csv').exists())
        # Directly reconstruct one role vector with the saved shared components.
        heads=pq.read_table(output/'dense_embeddings_heads.parquet').to_pydict()
        basis=np.load(output/'svd_basis.npz')['components']
        context_order={r['context']:int(r['context_id']) for r in contexts}
        raw=np.zeros(len(context_order),dtype=np.float32)
        raw[context_order['land_NOUN']]=9 # oil field 3+2, gas field4
        raw[context_order['energy_NOUN']]=4
        raw[context_order['green_ADJ']]=3
        expected=basis@raw
        expected/=np.linalg.norm(expected)
        row=heads['head'].index('field_NOUN')
        np.testing.assert_allclose([heads['dim_0'][row],heads['dim_1'][row]],expected,atol=1e-6)
        self.assertEqual(manifest['parameters']['dimensions'],2)
        self.assertEqual(manifest['schema_version'],2)
        self.assertEqual(manifest['code_sha256'],builder.implementation_identity())
        self.assertTrue({'source_info.py','io.py','progress.py','partitioned_parquet.py'}.issubset(manifest['code_sha256']))

    def test_temporal_rows_share_basis_and_exclude_future(self):
        self.run_builder(self.args('--temporal'))
        table=pq.read_table(self.root/'output'/'dense_embeddings_heads.parquet').to_pydict()
        rows=set(zip(table['head'],table['decade']))
        self.assertIn(('field_NOUN',1980),rows)
        self.assertIn(('field_NOUN',1990),rows)
        self.assertFalse(any(decade>=2000 for _,decade in rows))

    def test_optional_sklearn_solver_still_exports_the_shared_basis(self):
        manifest=self.run_builder(self.args('--svd-solver','sklearn'))
        self.assertEqual(manifest['svd_implementation'],'sklearn_randomized')
        basis=np.load(self.root/'output'/'svd_basis.npz')['components']
        self.assertEqual(basis.shape[0],2)

    def test_constituents_only_omits_compound_rows_and_uses_content_contexts(self):
        with self.contexts.open('a',newline='') as stream:
            csv.writer(stream).writerow(['oil_n','field_n',1990,'the_DET',999999])
        manifest=self.run_builder(self.args('--constituents-only'))
        self.assertNotIn('compounds',manifest['outputs'])
        self.assertFalse((self.root/'output'/'dense_embeddings_compounds.parquet').exists())
        with (self.root/'output'/'contexts.csv').open() as stream:
            self.assertNotIn('the_DET',{r['context'] for r in csv.DictReader(stream)})

    def test_context_agnostic_and_raw_v3_schema(self):
        import pyarrow as pa
        raw=self.root/'raw.parquet'
        pq.write_table(pa.table({
            'modifier_norm':['oil','gas','gas'], 'modifier_pos':['NOUN']*3,
            'head_norm':['field','plant','field'], 'head_pos':['NOUN']*3,
            'context_norm':['land','energy','green'],
            'context_pos':['NOUN','NOUN','ADJ'],
            'decade':[1990]*3,'count':[3,5,4]}),raw)
        word=self.root/'words.csv'
        write_csv(word,['word','time','context','count'],[
            ['oil_NOUN',1990,'land_NOUN',10],['gas_NOUN',1990,'green_ADJ',20],
            ['plant_NOUN',1990,'energy_NOUN',30],['field_NOUN',1990,'land_NOUN',40],
            ['field_NOUN',2000,'future_NOUN',99999],
            ['new_NOUN',1990,'land_NOUN',100], # future-only modifier role must not select this word
        ])
        manifest=self.run_builder(self.args('--pair-contexts',str(raw),'--word-contexts',str(word)))
        self.assertEqual(manifest['constituent_mode'],'word-context-agnostic')
        mods=pq.read_table(self.root/'output'/'dense_embeddings_modifiers.parquet')
        self.assertEqual(mods.num_rows,2)
        self.assertNotIn('new_NOUN',mods.column('modifier').to_pylist())

    def test_dimension_request_is_not_silently_reduced(self):
        with self.assertRaisesRegex(ValueError,'Requested 300 dimensions'):
            self.run_builder(self.args('--dimensions','300'))

    def test_exactly_300_output_dimensions(self):
        from threadpoolctl import threadpool_limits
        rows=[]
        counts=[]
        for i in range(302):
            modifier=f'm{i}_NOUN'
            head=f'h{i}_NOUN'
            counts.append([modifier,head,1990,5])
            rows.extend([[modifier,head,1990,f'c{i}_NOUN',5],
                         [modifier,head,1990,f'c{(i+1)%302}_NOUN',2]])
        write_csv(self.contexts,['modifier','head','decade','context','count'],rows)
        write_csv(self.counts,['modifier','head','decade','count'],counts)
        with threadpool_limits(limits=1):
            self.run_builder(self.args('--dimensions','300','--batch-size','100'))
        for role in ['heads','modifiers','compounds']:
            table=pq.read_table(self.root/'output'/f'dense_embeddings_{role}.parquet')
            self.assertEqual(len([name for name in table.column_names if name.startswith('dim_')]),300)

    def test_negative_counts_are_rejected(self):
        write_csv(self.counts,['modifier','head','decade','count'],[['oil_n','field_n',1990,-1]])
        with self.assertRaisesRegex(ValueError,'invalid lexical keys, dates'):
            self.run_builder(self.args())

    def test_ppmi_remains_sparse_and_has_expected_values(self):
        from scipy.sparse import csr_matrix
        original=csr_matrix(np.array([[4,0],[0,4]],dtype=np.float32))
        weighted=builder.weight_matrix(original,'ppmi')
        self.assertEqual(weighted.nnz,2)
        np.testing.assert_allclose(weighted.data,np.log(2))


if __name__=='__main__':
    unittest.main()
