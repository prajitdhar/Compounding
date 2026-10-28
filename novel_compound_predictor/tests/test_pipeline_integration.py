"""Actual-300d, corpus-free builder -> PyTorch training -> ranking integration."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import torch


ROOT=Path(__file__).resolve().parents[1]


def load_module(name,filename):
    import sys
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


builder=load_module('integration_builder','build_representations.py')
predictor=load_module('integration_predictor','predict_novel_compounds.py')
visualization=load_module('integration_visualization','export_visualization_dataset.py')


def write_csv(path,columns,rows):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.contexts=self.root/'contexts.csv'
        self.counts=self.root/'counts.csv'
        rows=[]
        counts=[]
        for i in range(302):
            modifier=f'm{i:03}_NOUN'
            head=f'h{i:03}_NOUN'
            for decade in [1980,1990]:
                counts.append([modifier,head,decade,5])
                rows.extend([[modifier,head,decade,f'c{i:03}_NOUN',5],
                             [modifier,head,decade,f'c{(i+1)%302:03}_NOUN',2]])
        self.novel=[(f'm{i:03}_NOUN',f'h{i+1:03}_NOUN') for i in range(3)]
        for modifier,head in self.novel:
            counts.append([modifier,head,2000,3])
            rows.append([modifier,head,2000,'FUTURE_ONLY_NOUN',999999])
        write_csv(self.contexts,['modifier','head','decade','context','count'],rows)
        write_csv(self.counts,['modifier','head','decade','count'],counts)
        self.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.directory.cleanup()

    def run_pipeline(self,encoder):
        output=self.root/'representations'
        args=builder.parser().parse_args([
            '--pair-contexts',str(self.contexts),'--pair-counts',str(self.counts),
            '--output',str(output),'--year-start','1980','--threads','1',
            '--batch-size','100','--aggregation-partitions','4',
            '--matrix-storage','disk',*(['--temporal'] if encoder=='lstm' else []),
        ])
        model_output=self.root/'model'
        with contextlib.redirect_stdout(io.StringIO()),threadpool_limits(limits=1):
            manifest=builder.run(args)
            mods=output/'dense_embeddings_modifiers.parquet'
            heads=output/'dense_embeddings_heads.parquet'
            for path in [mods,heads]:
                columns=pq.read_schema(path).names
                self.assertEqual(len([c for c in columns if c.startswith('dim_')]),300)
            self.assertEqual(manifest['time_window']['end_exclusive'],2000)
            training=predictor.parser().parse_args([
                'train','--modifiers',str(mods),'--heads',str(heads),
                '--positives',str(output/'novel_compounds.csv'),
                '--attested',str(output/'attested_compounds.csv'),
                '--output-dir',str(model_output),'--encoder',encoder,
                '--epochs','2','--batch-size','2','--start-year','1980',
                '--device','cpu',
                # Input/output embeddings remain 300d; a small LSTM hidden state
                # keeps this synthetic integration test quick on ordinary CPUs.
                *(['--lstm-hidden-size','16'] if encoder=='lstm' else []),
            ])
            metrics=predictor.train(training)
            self.assertEqual(metrics['positive_pairs_used'],3)
            self.assertEqual(metrics['positive_pairs_missing_historical_constituents'],0)
            self.assertEqual(metrics['model_config']['dimensions'],300)
            self.assertIsNone(metrics['embedding_source_info_warning'])
            examples=pd.read_csv(model_output/'split_examples.csv')
            self.assertEqual(set(examples.split),{'train','validation','test'})
            self.assertFalse(examples.duplicated(['modifier','head']).any())
            actual_attested=set(predictor.read_pairs(output/'compound_counts.csv'))
            negatives=set(examples.loc[examples.label.eq(0),['modifier','head']].itertuples(index=False,name=None))
            self.assertFalse(negatives&actual_attested)
            candidates=self.root/'candidates.csv'
            historical=('m000_NOUN','h000_NOUN')
            write_csv(candidates,['modifier','head'],[
                *self.novel,historical,('missing_NOUN','h000_NOUN')])
            ranking=predictor.parser().parse_args([
                'rank','--checkpoint',str(model_output/'model.pt'),
                '--modifiers',str(mods),'--heads',str(heads),
                '--attested',str(output/'historical_compounds.csv'),
                '--candidates',str(candidates),'--top-k','3','--batch-size','2',
                '--device','cpu','--output',str(self.root/'ranking.csv'),
            ])
            result=predictor.rank(ranking)
            if encoder=='pooled':
                visual_path=self.root/'compounds.csv'
                visual_manifest=visualization.main([
                    '--heads',str(heads),'--modifiers',str(mods),
                    '--compounds',str(output/'dense_embeddings_compounds.parquet'),
                    '--novel-pairs',str(output/'novel_compounds.csv'),
                    '--output',str(visual_path),'--batch-size','100',
                ])
                self.assertEqual(visual_manifest['counts']['additive_novel_rows'],3)
                self.assertEqual(visual_manifest['counts']['missing_novel_rows'],0)
                visual=pd.read_csv(visual_path,sep='\t')
                self.assertEqual(visual.columns.tolist(),['common','novel',*[str(i) for i in range(300)]])
                novel=visual.loc[visual.novel.eq(1)].set_index('common')
                self.assertEqual(set(novel.index),{f'm{i:03} h{i+1:03}' for i in range(3)})
                modifier_table=pq.read_table(mods).to_pandas().set_index('modifier')
                head_table=pq.read_table(heads).to_pandas().set_index('head')
                # Additive visualization vectors preserve all 300 coordinates
                # and keep their sum, with no second normalization.
                expected=(modifier_table.loc['m000_NOUN'].to_numpy(dtype=np.float64)
                          +head_table.loc['h001_NOUN'].to_numpy(dtype=np.float64))
                np.testing.assert_allclose(novel.loc['m000 h001'].drop('novel').to_numpy(dtype=float),
                                           expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(result['scored'],3)
        self.assertEqual(result['excluded_attested'],1)
        self.assertEqual(result['missing_constituents'],1)
        ranked=pd.read_csv(self.root/'ranking.csv')
        self.assertEqual(set(ranked[['modifier','head']].itertuples(index=False,name=None)),set(self.novel))
        self.assertTrue(np.isfinite(ranked.score).all())
        self.assertTrue(ranked.score.is_monotonic_decreasing)

    def test_pooled_300d_build_train_rank(self):
        self.run_pipeline('pooled')

    def test_temporal_300d_build_lstm_train_rank(self):
        self.run_pipeline('lstm')


if __name__=='__main__':
    unittest.main()
