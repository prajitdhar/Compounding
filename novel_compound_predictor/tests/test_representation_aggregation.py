"""Corpus-free checks of bounded aggregation and representation semantics."""
from collections import Counter
import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import duckdb

import build_representations as builder
from representation_aggregation import BoundedAggregation


class BoundedAggregationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.connection=duckdb.connect()
        self.connection.execute("SET threads=1")
        self.connection.execute("SET memory_limit='64MB'")
        self.connection.execute("SET preserve_insertion_order=false")

    def tearDown(self):
        self.connection.close()
        self.directory.cleanup()

    def aggregate(self,**kwargs):
        return BoundedAggregation(self.connection,self.root/'work',partitions=4,
                                  batch_size=7,buffer_mb=1,**kwargs)

    def test_projection_is_scanned_once_then_bucket_sums_match(self):
        # A Python scalar function counts actual projected source rows, unlike
        # SQL text counting, which would also count harmless schema binding.
        observed=[]
        self.connection.create_function('seen',lambda x: observed.append(x) or x,
                                        ['BIGINT'],'BIGINT',side_effects=True)
        aggregate=self.aggregate()
        source=aggregate.shuffle("""SELECT seen(i) AS key,i%3 AS context,
            1::BIGINT AS count FROM range(103) t(i)""",['key'],'shuffle')
        self.assertEqual(len(observed),103)
        result=aggregate.reduce(source,"""SELECT key,context,SUM(count)::BIGINT AS count
            FROM __INPUT__ GROUP BY key,context""",'reduced')
        actual=self.connection.execute(result.query()+' ORDER BY key').fetchall()
        self.assertEqual(actual,[(i,i%3,1) for i in range(103)])
        self.assertEqual(len(observed),103)
        self.assertGreater(len(result.buckets),1)

    def test_duplicate_keys_across_arrow_batches_are_summed_exactly(self):
        aggregate=self.aggregate()
        source=aggregate.shuffle("""SELECT i%5 AS key,i%3 AS context,
            1::BIGINT AS count FROM range(500) t(i)""",['key'],'shuffle')
        result=aggregate.reduce(source,"""SELECT key,context,SUM(count)::BIGINT AS count
            FROM __INPUT__ GROUP BY key,context""",'reduced')
        actual={tuple(row[:2]):row[2] for row in self.connection.execute(result.query()).fetchall()}
        self.assertEqual(actual,Counter((i%5,i%3) for i in range(500)))

    def test_oom_rehashes_only_failed_bucket_and_preserves_results(self):
        aggregate=self.aggregate()
        source=aggregate.shuffle("""SELECT i%11 AS key,i%3 AS context,
            1::BIGINT AS count FROM range(500) t(i)""",['key'],'shuffle',partitions=1)
        copy=aggregate._copy_reduction
        attempts=[]

        def fail_once(sql,path):
            attempts.append(path)
            if len(attempts)==1:
                raise duckdb.OutOfMemoryException('simulated bucket memory pressure')
            return copy(sql,path)

        with mock.patch.object(aggregate,'_copy_reduction',side_effect=fail_once):
            result=aggregate.reduce(source,"""SELECT key,context,SUM(count)::BIGINT AS count
                FROM __INPUT__ GROUP BY key,context""",'reduced')
        self.assertGreater(len(attempts),2)
        actual={tuple(row[:2]):row[2] for row in self.connection.execute(result.query()).fetchall()}
        self.assertEqual(actual,Counter((i%11,i%3) for i in range(500)))
        self.assertFalse(any(self.root.rglob('split')))

    def test_oom_depth_limit_reports_actionable_failure(self):
        aggregate=self.aggregate(max_split_depth=0)
        source=aggregate.shuffle('SELECT 1 AS key,1 AS count',['key'],'shuffle')
        with mock.patch.object(aggregate,'_copy_reduction',
                               side_effect=duckdb.OutOfMemoryException('simulated')):
            with self.assertRaisesRegex(RuntimeError,'aggregation-partitions'):
                aggregate.reduce(source,'SELECT key,SUM(count) AS count FROM __INPUT__ GROUP BY key','reduced')

    def test_join_oom_rehashes_both_sides_and_preserves_missing_matches(self):
        aggregate=self.aggregate()
        left=aggregate.shuffle('SELECT i AS key FROM range(20) t(i)',['key'],'left',partitions=1)
        right=aggregate.shuffle('SELECT i AS key FROM range(0,20,2) t(i)',['key'],'right',partitions=1)
        copy=aggregate._copy_reduction
        attempts=[]

        def fail_once(sql,path):
            attempts.append(path)
            if len(attempts)==1:
                raise duckdb.OutOfMemoryException('simulated join memory pressure')
            return copy(sql,path)

        with mock.patch.object(aggregate,'_copy_reduction',side_effect=fail_once):
            result=aggregate.combine(left,right,"""SELECT l.key,r.key IS NOT NULL AS matched
                FROM __LEFT__ l LEFT JOIN __RIGHT__ r USING(key)""",'joined')
        self.assertGreater(len(attempts),2)
        self.assertEqual(self.connection.execute(result.query()+' ORDER BY key').fetchall(),
                         [(i,i%2==0) for i in range(20)])

    def test_join_rejects_different_partition_layouts(self):
        aggregate=self.aggregate()
        left=aggregate.shuffle('SELECT 1 AS key',['key'],'left',partitions=1)
        right=aggregate.shuffle('SELECT 1 AS key',['key'],'right',partitions=4)
        with self.assertRaisesRegex(ValueError,'hash routing'):
            aggregate.combine(left,right,
                'SELECT l.key FROM __LEFT__ l LEFT JOIN __RIGHT__ r USING(key)','joined')


class PreparationEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.root=Path(self.directory.name)
        self.contexts=[
            ('a_NOUN','x_NOUN',1980,'land_NOUN',3),
            ('a_NOUN','x_NOUN',1980,'land_NOUN',4), # duplicate across batches
            ('a_NOUN','x_NOUN',1990,'grow_VERB',5),
            ('a_NOUN','x_NOUN',1990,'the_DET',100000),
            ('b_NOUN','y_NOUN',1980,'grow_VERB',30),
            ('b_NOUN','y_NOUN',1990,'land_NOUN',1),
            ('a_NOUN','y_NOUN',2000,'future_NOUN',99999),
        ]
        self.counts=[
            ('a_NOUN','x_NOUN',1750,1),
            ('a_NOUN','x_NOUN',1980,100),('a_NOUN','x_NOUN',1990,100),
            ('b_NOUN','y_NOUN',1980,1),('b_NOUN','y_NOUN',2000,50),
            ('a_NOUN','y_NOUN',2000,3),('c_NOUN','z_NOUN',2000,3),
        ]
        self.context_path=self.root/'contexts.csv'
        self.count_path=self.root/'counts.csv'
        self.write(self.context_path,['modifier','head','decade','context','count'],self.contexts)
        self.write(self.count_path,['modifier','head','decade','count'],self.counts)
        self.connection=duckdb.connect()
        self.connection.execute('SET threads=1')
        self.connection.execute("SET memory_limit='64MB'")
        self.connection.execute('SET preserve_insertion_order=false')

    def tearDown(self):
        self.connection.close()
        self.directory.cleanup()

    def write(self,path,columns,rows):
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def arguments(self,temporal=False,max_contexts=2):
        return builder.parser().parse_args([
            '--pair-contexts',str(self.context_path),'--pair-counts',str(self.count_path),
            '--output',str(self.root/'output'),'--min-compound-count','200',
            '--aggregation-partitions','4','--aggregation-batch-size','3',
            '--aggregation-buffer-mb','1','--max-contexts',str(max_contexts),
            *(['--temporal'] if temporal else []),
        ])

    def reference(self,temporal,max_contexts):
        eligible=Counter()
        past_modifiers,past_heads=set(),set()
        for modifier,head,decade,count in self.counts:
            if decade<2000 and count>0:
                past_modifiers.add(modifier)
                past_heads.add(head)
            if 1800<=decade<2000:
                eligible[modifier,head]+=count
        events=Counter()
        for modifier,head,decade,context,count in self.contexts:
            if not 1800<=decade<2000 or not context.endswith(('_NOUN','_VERB','_ADJ','_ADV')):
                continue
            period=decade if temporal else -1
            if eligible[modifier,head]>=200:
                events['compound',modifier,head,period,context]+=count
            if modifier in past_modifiers:
                events['modifier',modifier,'',period,context]+=count
            if head in past_heads:
                events['head','',head,period,context]+=count
        frequencies=Counter()
        for key,count in events.items():
            frequencies[key[-1]]+=count
        contexts=sorted(frequencies,key=lambda context:(-frequencies[context],context))[:max_contexts]
        return {key:count for key,count in events.items() if key[-1] in contexts}

    def verify(self,temporal,max_contexts):
        builder.prepare_tables(self.connection,self.arguments(temporal,max_contexts),
                               work_directory=self.root/'partitions')
        actual=self.connection.execute("""SELECT e.kind,e.modifier,e.head,e.decade,c.context,e.count
            FROM events e JOIN contexts c USING(context_id)""").fetchall()
        self.assertEqual({tuple(row[:5]):row[5] for row in actual},self.reference(temporal,max_contexts))
        rows=self.connection.execute('SELECT kind,modifier,head,decade FROM rows').fetchall()
        self.assertEqual(len(rows),len(set(rows)))
        # The large pair-context GROUP BY is gone; duplicate raw rows remain
        # safely on disk until retained target/context bucket reduction.
        self.assertEqual(self.connection.execute('SELECT count(*) FROM pair_contexts').fetchone()[0],6)
        historical=dict(((m,h),count) for m,h,count in
                        self.connection.execute('SELECT * FROM historical_compounds').fetchall())
        training=dict(((m,h),count) for m,h,count in
                      self.connection.execute('SELECT * FROM training_compounds').fetchall())
        self.assertEqual(historical['a_NOUN','x_NOUN'],201)
        self.assertEqual(training['a_NOUN','x_NOUN'],200)
        metadata={(row[0],row[1]):row for row in
                  self.connection.execute('SELECT * FROM compound_metadata').fetchall()}
        self.assertEqual(metadata['a_NOUN','x_NOUN'][2],1750)
        self.assertFalse(metadata['a_NOUN','x_NOUN'][-2]) # rare early attestation prevents novelty
        self.assertTrue(metadata['a_NOUN','x_NOUN'][-1]) # compound has retained contexts
        self.assertTrue(metadata['a_NOUN','y_NOUN'][-2])
        self.assertFalse(metadata['a_NOUN','y_NOUN'][-1])
        self.assertFalse(metadata['b_NOUN','y_NOUN'][-1]) # threshold applies only to compound rows

    def test_pooled_counts_match_unpartitioned_reference(self):
        self.verify(False,2)

    def test_temporal_counts_match_unpartitioned_reference(self):
        self.verify(True,2)

    def test_top_context_filtering_before_target_aggregation_preserves_ranking(self):
        self.verify(False,1)
        self.assertEqual(self.connection.execute('SELECT context FROM contexts').fetchall(),[('grow_VERB',)])

    def test_unknown_pair_is_rejected_even_if_context_would_be_filtered(self):
        self.contexts.append(('unknown_NOUN','pair_NOUN',1990,'the_DET',1))
        self.write(self.context_path,['modifier','head','decade','context','count'],self.contexts)
        with self.assertRaisesRegex(ValueError,'1 context pairs have no matching occurrence count'):
            builder.prepare_tables(self.connection,self.arguments(),work_directory=self.root/'partitions')


if __name__=='__main__':
    unittest.main()
