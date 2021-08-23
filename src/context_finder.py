import pandas as pd
from fastparquet import write
import fastparquet
import glob
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import time
from os.path import isfile

num_partitions=200
all_context_files=[]
from_path='/data/dharp/compounds/datasets/entire_df/'
to_path='/data/dharp/compounds/datasets/contexts/'
for filename in glob.glob(from_path+'*parq'):
    all_context_files.append(filename)
    

def context_maker(df):
    df[['w1','w2','w3','w4','w5']]=df.lemma_sent.str.split(" ",expand=True)
    df[['p1','p2','p3','p4','p5']]=df.pos_sent.str.split(" ",expand=True)
    w1_df=df.groupby(['w1','p1','year'])['count'].sum().to_frame()
    w1_df.reset_index(inplace=True)
    w1_df.columns=['word','pos','year','count']

    w2_df=df.groupby(['w2','p2','year'])['count'].sum().to_frame()
    w2_df.reset_index(inplace=True)
    w2_df.columns=['word','pos','year','count']

    w3_df=df.groupby(['w3','p3','year'])['count'].sum().to_frame()
    w3_df.reset_index(inplace=True)
    w3_df.columns=['word','pos','year','count']

    w4_df=df.groupby(['w4','p4','year'])['count'].sum().to_frame()
    w4_df.reset_index(inplace=True)
    w4_df.columns=['word','pos','year','count']

    w5_df=df.groupby(['w5','p5','year'])['count'].sum().to_frame()
    w5_df.reset_index(inplace=True)
    w5_df.columns=['word','pos','year','count']

    context_df=pd.concat([w1_df,w2_df,w3_df,w4_df,w5_df],ignore_index=True,sort=False)
    context_df=context_df.groupby(['word','pos','year'])['count'].sum().to_frame()
    context_df.reset_index(inplace=True)
    context_df=context_df.loc[context_df.pos.isin(['NOUN','ADJ','VERB','ADV'])]
    return context_df



for f,cur_parq_file in enumerate(all_context_files):
    cur_parq_dfs=[]
    print(f'File num: {f+1} name: {cur_parq_file}')
    cur_parq=fastparquet.ParquetFile(cur_parq_file)
    print(f'Number of groups: {len(cur_parq.row_groups)}')
    for i,df in enumerate(cur_parq.iter_row_groups()):
        print(f'Group {i+1}')
        cur_time=time.time()

        df_split = np.array_split(df, num_partitions)
        pool = Pool(num_partitions)
        print('Started parallelization')
        results=pool.map_async(context_maker,df_split)
        pool.close()
        pool.join()
        print('Done parallelization')
       
        curr_df_list=results.get()
        context_df=pd.concat(curr_df_list,ignore_index=True)
        print(context_df.shape[0])
        context_df=context_df.groupby(['word','pos','year'])['count'].sum().to_frame()
        context_df.reset_index(inplace=True)
        print(f'Total time taken for Group {i+1}: {round(time.time()-cur_time)} secs')        
        print(context_df.shape[0])
        cur_parq_dfs.append(context_df)
    
    parq_context_df=pd.concat(cur_parq_dfs,ignore_index=True)
    parq_context_df=parq_context_df.groupby(['word','pos','year'])['count'].sum().to_frame()
    parq_context_df.to_csv(f'{to_path}/no_ner_{f+1}.csv',sep="\t")