import pandas as pd
import glob
import os
import numpy as np
import pickle as pkl
import time
import dask.dataframe as dd
from fastparquet import write
import fastparquet


all_pkl_files=[]
pkl_file_sizes=[]
path='/data/dharp/compounds/datasets/googleV3/'
for filename in glob.glob(path+'*pkl'):
    pkl_file_sizes.append(os.path.getsize(filename))
    all_pkl_files.append(filename)
    
    
pkl_df=pd.DataFrame(all_pkl_files,pkl_file_sizes)
pkl_df.reset_index(inplace=True)
pkl_df.columns=['fsize','fname']
pkl_df['fsize_perc']=pkl_df.fsize/pkl_df.fsize.sum()*100
pkl_df.sort_values(by=['fsize'],ascending=False,inplace=True,ignore_index=True)
pkl_df.fsize/=1024*1024



maxvalue = 30_000

lastvalue = 0
newcum = []
labels=[]
cur_label=1
for row in pkl_df.itertuples():
    thisvalue =  row.fsize + lastvalue
    if thisvalue > maxvalue:
        thisvalue = 0
        cur_label+=1
    newcum.append( thisvalue )
    labels.append(cur_label)
    lastvalue = thisvalue
pkl_df['fcat']=labels


def write_to_parquet(data_bin):
    print(data_bin.iloc[0].fcat)
    cur_time=time.time()
    df_list=[]
    print(data_bin.shape[0])
    for row in data_bin.itertuples():
        #print(row.fname)
        cur_df=pd.read_pickle(row.fname)
        cur_df.reset_index(inplace=True)
        df_list.append(cur_df)
    concat_df=pd.concat(df_list,ignore_index=True,sort=False)
    print(concat_df.shape[0])
    total_df_shape=concat_df.shape[0]
    print('Done concatenating')
    
    ddf = dd.from_pandas(concat_df, npartitions=30)
    
    ddf=ddf.groupby(['lemma_pos','pos_sent','year','comp_class'])['count'].sum()
    print('Done grouping')
    ddf=ddf.to_frame().reset_index().compute()
    
    print(ddf.shape[0])
    after_shape=ddf.shape[0]
    
    ddf.to_parquet(
    path=f'{save_path}/df_{row.fcat}.parq', 
    engine='fastparquet',
    compression='snappy',
    row_group_offsets=10_000_000)
    print(f"Finished df {row.fcat} ; Before : {total_df_shape}, After : {after_shape} Change in percentage : {(total_df_shape-after_shape)/total_df_shape*100:0.2f}%")
    print(f'Time taken {time.time()-cur_time} secs')
    
    
save_path='/data/dharp/compounds/datasets/entire_df/'


pkl_df.groupby('fcat').apply(write_to_parquet)