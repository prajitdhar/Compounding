import pandas as pd
import glob
import os
import numpy as np
import pickle as pkl
import time
import dask.dataframe as dd
from fastparquet import write
import fastparquet
import snappy
import argparse


parser = argparse.ArgumentParser(description='Program to combine pickle data into parquet files for version 3')

parser.add_argument('--server', type=str,
                    help='which server to be used')
parser.add_argument('--spath', type=str,
                    help='directory where to save output')
parser.add_argument('--use_dask', action='store_true',
                    help='Should dask be used to perform final groupby operation?')
parser.add_argument('--use_parquet', action='store_true',
                    help='Should parquet be used to save the final dataset?')
args = parser.parse_args()

pkl_df=pd.read_csv(f'/data/dharp/compounds/datasets/{args.server}_fcat.txt',sep="\t")





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
    
    concat_df.drop("pos_sent",axis=1,inplace=True)
    concat_df["lemma_pos"] = concat_df["lemma_pos"].astype("category")
    #concat_df["pos_sent"] = concat_df["pos_sent"].astype("category")
    concat_df["comp_class"] = concat_df["comp_class"].astype("category")
    concat_df["comp_ner_sent"] = concat_df["comp_ner_sent"].astype("category")
    
    if args.use_dask:
        print("Using Dask")
        ddf = dd.from_pandas(concat_df, npartitions=100)

        ddf=ddf.map_partitions(lambda x: x.groupby(['lemma_pos','year','comp_class','num_comp','comp_ner_sent'],observed=True)['count'].sum())

        print('Done grouping')
        ddf=ddf.to_frame().reset_index().compute()
    else:
        print("Using Pandas")
        ddf=concat_df.groupby(['lemma_pos','year','comp_class','num_comp','comp_ner_sent'],observed=True)['count'].sum().to_frame().reset_index()

    
    print(ddf.shape[0])
    after_shape=ddf.shape[0]
    
    if args.use_parquet:
        ddf.to_parquet(path=f'{save_path}/df_{row.fcat}.parq', engine='fastparquet',compression='snappy',row_group_offsets=25_000_000)
    else:
        ddf.to_pickle(f'{save_path}/df_{row.fcat}.pkl')
    print(f"Finished df {row.fcat} ; Before : {total_df_shape}, After : {after_shape} Change in percentage : {(total_df_shape-after_shape)/total_df_shape*100:0.2f}%")
    print(f'Time taken {time.time()-cur_time} secs')
    

    
    
save_path=args.spath


pkl_df.groupby('fcat').apply(write_to_parquet)