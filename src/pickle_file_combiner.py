import pandas as pd
import glob
import os
import numpy as np
import pickle as pkl
import time
from fastparquet import write
import fastparquet
import multiprocessing as mp
import snappy
import argparse
import multiprocessing as mp
from multiprocessing import Pool


parser = argparse.ArgumentParser(description='Program to combine pickle data into parquet files for version 3')


parser.add_argument('--spath', type=str,
                    help='directory where to save output')
parser.add_argument('--dec', type=str,
                    help='Which decade to be processed?')


args = parser.parse_args()

save_path=args.spath


pkl_files=glob.glob(f'{args.spath}/{args.dec}/*pkl')


def df_list_processor(df_list):
    
    if type(df_list)==str:
        df_list=[df_list]
    concat_list=[]
    for cur_file in df_list:
        #print(row.fname)
        cur_df=pd.read_pickle(cur_file)
        concat_list.append(cur_df)
    concat_df=pd.concat(concat_list,ignore_index=True,sort=True)
    total_df_shape=concat_df.shape[0]
    
    new_df=concat_df.groupby(['lemma_pos','comp_class','is_comp','comp_ner_sent'])['count'].sum().to_frame().reset_index()
    
    new_df["comp_class"] = new_df["comp_class"].astype("category")
    new_df["comp_ner_sent"] = new_df["comp_ner_sent"].astype("category")
    
    return new_df

    
print(f'Decade {args.dec}')

num_partitions=round(0.95*mp.cpu_count())
cur_time=time.time()
pool = Pool(num_partitions)
print('Started parallelization')
results=pool.map_async(df_list_processor,pkl_files,chunksize=len(pkl_files)//num_partitions)
pool.close()
pool.join()
    
print('Done parallelization')

curr_df_list=results.get()
new_index_df=pd.concat(curr_df_list,ignore_index=True)
old_shape=new_index_df.shape[0]

new_index_df=new_index_df.groupby(['lemma_pos','comp_class','is_comp','comp_ner_sent'],observed=True)['count'].sum().to_frame().reset_index()

print(f'{new_index_df.shape[0]/old_shape}')

print('Saving')
new_index_df.to_parquet(path=f'{save_path}/{args.dec}.parq', engine='fastparquet',compression='snappy',row_group_offsets=50_000_000)

print(f'Total time taken {round(time.time()-cur_time)} secs')