import pandas as pd
import random
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import numpy as np
from itertools import repeat
import time

parser = argparse.ArgumentParser(description='Program to combine pickle data into one file for all settings')

parser.add_argument('--setting', type=str,
                    help='phr:Phrases const:Constituent')

parser.add_argument('--ipath', type=str,
                    help='directory where to load the datasets from')
parser.add_argument('--spath', type=str,
                    help='directory where to save output')

args = parser.parse_args()


if args.setting=="phr":
    pkl_files=glob.glob(f'{args.ipath}/phrases/*pkl')
    save_str='phrase'
    
elif args.setting=="const":
    pkl_files=glob.glob(f'{args.ipath}/words/*pkl')
    save_str='word'
    
    
random.shuffle(pkl_files)
div_lsts=np.array_split(pkl_files, 5)

dec_list=list(range(1840,2020,10))



def process_dataset(f,dec):
    cur_time=time.time()
    df=pd.read_pickle(f)
    if dec==1790:
        df=df.loc[df.year<1800]
    else:
        df=df.loc[df.year.between(dec,dec+9,inclusive='both')]
    
    if args.setting=='phr':
        df=df.groupby(['modifier','head','year','context'],observed=True)['count'].sum().to_frame()
    else:
        df=df.groupby(['word','year','context'],observed=True)['count'].sum().to_frame()

    df.reset_index(inplace=True)
    print(f'Done with file {f} in {time.time()-cur_time} secs')
    return df



def batch_processor(cur_list,dec):
    cur_time=time.time()

    dfs=[]
    n_proc = len(cur_list)
    pool = Pool(n_proc)
    dfs=pool.starmap(process_dataset, zip(cur_list,repeat(dec)))
    pool.close()
    pool.join()  
    
    print('Done parallelizing')
    combined_df=pd.concat(dfs,ignore_index=True,sort=True)
    orig_shape=combined_df.shape[0]

    if args.setting=="phr":
        df_reduced=combined_df.groupby(['modifier','head','context','year'],observed=True)['count'].sum().to_frame().reset_index()
        
    else:
        df_reduced=combined_df.groupby(['word','context','year'],observed=True)['count'].sum().to_frame().reset_index()
        
        
    print(f'Done with batch in {round(time.time()-cur_time)} secs and current size is {round(df_reduced.shape[0]/orig_shape*100,2)}% of the original dataset')

    return df_reduced


for dec in dec_list:
    if dec==2000:
        continue
    print(dec)
    for i,cur_list in enumerate(div_lsts):
        print(f'Batch {i+1}')
        cur_time=time.time()


        cur_df=batch_processor(cur_list,dec)

        cur_df.to_pickle(f'{args.spath}/{save_str}_{dec}_{i+1}')
        print(f'Time taken for batch {i+1} = {time.time()-cur_time} secs')