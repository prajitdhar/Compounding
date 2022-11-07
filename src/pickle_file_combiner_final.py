import random
import pandas as pd
import glob
import os
import numpy as np
import pickle as pkl
import time
import multiprocessing as mp
from multiprocessing import Pool
import re
import argparse


parser = argparse.ArgumentParser(description='Program to combine pickle data into one file for all settings')

parser.add_argument('--setting', type=str,
                    help='comp:Compounds mod:Modifiers head:Heads phr:Phrases const:Constituent')
parser.add_argument('--spath', type=str,
                    help='directory where to save output')

args = parser.parse_args()



if args.setting=="comp":
    pkl_files=glob.glob(f'{args.spath}/compounds/*pkl')
    
elif args.setting=="mod":
    pkl_files=glob.glob(f'{args.spath}/modifiers/*pkl')
    
elif args.setting=="head":
    pkl_files=glob.glob(f'{args.spath}/heads/*pkl')
    
elif args.setting=="phr":
    pkl_files=glob.glob(f'{args.spath}/phrases/*pkl')
    
elif args.setting=="const":
    pkl_files=glob.glob(f'{args.spath}/words/*pkl')
    
random.shuffle(pkl_files)


contexts=pd.read_pickle('/data/dharp/compounds/datasets/contexts/contexts_top50k.pkl')
heads=pd.read_pickle('/data/dharp/compounds/datasets/contexts/heads.pkl')
modifiers=pd.read_pickle('/data/dharp/compounds/datasets/contexts/modifiers.pkl')


div_lsts=np.array_split(pkl_files, 10)


def mem_reducer(pkl_file):
    print(f'\nStarted with file {pkl_file}\n')
    cur_time=time.time()
    
    df=pd.read_pickle(pkl_file)
    df["year"] = pd.to_numeric(df["year"], downcast="unsigned")
    orig_shape=df.shape[0]
    #print(orig_shape,((orig_shape-df.shape[0])/orig_shape*100))
    df=df.loc[df['year']>=1800]
    #print(df.shape[0],((orig_shape-df.shape[0])/orig_shape*100))
    df=df.loc[df.modifier.isin(modifiers)]
    #print(df.shape[0],((orig_shape-df.shape[0])/orig_shape*100))
    df=df.loc[df['head'].isin(heads)]
    #print(df.shape[0],((orig_shape-df.shape[0])/orig_shape*100))
    df=df.loc[df.context.isin(contexts)]
    #print(df.shape[0],((orig_shape-df.shape[0])/orig_shape*100))
    #df["comp_ner_sent"] = df["comp_ner_sent"].astype("category")
    df=df.groupby(['modifier','head','context','year'])['count'].sum().to_frame().reset_index()
    #print(df.shape[0],((orig_shape-df.shape[0])/orig_shape*100))
    print(f'Done with file {pkl_file} in {round(time.time()-cur_time)} secs and current size is {round(df.shape[0]/orig_shape*100,2)}% of the original dataset')
    return df


def batch_processor(cur_list):
    n_proc = len(cur_list)
    pool = Pool(n_proc)
    results=pool.map_async(mem_reducer,cur_list) 
    pool.close()
    pool.join()
    
    dfs=results.get()
    combined_df=pd.concat(dfs,ignore_index=True,sort=True)
    
    
    df_reduced=combined_df.groupby(['modifier','head','context','year'])['count'].sum().to_frame().reset_index()
    
    return df_reduced

all_df_files=[]
for i,cur_list in enumerate(div_lsts):
    print(f'Batch {i+1}')
    cur_time=time.time()
    cur_df=batch_processor(cur_list)
    all_df_files.append(cur_df)
    print(f'Time taken for batch {i+1} = {time.time()-cur_time} secs')

    

final_combined_df=pd.concat(all_df_files,ignore_index=True,sort=True)

final_df_reduced=final_combined_df.groupby(['modifier','head','context','year'])['count'].sum().to_frame().reset_index()

if args.setting=="comp":
    final_df_reduced.to_pickle(f'{args.spath}/compounds/compounds.pkl')
    
elif args.setting=="mod":
    pkl_files=glob.glob(f'{args.spath}/modifiers/*pkl')
    
elif args.setting=="head":
    pkl_files=glob.glob(f'{args.spath}/heads/*pkl')
    
elif args.setting=="phr":
    pkl_files=glob.glob(f'{args.spath}/phrases/*pkl')
    
elif args.setting=="const":
    pkl_files=glob.glob(f'{args.spath}/words/*pkl')