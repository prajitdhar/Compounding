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

dec_list=list(range(1800,2020,10))


if args.setting=="phr":
    save_str='phrase'
    
elif args.setting=="const":
    save_str='word'

    
    
    

for dec in dec_list:
    cur_time=time.time()

    print(dec)
    pkl_files=glob.glob(f'{args.ipath}/*{dec}*')

    
    df_list=[]
    for f in pkl_files:
        df = pd.read_pickle(f)
        df_list.append(df)
    
    combined_dec_df=pd.concat(df_list,ignore_index=True,sort=True)
    orig_shape=combined_dec_df.shape[0]
    
    if args.setting=="phr":
        df_reduced=combined_dec_df.groupby(['modifier','head','context','year'],observed=True)['count'].sum().to_frame().reset_index()
        
    else:
        df_reduced=combined_dec_df.groupby(['word','context','year'],observed=True)['count'].sum().to_frame().reset_index()
        
        
    print(f'Done with {args.setting} {dec} in {round(time.time()-cur_time)} secs and current size is {round(df_reduced.shape[0]/orig_shape*100,2)}% of the original dataset')

    df_reduced.to_pickle(f'{args.spath}/{save_str}_{dec}')

