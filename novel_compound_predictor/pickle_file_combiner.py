import tables
import pandas as pd
import glob
import os
import numpy as np
import pickle as pkl



batched_pkl_files=pkl.load(open('batched_pkl_files.pkl','rb'))


for num,pkl_files in enumerate(batched_pkl_files):
    df_list=[]
    for f in pkl_files:
        
        tmp_pkl=pd.read_pickle(f)
        tmp_pkl.reset_index(inplace=True)
        df_list.append(tmp_pkl)
        
    tmp_df=pd.concat(df_list,sort=False)
    tmp_df.to_hdf('/data/dharp/compounding/datasets/entire_df_'+str(num)+'.h5',format='table', key='df',complib='zlib', complevel=5)
