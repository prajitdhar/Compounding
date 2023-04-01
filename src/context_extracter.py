import pandas as pd
import fastparquet
import glob
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import time
import random
import argparse



parser = argparse.ArgumentParser(description='Compute context list for the corpus')


parser.add_argument('--inputdir',type=str,
                    help='Location of the 5-gram datasets, in parq format')
parser.add_argument('--outputdir',type=str,
                    help='Where should the context list be stored?')

args = parser.parse_args()



num_partitions=round(0.95*mp.cpu_count())
all_parquet_files=[]
from_path=args.inputdir
to_path=args.outputdir

for filename in glob.glob(from_path+'*parq'):
    all_parquet_files.append(filename)


random.shuffle(all_parquet_files)
all_parquet_files_list=np.array_split(all_parquet_files,25)    

def context_maker(df):
    df[['w1','w2','w3','w4','w5']]=df.lemma_pos.str.split(" ",expand=True)

    context_df=pd.melt(df,id_vars=['year','count'],value_vars=['w1','w2','w3','w4','w5'])
    
    context_df=context_df.groupby(['value','year'],observed=True)['count'].sum().to_frame()
    context_df.reset_index(inplace=True)
    context_df[['word','pos']]=context_df['value'].str.rsplit('_',expand=True,n=1)
    
    context_df=context_df.groupby(['word','pos','year'],observed=True)['count'].sum().to_frame()
    context_df.reset_index(inplace=True)
    context_df=context_df.loc[context_df.pos.isin(['NOUN','ADJ','VERB','ADV'])]
    context_df['context']=context_df['word']+"_"+context_df['pos']
    context_df=context_df.groupby(['context','year'],observed=True)['count'].sum().to_frame()
    context_df.reset_index(inplace=True)

    return context_df    



def parquet_file_processer(fname):
    cur_time=time.time()
    print(f'File name: {fname}')

    parquet_context_df_list=[]
    cur_parq=fastparquet.ParquetFile(fname)
    print(f'Number of groups: {len(cur_parq.row_groups)}')
    for i,df in enumerate(cur_parq.iter_row_groups()):
        print(f'Group {i+1}')
        context_df=context_maker(df)
        parquet_context_df_list.append(context_df)
    
    parq_context_df=pd.concat(parquet_context_df_list,ignore_index=True)
    print(f'File: {parq_context_df.shape[0]}')

    parq_context_df=parq_context_df.groupby(['context','year'])['count'].sum().to_frame()
    print(f'File: {parq_context_df.shape[0]}')
    parq_context_df.reset_index(inplace=True)
    
    print(f'Total time taken for File {fname}: {round(time.time()-cur_time)} secs')        
    return parq_context_df


def parquet_batch_processer(cur_list):

    cur_time=time.time()
    parquet_batch_df_list=[]
    
    for i,cur_file in enumerate(cur_list):
        #print(f'File num: {i} name: {cur_file}')
        parquet_batch_df_list.append(parquet_file_processer(cur_file))
        
    batch_context_df=pd.concat(parquet_batch_df_list,ignore_index=True)
    print(f'Batch {batch_context_df.shape[0]}')
    
    batch_context_df=batch_context_df.groupby(['context','year'])['count'].sum().to_frame()
    batch_context_df.reset_index(inplace=True)
    print(f'Batch {batch_context_df.shape[0]}')
    print(f'Total time taken for Batch {cur_list}: {round(time.time()-cur_time)} secs')        

    return batch_context_df



cur_time=time.time()
    
pool = Pool(len(all_parquet_files_list))

#pool = Pool(len(all_parquet_files))


results=pool.map_async(parquet_batch_processer,all_parquet_files_list,chunksize=1)
pool.close()
pool.join()
    
all_context_df_list=results.get()
final_context_df=pd.concat(all_context_df_list,ignore_index=True)
    
print(f'Complete: {final_context_df.shape[0]}')
    

final_context_df=final_context_df.groupby(['context','year'])['count'].sum().to_frame()
final_context_df.reset_index(inplace=True)
print(f'Complete: {final_context_df.shape[0]}')

final_context_df.to_pickle(to_path+'/context_new.pkl')