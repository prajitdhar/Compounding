import pandas as pd
import glob
import os
import numpy as np
import time
import fastparquet
import argparse
from multiprocessing import Pool
import multiprocessing as mp
from os.path import isfile


parser = argparse.ArgumentParser(description='Program to run google compounder for a particular file and setting')

parser.add_argument('--data', type=str,
                    help='location of the pickle file')

parser.add_argument('--word', action='store_true',
                    help='Extracting context for words only?')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')


args = parser.parse_args()


with open('/data/dharp/compounds/datasets/contexts/no_ner_0_50000.txt','r') as f:
    contexts=f.read().split("\n")
    contexts=contexts[:-1]
len(contexts)


def left_side_parser(df): # N N _ _ _
    cur_df=df.copy()

    try:
        cur_df[['modifier','head','w1','w2','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
    except ValueError:
        compound_df=pd.DataFrame()
        modifier_df=pd.DataFrame()
        head_df=pd.DataFrame()
        return compound_df,modifier_df,head_df
    
    compound_df=pd.melt(cur_df,id_vars=['modifier','head','year','count'],value_vars=['w1','w2','w3'],value_name='context')
    compound_df=compound_df.loc[compound_df.context.isin(contexts)]

    modifier_df=pd.melt(cur_df,id_vars=['modifier','year','count'],value_vars=['head','w1','w2'],value_name='context')
    modifier_df=modifier_df.loc[modifier_df.context.isin(contexts)]
    
    head_df=pd.melt(cur_df,id_vars=['head','year','count'],value_vars=['modifier','w1','w2','w3'],value_name='context')
    head_df=head_df.loc[head_df.context.isin(contexts)]
    
    return compound_df,modifier_df,head_df

def mid1_parser(df): # _ N N _ _
    cur_df=df.copy()
    try:
        cur_df[['w1','modifier','head','w2','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
    except ValueError:
        compound_df=pd.DataFrame()
        modifier_df=pd.DataFrame()
        head_df=pd.DataFrame()
        return compound_df,modifier_df,head_df
    
    compound_df=pd.melt(cur_df,id_vars=['modifier','head','year','count'],value_vars=['w1','w2','w3'],value_name='context')
    compound_df=compound_df.loc[compound_df.context.isin(contexts)]

    modifier_df=pd.melt(cur_df,id_vars=['modifier','year','count'],value_vars=['head','w1','w2','w3'],value_name='context')
    modifier_df=modifier_df.loc[modifier_df.context.isin(contexts)]
    
    head_df=pd.melt(cur_df,id_vars=['head','year','count'],value_vars=['modifier','w1','w2','w3'],value_name='context')
    head_df=head_df.loc[head_df.context.isin(contexts)]
    
    return compound_df,modifier_df,head_df

def mid2_parser(df): # _ _ N N _
    cur_df=df.copy()
    try:
        cur_df[['w1','w2','modifier','head','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
    except ValueError:
        compound_df=pd.DataFrame()
        modifier_df=pd.DataFrame()
        head_df=pd.DataFrame()
        return compound_df,modifier_df,head_df
       
    compound_df=pd.melt(cur_df,id_vars=['modifier','head','year','count'],value_vars=['w1','w2','w3'],value_name='context')
    compound_df=compound_df.loc[compound_df.context.isin(contexts)]

    modifier_df=pd.melt(cur_df,id_vars=['modifier','year','count'],value_vars=['head','w1','w2','w3'],value_name='context')
    modifier_df=modifier_df.loc[modifier_df.context.isin(contexts)]
    
    head_df=pd.melt(cur_df,id_vars=['head','year','count'],value_vars=['modifier','w1','w2','w3'],value_name='context')
    head_df=head_df.loc[head_df.context.isin(contexts)]
    
    return compound_df,modifier_df,head_df

def right_side_parser(df): # _ _ _ N N
    cur_df=df.copy()
    try:
        cur_df[['w1','w2','w3','modifier','head']]=cur_df.lemma_pos.str.split(' ',expand=True)
    except ValueError:
        compound_df=pd.DataFrame()
        modifier_df=pd.DataFrame()
        head_df=pd.DataFrame()
        return compound_df,modifier_df,head_df
    
    compound_df=pd.melt(cur_df,id_vars=['modifier','head','year','count'],value_vars=['w1','w2','w3'],value_name='context')
    compound_df=compound_df.loc[compound_df.context.isin(contexts)]
    
    modifier_df=pd.melt(cur_df,id_vars=['modifier','year','count'],value_vars=['head','w1','w2','w3'],value_name='context')
    modifier_df=modifier_df.loc[modifier_df.context.isin(contexts)]
    
    head_df=pd.melt(cur_df,id_vars=['head','year','count'],value_vars=['modifier','w2','w3'],value_name='context')
    head_df=head_df.loc[head_df.context.isin(contexts)]
    
    return compound_df,modifier_df,head_df


def syntactic_reducer(df):
    pattern=df.iloc[0].comp_class
    if pattern==1: # N N _ _ N N
        compound_left_df,modifier_left_df,head_left_df=left_side_parser(df)
        compound_right_df,modifier_right_df,head_right_df=right_side_parser(df)
        
        final_compound_df=pd.concat([compound_left_df,compound_right_df],ignore_index=True)
        final_modifier_df=pd.concat([modifier_left_df,modifier_right_df],ignore_index=True)
        final_head_df=pd.concat([head_left_df,head_right_df],ignore_index=True)
           
    elif pattern==2: # N N _ _ _
        final_compound_df,final_modifier_df,final_head_df=left_side_parser(df)

    elif pattern==3: # _ N N _ _
        final_compound_df,final_modifier_df,final_head_df=mid1_parser(df)
    
    elif pattern==4: # _ _ N N _
        final_compound_df,final_modifier_df,final_head_df=mid2_parser(df)
        
    elif pattern==5: # _ _ _ N N
        final_compound_df,final_modifier_df,final_head_df=right_side_parser(df)

    return final_compound_df,final_modifier_df,final_head_df


def compound_extracter(df):
    if df.loc[df.comp_class==1].shape[0]!=0:
        sides_comp_df,sides_mod_df,sides_head_df=syntactic_reducer(df.loc[df.comp_class==1])
    else:
        sides_comp_df=pd.DataFrame()
        sides_mod_df=pd.DataFrame()
        sides_head_df=pd.DataFrame()
    
    if df.loc[df.comp_class==2].shape[0]!=0:
        left_comp_df,left_mod_df,left_head_df=syntactic_reducer(df.loc[df.comp_class==2])
    else:
        left_comp_df=pd.DataFrame()
        left_mod_df=pd.DataFrame()
        left_head_df=pd.DataFrame()       
        
    if df.loc[df.comp_class==3].shape[0]!=0:
        mid1_comp_df,mid1_mod_df,mid1_head_df=syntactic_reducer(df.loc[df.comp_class==3])
    else:
        mid1_comp_df=pd.DataFrame()
        mid1_mod_df=pd.DataFrame()
        mid1_head_df=pd.DataFrame()
        
    if df.loc[df.comp_class==4].shape[0]!=0:
        mid2_comp_df,mid2_mod_df,mid2_head_df=syntactic_reducer(df.loc[df.comp_class==4])
    else:
        mid2_comp_df=pd.DataFrame()
        mid2_mod_df=pd.DataFrame()
        mid2_head_df=pd.DataFrame()

    if df.loc[df.comp_class==5].shape[0]!=0:
        right_comp_df,right_mod_df,right_head_df=syntactic_reducer(df.loc[df.comp_class==5])
        
    else:
        right_comp_df=pd.DataFrame()
        right_mod_df=pd.DataFrame()
        right_head_df=pd.DataFrame()

    compounds=pd.concat([sides_comp_df,left_comp_df,mid1_comp_df,mid2_comp_df,right_comp_df],ignore_index=True,sort=False)
    modifiers=pd.concat([sides_mod_df,left_mod_df,mid1_mod_df,mid2_mod_df,right_mod_df],ignore_index=True,sort=False)
    heads=pd.concat([sides_head_df,left_head_df,mid1_head_df,mid2_head_df,right_head_df],ignore_index=True,sort=False)
    
    if len(compounds)==0:
        return compounds,modifiers,heads
    
    compounds.dropna(inplace=True)
    compounds=compounds.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
    compounds.reset_index(inplace=True)
    
    modifiers.dropna(inplace=True)
    modifiers=modifiers.groupby(['modifier','context','year'])['count'].sum().to_frame()
    modifiers.reset_index(inplace=True)
    
    heads.dropna(inplace=True)
    heads=heads.groupby(['head','context','year'])['count'].sum().to_frame()
    heads.reset_index(inplace=True)
    
    return compounds,modifiers,heads



def parallelize_dataframe(df):
    num_partitions=round(0.95*mp.cpu_count())
    df_split = np.array_split(df, num_partitions)
    print("Done splitting the datasets")
    pool = Pool(num_partitions)

    cur_time=time.time()
    print("Starting parallelizing")
    if not args.word:

        results=pool.map_async(compound_extracter,df_split)
        pool.close()
        pool.join()

        results=results.get()

        
        print("Done parallelizing")
        print("Total time taken",round(time.time()-cur_time),"secs")
        compound_list = [ result[0] for result in results]
        compounds=pd.concat(compound_list,ignore_index=True)
        compounds=compounds.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)
        
        #if not isfile(f'{args.output}/compounds.csv'):
            #compounds.to_csv(f'{args.output}/compounds.csv',sep="\t",index=False)
        #else:
            #compounds.to_csv(f'{args.output}/compounds.csv', mode='a',sep="\t", header=False,index=False)
        
        
        modifier_list = [ result[1] for result in results]
        modifiers=pd.concat(modifier_list,ignore_index=True)
        modifiers=modifiers.groupby(['modifier','context','year'])['count'].sum().to_frame()
        modifiers.reset_index(inplace=True)

        #if not isfile(f'{args.output}/modifiers.csv'):
            #modifiers.to_csv(f'{args.output}/modifiers.csv',sep="\t",index=False)
        #else:
            #modifiers.to_csv(f'{args.output}/modifiers.csv', mode='a',sep="\t",header=False,index=False)
        
        head_list = [ result[2] for result in results]
        heads=pd.concat(head_list,ignore_index=True)
        heads=heads.groupby(['head','context','year'])['count'].sum().to_frame()
        heads.reset_index(inplace=True)

        return compounds,modifiers,heads
        #if not isfile(f'{args.output}/heads.csv'):
            #heads.to_csv(f'{args.output}/heads.csv',sep="\t",index=False)
        #else:
            #heads.to_csv(f'{args.output}/heads.csv', mode='a',sep="\t",header=False,index=False)
            
#        phrase_list = [ result[3] for result in results]
#        phrases=pd.concat(phrase_list,ignore_index=True)
#        phrases=phrases.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
#        phrases.reset_index(inplace=True)
        
#        if not isfile(f'{args.output}/phrases.csv'):
#            phrases.to_csv(f'{args.output}/phrases.csv',sep="\t",index=False)
#        else:
#            phrases.to_csv(f'{args.output}/phrases.csv', mode='a',sep="\t",header=False,index=False)

    else:
        words_list=[]
        results=pool.map_async(cdsm_word_reducer,df_split)
  
        
        pool.close()
        pool.join()
        print("Done parallelizing")
        print("Total time taken",round(time.time()-cur_time),"secs")
        words_list=results.get()
        words = pd.concat(words_list,ignore_index=True,sort=False)
        words=words.groupby(['word','context','year'])['count'].sum().to_frame()
        words.reset_index(inplace=True)
        print(words.shape)
                
        if not isfile(f'{args.output}/words.csv'):
            words.to_csv(f'{args.output}/words.csv',sep="\t",index=False,header=True)
        else:
            words.to_csv(f'{args.output}/words.csv', mode='a',sep="\t", header=False,index=False)
        
    print("Done concatenations \n")
    
    
    
def parquet_processor(f):   
    cur_fname=f.split('.')[0].split('/')[-1]
    print(f'Current parquet file: {f}')
    cur_parq=fastparquet.ParquetFile(f)
    print(f'Number of partitions: {len(cur_parq.row_groups)}')
    compounds_list=[]
    modifiers_list=[]
    heads_list=[]

    for i,cur_df in enumerate(cur_parq.iter_row_groups()):
        print(f'Partition {i+1} out of {len(cur_parq.row_groups)}')
        cur_df.year=cur_df.year.astype("int32")
        cur_df=cur_df.loc[cur_df.comp_class!=0].reset_index(drop=True)
        cur_compounds,cur_modifiers,cur_heads=parallelize_dataframe(cur_df)
        compounds_list.append(cur_compounds)
        modifiers_list.append(cur_modifiers)
        heads_list.append(cur_heads)

        
    compounds=pd.concat(compounds_list,ignore_index=True)
    compounds=compounds.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
    compounds.reset_index(inplace=True)
    
    compounds.to_pickle(f'{args.output}/compounds/{cur_fname}.pkl')
    
    #compounds.to_parquet(
    #path=f'{args.output}/compounds/{cur_fname}.parq', 
    #engine='fastparquet',
    #compression='snappy')        
        
   
    modifiers=pd.concat(modifiers_list,ignore_index=True)
    modifiers=modifiers.groupby(['modifier','context','year'])['count'].sum().to_frame()
    modifiers.reset_index(inplace=True)
    
    modifiers.to_pickle(f'{args.output}/modifiers/{cur_fname}.pkl')

    
    #modifiers.to_parquet(
    #path=f'{args.output}/modifiers/{cur_fname}.parq', 
    #engine='fastparquet',
    #compression='snappy')

    
    heads=pd.concat(heads_list,ignore_index=True)
    heads=heads.groupby(['head','context','year'])['count'].sum().to_frame()
    heads.reset_index(inplace=True)

    heads.to_pickle(f'{args.output}/heads/{cur_fname}.pkl')
    #heads.to_parquet(
    #path=f'{args.output}/heads/{cur_fname}.parq', 
    #engine='fastparquet',
    #compression='snappy')
    print("Done with file \n")
    
    
    
for i in range(8,27):
    print(i)
    parquet_processor(f"/data/dharp/compounds/datasets/entire_df/df_{i}.parq")


