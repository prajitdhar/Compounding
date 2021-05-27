from os import listdir
from os.path import isfile, join,getsize
import glob
import time
import random
from multiprocessing import Pool
import multiprocessing as mp
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import pickle as pkl

import warnings
#warnings.simplefilter(action='ignore', category=ResourceWarning)
np.random.seed(seed=1991)
import tables
from itertools import chain
import argparse



parser = argparse.ArgumentParser(description='Program to run google compounder for a particular file and setting')

parser.add_argument('--data', type=str,
                    help='location of the hdf5 file')

parser.add_argument('--word', action='store_true',
                    help='Extracting context for words only?')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')

parser.add_argument('--chunksize', type=int,default=50_000_000,
                    help='Value of chunksize to read datasets in')


args = parser.parse_args()






br_to_us=pd.read_excel("Book.xlsx",header=1)
br_to_us_dict=dict(zip(br_to_us.UK.tolist(),br_to_us.US.tolist()))

contextwords=pkl.load( open( "contexts.pkl", "rb" ) )

batched_pkl_files=pkl.load(open('batched_pkl_files.pkl','rb'))

spelling_replacement={'context':br_to_us_dict,'modifier':br_to_us_dict,'head':br_to_us_dict,'word':br_to_us_dict}

words_list=pkl.load(open('words_list.pkl','rb'))


any_word=r'.+_.+'
any_noun=r'.+_noun'
proper_noun=r'[a-z.-]+_noun'
space=r'\s'


def lemma_maker(x, y):
    #print(lemmatizer(x,y)[0])
    return lemmatizer(x,y)[0]

def relemjoin(df,col_name):
    new_col=col_name.split('_')[0]
    new_col_pos=new_col[0]+"_pos"
    df[new_col]=df[col_name].str.split('_', 1).str[0]
    df[new_col_pos]="noun"
    df[new_col]=np.vectorize(lemma_maker)(df[new_col], df[new_col_pos])
    df.replace(spelling_replacement,inplace=True)
    df[new_col]=df[new_col]+"_noun"
    return df
    
def syntactic_reducer(df,align,level=None):
    if len(df) == 0:
        print("Am here")
        return df
    if align=="right":
        if level=="word":
            #t1=time.time()
            df=df.loc[df.fivegram_pos.str.match(r"^"+any_noun+space+(any_word+space)*3+any_word+"$")]
            if len(df) == 0:
                return df
            
            df['word_pos'],df['r1_pos'],df['r2_pos'],df['r3_pos'],_=df['fivegram_pos'].str.split(space).str
            #df=df.query('word_pos == @word_list')
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','year','count'],value_vars=['r1_pos','r2_pos','r3_pos'])
            #print(time.time()-t1)
            return df
        else:
            #phrases=df.loc[df.fivegram_pos.str.match(r'^[-a-z]+_noun\s+[-a-z]+_noun\s+[-a-z]+_.+\s+[-a-z]+_.+\s+[-a-z]+_.+$')]
            phrases=df.loc[df.fivegram_pos.str.match(r'^'+(any_noun+space)*2+(any_word+space)*2+any_word+'$')]

            #cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[-a-z]+_noun\s+[-a-z]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^'+(any_noun+space)*3+(any_word+space)+any_word+'$')]
            
            cdsm=cdsm.loc[cdsm.fivegram_pos.str.match(r'^'+(proper_noun+space)*2+(any_word+space)*2+any_word+'$')]

            try:
                phrases['modifier_pos'],phrases['head_pos'],phrases['r1_pos'],phrases['r2_pos'],phrases['r3_pos']=phrases['fivegram_pos'].str.split(space).str
                cdsm['modifier_pos'],cdsm['head_pos'],cdsm['r1_pos'],cdsm['r2_pos'],cdsm['r3_pos']=cdsm['fivegram_pos'].str.split(space).str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos')
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos')
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','year','count'],value_vars=['r1_pos','r2_pos','r3_pos'])
            
            compounds=pd.melt(cdsm,id_vars=['modifier','head','year','count'],value_vars=['r1_pos','r2_pos','r3_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','year','count'],value_vars=['head','r1_pos','r2_pos'])
            heads=pd.melt(cdsm,id_vars=['head','year','count'],value_vars=['modifier','r1_pos','r2_pos','r3_pos'])

            return phrases,compounds,modifiers,heads
            
            
    elif align=="mid1":
        if level=="word":
            #df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            #df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            df=df.loc[df.fivegram_pos.str.match(r"^"+any_word+space+any_noun+space+(any_word+space)*2+any_word+"$")]

            if len(df) == 0:
                return df
            
            df['l1_pos'],df['word_pos'],df['r1_pos'],df['r2_pos'],df['r3_pos']=df['fivegram_pos'].str.split(space).str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','year','count'],value_vars=['l1_pos','r1_pos','r2_pos','r3_pos'])
            return df
        else:
            #phrases=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            phrases=df.loc[df.fivegram_pos.str.match(r'^'+any_word+space+(any_noun+space)*2+any_word+space+any_word+'$')]
            #cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+$')]

            
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^'+(any_noun+space)*4+any_word+'$')]
            
            cdsm=cdsm.loc[cdsm.fivegram_pos.str.match(r'^'+any_word+space+(proper_noun+space)*2+any_word+space+any_word+'$')]
            
            
            try:
                phrases['l1_pos'],phrases['modifier_pos'],phrases['head_pos'],phrases['r1_pos'],phrases['r2_pos']=phrases['fivegram_pos'].str.split(space).str
                cdsm['l1_pos'],cdsm['modifier_pos'],cdsm['head_pos'],cdsm['r1_pos'],cdsm['r2_pos']=cdsm['fivegram_pos'].str.split(space).str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos')
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos')
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','year','count'],value_vars=['l1_pos','r1_pos','r2_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','year','count'],value_vars=['l1_pos','r1_pos','r2_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','year','count'],value_vars=['head','l1_pos','r1_pos','r2_pos'])
            heads=pd.melt(cdsm,id_vars=['head','year','count'],value_vars=['modifier','l1_pos','r1_pos','r2_pos'])
            return phrases,compounds,modifiers,heads
    
            
    elif align=="mid2":
        if level=="word":
            #df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            df=df.loc[df.fivegram_pos.str.match(r'^'+(any_word+space)*2+any_noun+space+any_word+space+any_word+'$')]
            if len(df) == 0:
                return df
            
            df['l1_pos'],df['l2_pos'],df['word_pos'],df['r1_pos'],df['r2_pos']=df['fivegram_pos'].str.split(space).str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','year','count'],value_vars=['l1_pos','l2_pos','r1_pos','r2_pos'])
            return df
        else:
            
            #phrases=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+$')]
            
            phrases=df.loc[df.fivegram_pos.str.match(r'^'+(any_word+space)*2+(any_noun+space)*2+any_word+'$')]
            #cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun$')]

            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^'+any_word+space+(any_noun+space)*3+any_word+'$')]
            cdsm=cdsm.loc[cdsm.fivegram_pos.str.match(r'^'+(any_word+space)*2+(proper_noun+space)*2+any_word+'$')]
            try:
                phrases['l1_pos'],phrases['l2_pos'],phrases['modifier_pos'],phrases['head_pos'],phrases['r1_pos']=phrases['fivegram_pos'].str.split(space).str
                cdsm['l1_pos'],cdsm['l2_pos'],cdsm['modifier_pos'],cdsm['head_pos'],cdsm['r1_pos']=cdsm['fivegram_pos'].str.split(space).str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos')
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos')
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','year','count'],value_vars=['l1_pos','l2_pos','r1_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','year','count'],value_vars=['l1_pos','l2_pos','r1_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','year','count'],value_vars=['head','l1_pos','l2_pos','r1_pos'])
            heads=pd.melt(cdsm,id_vars=['head','year','count'],value_vars=['modifier','l1_pos','l2_pos','r1_pos'])
            return phrases,compounds,modifiers,heads
            
            
    elif align=="mid3":
        #df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+$')]
        df=df.loc[df.fivegram_pos.str.match(r'^'+(any_word+space)*3+any_noun+space+any_word+'$')]
        if len(df)==0:
            return df

        df['l1_pos'],df['l2_pos'],df['word_pos'],df['r1_pos'],df['r2_pos']=df['fivegram_pos'].str.split(space).str
        df=relemjoin(df,'word_pos')
        df=pd.melt(df,id_vars=['word','year','count'],value_vars=['l1_pos','l2_pos','r1_pos','r2_pos'])
        return df
        
    elif align=="left":
        
        if level=="word":
            #df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun$')]
            
            df=df.loc[df.fivegram_pos.str.match(r'^'+(any_word+space)*4+any_noun+'$')]
            if len(df) == 0:
                return df
            _,df['l1_pos'],df['l2_pos'],df['l3_pos'],df['word_pos']=df['fivegram_pos'].str.split(space).str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','year','count'],value_vars=['l1_pos','l2_pos','l3_pos'])
            return df
        else:
            #phrases=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun$')]
            
            phrases=df.loc[df.fivegram_pos.str.match(r'^'+(any_word+space)*3+any_noun+space+any_noun+'$')]
            #cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun$')]
            
            
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^'+(any_word+space)*2+(any_noun+space)*2+any_noun+'$')]
            cdsm=cdsm.loc[cdsm.fivegram_pos.str.match(r'^'+(any_word+space)*3+proper_noun+space+proper_noun+'$')]
            
            try:
                phrases['l1_pos'],phrases['l2_pos'],phrases['l3_pos'],phrases['modifier_pos'],phrases['head_pos']=phrases['fivegram_pos'].str.split(space).str
                cdsm['l1_pos'],cdsm['l2_pos'],cdsm['l3_pos'],cdsm['modifier_pos'],cdsm['head_pos']=cdsm['fivegram_pos'].str.split(space).str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos')
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos')
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','year','count'],value_vars=['l1_pos','l2_pos','l3_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','year','count'],value_vars=['l1_pos','l2_pos','l3_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','year','count'],value_vars=['head','l1_pos','l2_pos','l3_pos'])
            heads=pd.melt(cdsm,id_vars=['head','year','count'],value_vars=['modifier','l1_pos','l2_pos','l3_pos'])
            return phrases,compounds,modifiers,heads
        

        
def context_reducer(df):
    if len(df)==0:
        return df
    df["variable"]=df["variable"].str.replace(r"_pos","")
    df["context"],df["context_pos"]=df['value'].str.split('_', 1).str
    df=df.loc[df.context_pos.isin(["noun","adj","adv","verb"])]
    #df.replace(adv_replacement,inplace=True)
    #df['context_pos']=df['context_pos'].str[0]
    if len(df)==0:
        return df
    df['context']=np.vectorize(lemma_maker)(df['context'], df['context_pos'])
    df.replace(spelling_replacement,inplace=True)
    df['context']=df['context']+"_"+df['context_pos']
    df.query('context in @contextwords',inplace=True)
    #df.reset_index(inplace=True)
    return df


def cdsm_word_reducer(df):
    rightgram=syntactic_reducer(df,align="right",level="word")
    rightgram=context_reducer(rightgram)
    
    mid1gram=syntactic_reducer(df,align="mid1",level="word")
    mid1gram=context_reducer(mid1gram)
    
    mid2gram=syntactic_reducer(df,align="mid2",level="word")
    mid2gram=context_reducer(mid2gram)
    
    mid3gram=syntactic_reducer(df,align="mid3",level="word")
    mid3gram=context_reducer(mid3gram)
    
    leftgram=syntactic_reducer(df,align="left",level="word")
    leftgram=context_reducer(leftgram)    
    
    words_df=pd.concat([rightgram,mid1gram,mid2gram,mid3gram,leftgram],ignore_index=True,sort=False)
    words_df.dropna(inplace=True)
    words_df=words_df.query('word in @words_list')
    words_df=words_df.groupby(['word','context','year'])['count'].sum().to_frame()
    words_df.reset_index(inplace=True)
    words_df.year=words_df.year.astype("int32")
    #print(words_df.shape)
    return words_df


def cdsm_reducer(df):
    phrase_rightgram,compound_rightgram,modifier_rightgram,head_rightgram=syntactic_reducer(df,align="right")
    phrase_rightgram=context_reducer(phrase_rightgram)
    compound_rightgram=context_reducer(compound_rightgram)
    modifier_rightgram=context_reducer(modifier_rightgram)
    head_rightgram=context_reducer(head_rightgram)


    phrase_mid1gram,compound_mid1gram,modifier_mid1gram,head_mid1gram=syntactic_reducer(df,align="mid1")
    phrase_mid1gram=context_reducer(phrase_mid1gram)
    compound_mid1gram=context_reducer(compound_mid1gram)
    modifier_mid1gram=context_reducer(modifier_mid1gram)
    head_mid1gram=context_reducer(head_mid1gram)
 

    phrase_mid2gram,compound_mid2gram,modifier_mid2gram,head_mid2gram=syntactic_reducer(df,align="mid2")
    phrase_mid2gram=context_reducer(phrase_mid2gram)
    compound_mid2gram=context_reducer(compound_mid2gram)
    modifier_mid2gram=context_reducer(modifier_mid2gram)
    head_mid2gram=context_reducer(head_mid2gram)
    
    phrase_leftgram,compound_leftgram,modifier_leftgram,head_leftgram=syntactic_reducer(df,align="left")
    phrase_leftgram=context_reducer(phrase_leftgram)
    compound_leftgram=context_reducer(compound_leftgram)
    modifier_leftgram=context_reducer(modifier_leftgram)
    head_leftgram=context_reducer(head_leftgram)
    
    
    phrases=pd.concat([phrase_rightgram,phrase_mid1gram,phrase_mid2gram,phrase_leftgram],ignore_index=True,sort=False)
    compounds=pd.concat([compound_rightgram,compound_mid1gram,compound_mid2gram,compound_leftgram],ignore_index=True,sort=False)
    modifiers=pd.concat([modifier_rightgram,modifier_mid1gram,modifier_mid2gram,modifier_leftgram],ignore_index=True,sort=False)
    heads=pd.concat([head_rightgram,head_mid1gram,head_mid2gram,head_leftgram],ignore_index=True,sort=False)

    
    phrases.dropna(inplace=True)
    phrases=phrases.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
    phrases.reset_index(inplace=True)
    phrases.year=phrases.year.astype("int32")
    
    compounds.dropna(inplace=True)
    compounds=compounds.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
    compounds.reset_index(inplace=True)
    compounds.year=compounds.year.astype("int32")
    
    modifiers.dropna(inplace=True)
    modifiers=modifiers.groupby(['modifier','context','year'])['count'].sum().to_frame()
    modifiers.reset_index(inplace=True)
    modifiers.year=modifiers.year.astype("int32")
    
    heads.dropna(inplace=True)
    heads=heads.groupby(['head','context','year'])['count'].sum().to_frame()
    heads.reset_index(inplace=True)
    heads.year=heads.year.astype("int32")
    return compounds,modifiers,heads,phrases


def parallelize_dataframe(df,save_loc,num_cores):
    num_partitions = num_cores
    df_split = np.array_split(df, num_partitions)
    print("Done splitting the datasets")
    pool = Pool(num_cores)

    cur_time=time.time()
    print("Starting parallelizing")
    if not args.word:

        results=pool.map_async(cdsm_reducer,df_split)
        pool.close()
        pool.join()

        results=results.get()

        
        print("Done parallelizing")
        print("Total time taken",round(time.time()-cur_time),"secs")
        compound_list = [ result[0] for result in results]
        compounds=pd.concat(compound_list,ignore_index=True)
        compounds=compounds.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)
        
        if not isfile("/data/dharp/compounding/datasets/compounds.csv"):
            compounds.to_csv("/data/dharp/compounding/datasets/compounds.csv",sep="\t",index=False)
        else:
            compounds.to_csv("/data/dharp/compounding/datasets/compounds.csv", mode='a',sep="\t", header=False,index=False)
        
        
        modifier_list = [ result[1] for result in results]
        modifiers=pd.concat(modifier_list,ignore_index=True)
        modifiers=modifiers.groupby(['modifier','context','year'])['count'].sum().to_frame()
        modifiers.reset_index(inplace=True)

        if not isfile("/data/dharp/compounding/datasets/modifiers.csv"):
            modifiers.to_csv("/data/dharp/compounding/datasets/modifiers.csv",sep="\t",index=False)
        else:
            modifiers.to_csv("/data/dharp/compounding/datasets/modifiers.csv", mode='a',sep="\t",header=False,index=False)
        
        head_list = [ result[2] for result in results]
        heads=pd.concat(head_list,ignore_index=True)
        heads=heads.groupby(['head','context','year'])['count'].sum().to_frame()
        heads.reset_index(inplace=True)

        if not isfile("/data/dharp/compounding/datasets/heads.csv"):
            heads.to_csv("/data/dharp/compounding/datasets/heads.csv",sep="\t",index=False)
        else:
            heads.to_csv("/data/dharp/compounding/datasets/heads.csv", mode='a',sep="\t",header=False,index=False)
            
        phrase_list = [ result[3] for result in results]
        phrases=pd.concat(phrase_list,ignore_index=True)
        phrases=phrases.groupby(['modifier','head','context','year'])['count'].sum().to_frame()
        phrases.reset_index(inplace=True)
        
        if not isfile("/data/dharp/compounding/datasets/phrases.csv"):
            phrases.to_csv("/data/dharp/compounding/datasets/phrases.csv",sep="\t",index=False)
        else:
            phrases.to_csv("/data/dharp/compounding/datasets/phrases.csv", mode='a',sep="\t",header=False,index=False)

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
                
        if not isfile(save_loc):
            words.to_csv(save_loc,sep="\t",index=False,header=True)
        else:
            words.to_csv(save_loc, mode='a',sep="\t", header=False,index=False)
        
    print("Done concatenations \n")

    
file_name=args.data
str_num=file_name.split('/')[-1].split('.')[0].split('_')[-1]
if args.word:
    output_file=args.output+'/words_'+str_num+'.csv'
num_cores=mp.cpu_count()-1
store = pd.HDFStore(args.data)

print(f'File {args.data} is read in')
chunksize=args.chunksize
nrows = store.get_storer('df').nrows
print(f'Num of iterations : {nrows//chunksize}')

for i in range(nrows//chunksize + 1):
    chunk = store.select('df',start=i*chunksize,stop=(i+1)*chunksize)
    parallelize_dataframe(chunk,save_loc=output_file,num_cores=num_cores)
    
    
    
print("Done with file \n")

store.close()