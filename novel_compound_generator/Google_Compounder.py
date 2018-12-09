import csv
from os import listdir
from os.path import isfile, join,getsize
import glob
import time
import random
from multiprocessing import Pool
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
#from scipy.interpolate import interp1d
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import pickle as pkl
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import dask.array as da
#from sklearn.decomposition import PCA,TruncatedSVD,NMF
import dask.dataframe as dd
import warnings
warnings.simplefilter(action='ignore', category=ResourceWarning)
np.random.seed(seed=1991)
import feather
import tables
#from tqdm import tqdm

input_files = [f for f in glob.glob("/data/dharp/compounding/datasets/entire_df_*_.h5") if isfile(join("/data/dharp/compounding/datasets/", f))]
files=[]
for f in input_files:
    if "_18_" in f:
        continue
    elif "_11_" in f:
        continue
    files.append(f)
np.random.shuffle(files)


br_to_us=pd.read_excel("Book.xlsx")
br_to_us_dict=dict(zip(br_to_us.UK.tolist(),br_to_us.US.tolist()))

contextwords_df=pd.read_csv("contexts.csv",sep="\t")
contextwords=contextwords_df.context.tolist()


adv_dict=dict(zip(['adv'],['r']))
adv_replacement={'context_pos':adv_dict}
spelling_replacement={'context':br_to_us_dict,'modifier':br_to_us_dict,'head':br_to_us_dict,'word':br_to_us_dict}


pos_replacement={'pos':dict(zip(["noun","verb","adj"],['n','v','a']))}

def lemma_maker(x, y):
    return lemmatizer.lemmatize(x,y)

def relemjoin(df,col_name,lemmatize=True):
    new_col=col_name.split('_')[0]
    new_col_pos=new_col[0]+"_pos"
    df[new_col]=df[col_name].str.split('_', 1).str[0]
    df[new_col_pos]="n"
    if lemmatize==True:
        df[new_col]=np.vectorize(lemma_maker)(df[new_col], df[new_col_pos])
    df[new_col]=df[new_col]+"_n"
    return df


def syntactic_reducer(df,align,level=None):
    if len(df) == 0:
        print("Am here")
        return df
    if align=="right":
        if level=="word":
            df=df.loc[df.fivegram_pos.str.match(r'^[-a-z]+_noun\s+[-a-z]+_.+\s+[-a-z]+_.+\s+[-a-z]+_.+\s+[-a-z]+_.+$')]
            if len(df) == 0:
                return df
            
            df['word_pos'],df['r1_pos'],df['r2_pos'],df['r3_pos'],_=df['fivegram_pos'].str.split(r'\s+').str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','decade','count'],value_vars=['r1_pos','r2_pos','r3_pos'])
            return df
        else:
            phrases=df.loc[df.fivegram_pos.str.match(r'^[-a-z]+_noun\s+[-a-z]+_noun\s+[-a-z]+_.+\s+[-a-z]+_.+\s+[-a-z]+_.+$')]
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[-a-z]+_noun\s+[-a-z]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+$')]

            try:
                phrases['modifier_pos'],phrases['head_pos'],phrases['r1_pos'],phrases['r2_pos'],phrases['r3_pos']=phrases['fivegram_pos'].str.split(r'\s+').str
                cdsm['modifier_pos'],cdsm['head_pos'],cdsm['r1_pos'],cdsm['r2_pos'],cdsm['r3_pos']=cdsm['fivegram_pos'].str.split(r'\s+').str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos',lemmatize=False)
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos',lemmatize=False)
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','decade','count'],value_vars=['r1_pos','r2_pos','r3_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','decade','count'],value_vars=['r1_pos','r2_pos','r3_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','decade','count'],value_vars=['head','r1_pos','r2_pos'])
            heads=pd.melt(cdsm,id_vars=['head','decade','count'],value_vars=['modifier','r1_pos','r2_pos','r3_pos'])
            return phrases,compounds,modifiers,heads
            
            
    elif align=="mid1":
        if level=="word":
            df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            if len(df) == 0:
                return df
            
            df['l1_pos'],df['word_pos'],df['r1_pos'],df['r2_pos'],df['r3_pos']=df['fivegram_pos'].str.split(r'\s+').str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','decade','count'],value_vars=['l1_pos','r1_pos','r2_pos','r3_pos'])
            return df
        else:
            phrases=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+$')]

            try:
                phrases['l1_pos'],phrases['modifier_pos'],phrases['head_pos'],phrases['r1_pos'],phrases['r2_pos']=phrases['fivegram_pos'].str.split(r'\s+').str
                cdsm['l1_pos'],cdsm['modifier_pos'],cdsm['head_pos'],cdsm['r1_pos'],cdsm['r2_pos']=cdsm['fivegram_pos'].str.split(r'\s+').str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos',lemmatize=False)
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos',lemmatize=False)
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','decade','count'],value_vars=['l1_pos','r1_pos','r2_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','decade','count'],value_vars=['l1_pos','r1_pos','r2_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','decade','count'],value_vars=['head','l1_pos','r1_pos','r2_pos'])
            heads=pd.melt(cdsm,id_vars=['head','decade','count'],value_vars=['modifier','l1_pos','r1_pos','r2_pos'])
            return phrases,compounds,modifiers,heads
    
            
    elif align=="mid2":
        if level=="word":
            df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+\s+[a-z-]+_.+$')]
            if len(df) == 0:
                return df
            
            df['l1_pos'],df['l2_pos'],df['word_pos'],df['r1_pos'],df['r2_pos']=df['fivegram_pos'].str.split(r'\s+').str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','decade','count'],value_vars=['l1_pos','l2_pos','r1_pos','r2_pos'])
            return df
        else:
            
            phrases=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_.+$')]
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun$')]

            try:
                phrases['l1_pos'],phrases['l2_pos'],phrases['modifier_pos'],phrases['head_pos'],phrases['r1_pos']=phrases['fivegram_pos'].str.split(r'\s+').str
                cdsm['l1_pos'],cdsm['l2_pos'],cdsm['modifier_pos'],cdsm['head_pos'],cdsm['r1_pos']=cdsm['fivegram_pos'].str.split(r'\s+').str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos',lemmatize=False)
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos',lemmatize=False)
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','decade','count'],value_vars=['l1_pos','l2_pos','r1_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','decade','count'],value_vars=['l1_pos','l2_pos','r1_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','decade','count'],value_vars=['head','l1_pos','l2_pos','r1_pos'])
            heads=pd.melt(cdsm,id_vars=['head','decade','count'],value_vars=['modifier','l1_pos','l2_pos','r1_pos'])
            return phrases,compounds,modifiers,heads
            
            
    elif align=="mid3":
        df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_.+$')]
        if len(df)==0:
            return df

        df['l1_pos'],df['l2_pos'],df['word_pos'],df['r1_pos'],df['r2_pos']=df['fivegram_pos'].str.split(r'\s+').str
        df=relemjoin(df,'word_pos')
        df=pd.melt(df,id_vars=['word','decade','count'],value_vars=['l1_pos','l2_pos','r1_pos','r2_pos'])
        return df
        
    elif align=="left":
        
        if level=="word":
            df=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun$')]
            if len(df) == 0:
                return df
            _,df['l1_pos'],df['l2_pos'],df['l3_pos'],df['word_pos']=df['fivegram_pos'].str.split(r'\s+').str
            df=relemjoin(df,'word_pos')
            df=pd.melt(df,id_vars=['word','decade','count'],value_vars=['l1_pos','l2_pos','l3_pos'])
            return df
        else:
            phrases=df.loc[df.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun$')]
            cdsm=phrases.loc[~phrases.fivegram_pos.str.match(r'^[a-z-]+_.+\s+[a-z-]+_.+\s+[a-z-]+_noun\s+[a-z-]+_noun\s+[a-z-]+_noun$')]
            
            try:
                phrases['l1_pos'],phrases['l2_pos'],phrases['l3_pos'],phrases['modifier_pos'],phrases['head_pos']=phrases['fivegram_pos'].str.split(r'\s+').str
                cdsm['l1_pos'],cdsm['l2_pos'],cdsm['l3_pos'],cdsm['modifier_pos'],cdsm['head_pos']=cdsm['fivegram_pos'].str.split(r'\s+').str
            except ValueError:
                phrases=pd.DataFrame()
                compounds=pd.DataFrame()
                modifiers=pd.DataFrame()
                heads=pd.DataFrame()
                return phrases,compounds,modifiers,heads
            
            phrases=relemjoin(phrases,'modifier_pos',lemmatize=False)
            phrases=relemjoin(phrases,'head_pos')
            cdsm=relemjoin(cdsm,'modifier_pos',lemmatize=False)
            cdsm=relemjoin(cdsm,'head_pos')
            
            phrases=pd.melt(phrases,id_vars=['modifier','head','decade','count'],value_vars=['l1_pos','l2_pos','l3_pos'])
            compounds=pd.melt(cdsm,id_vars=['modifier','head','decade','count'],value_vars=['l1_pos','l2_pos','l3_pos'])
            modifiers=pd.melt(cdsm,id_vars=['modifier','decade','count'],value_vars=['head','l1_pos','l2_pos','l3_pos'])
            heads=pd.melt(cdsm,id_vars=['head','decade','count'],value_vars=['modifier','l1_pos','l2_pos','l3_pos'])
            return phrases,compounds,modifiers,heads
            

def context_reducer(df):
    if len(df)==0:
        return df
    df["variable"]=df["variable"].str.replace(r"_pos","")
    df["context"],df["context_pos"]=df['value'].str.split('_', 1).str
    df.replace(spelling_replacement,inplace=True)
    df=df.loc[df.context_pos.isin(["noun","adj","adv","verb"])]
    df.replace(adv_replacement,inplace=True)
    df['context_pos']=df['context_pos'].str[0]
    if len(df)==0:
        return df
    df['context']=np.vectorize(lemma_maker)(df['context'], df['context_pos'])
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
    
    words_df=pd.concat([rightgram,mid1gram,mid2gram,mid3gram,leftgram],ignore_index=True)
    words_df.dropna(inplace=True)
    words_df=words_df.groupby(['word','context','decade'])['count'].sum().to_frame()
    words_df.reset_index(inplace=True)
    words_df.decade=words_df.decade.astype("int32")
    return words_df

def cdsm_phrase_reducer(df):
    rightgram=syntactic_reducer(df,align="right",level="phrase")
    rightgram=context_reducer(rightgram)
    
    mid1gram=syntactic_reducer(df,align="mid1",level="phrase")
    mid1gram=context_reducer(mid1gram)
    
    mid2gram=syntactic_reducer(df,align="mid2",level="phrase")
    mid2gram=context_reducer(mid2gram)
    
    leftgram=syntactic_reducer(df,align="left",level="phrase")
    leftgram=context_reducer(leftgram) 
    
    phrase_df=pd.concat([rightgram,mid1gram,mid2gram,leftgram],ignore_index=True)
    phrase_df.dropna(inplace=True)
    phrase_df=phrase_df.groupby(['modifier','head','context','decade'])['count'].sum().to_frame()
    phrase_df.reset_index(inplace=True)
    phrase_df.decade=phrase_df.decade.astype("int32")
    return phrase_df


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
    
    compounds=pd.concat([compound_rightgram,compound_mid1gram,compound_mid2gram,compound_leftgram],ignore_index=True)
    modifiers=pd.concat([modifier_rightgram,modifier_mid1gram,modifier_mid2gram,modifier_leftgram],ignore_index=True)
    heads=pd.concat([head_rightgram,head_mid1gram,head_mid2gram,head_leftgram],ignore_index=True)

    phrases=pd.concat([phrase_rightgram,phrase_mid1gram,phrase_mid2gram,phrase_leftgram],ignore_index=True)
    phrases.dropna(inplace=True)
    phrases=phrases.groupby(['modifier','head','context','decade'])['count'].sum().to_frame()
    phrases.reset_index(inplace=True)
    phrases.decade=phrases.decade.astype("int32")
    
    compounds.dropna(inplace=True)
    compounds=compounds.groupby(['modifier','head','context','decade'])['count'].sum().to_frame()
    compounds.reset_index(inplace=True)
    compounds.decade=compounds.decade.astype("int32")
    
    modifiers.dropna(inplace=True)
    modifiers=modifiers.groupby(['modifier','context','decade'])['count'].sum().to_frame()
    modifiers.reset_index(inplace=True)
    modifiers.decade=modifiers.decade.astype("int32")
    
    heads.dropna(inplace=True)
    heads=heads.groupby(['head','context','decade'])['count'].sum().to_frame()
    heads.reset_index(inplace=True)
    heads.decade=heads.decade.astype("int32")
    return compounds,modifiers,heads,phrases


def parallelize_dataframe(df, context_type="independant_word",num_cores = 70):
    num_partitions = num_cores
    df_split = np.array_split(df, num_partitions)
    print("Done splitting the datasets")
    pool = Pool(num_cores)

    cur_time=time.time()
    print("Starting parallelizing")
    if context_type=="dependant":

        results=pool.map_async(cdsm_reducer,df_split)
        pool.close()
        pool.join()

        results=results.get()

        
        print("Done parallelizing")
        print("Total time taken",round(time.time()-cur_time),"secs")
        compound_list = [ result[0] for result in results]
        compounds=pd.concat(compound_list,ignore_index=True)
        compounds=compounds.groupby(['modifier','head','context','decade'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)
        
        if not isfile("/data/dharp/compounding/datasets/compounds.csv"):
            compounds.to_csv("/data/dharp/compounding/datasets/compounds.csv",sep="\t",index=False)
        else:
            compounds.to_csv("/data/dharp/compounding/datasets/compounds.csv", mode='a', header=False,index=False)
        
        
        modifier_list = [ result[1] for result in results]
        modifiers=pd.concat(modifier_list,ignore_index=True)
        modifiers=modifiers.groupby(['modifier','context','decade'])['count'].sum().to_frame()
        modifiers.reset_index(inplace=True)

        if not isfile("/data/dharp/compounding/datasets/modifiers.csv"):
            modifiers.to_csv("/data/dharp/compounding/datasets/modifiers.csv",sep="\t",index=False)
        else:
            modifiers.to_csv("/data/dharp/compounding/datasets/modifiers.csv", mode='a', header=False,index=False)
        
        head_list = [ result[2] for result in results]
        heads=pd.concat(head_list,ignore_index=True)
        heads=heads.groupby(['head','context','decade'])['count'].sum().to_frame()
        heads.reset_index(inplace=True)

        if not isfile("/data/dharp/compounding/datasets/heads.csv"):
            heads.to_csv("/data/dharp/compounding/datasets/heads.csv",sep="\t",index=False)
        else:
            heads.to_csv("/data/dharp/compounding/datasets/heads.csv", mode='a', header=False,index=False)
            
        phrase_list = [ result[3] for result in results]
        phrases=pd.concat(phrase_list,ignore_index=True)
        phrases=phrases.groupby(['modifier','head','context','decade'])['count'].sum().to_frame()
        phrases.reset_index(inplace=True)
        
        if not isfile("/data/dharp/compounding/datasets/phrases.csv"):
            phrases.to_csv("/data/dharp/compounding/datasets/phrases.csv",sep="\t",index=False)
        else:
            phrases.to_csv("/data/dharp/compounding/datasets/phrases.csv", mode='a', header=False,index=False)

    elif context_type=="independant_word":
        words_list=[]
        results=pool.map_async(cdsm_word_reducer,df_split)
  
        
        pool.close()
        pool.join()
        print("Done parallelizing")
        print("Total time taken",round(time.time()-cur_time),"secs")
        word_list=results.get()
        words = pd.concat(words_list,ignore_index=True)
        words=words.groupby(['word','context','decade'])['count'].sum().to_frame()
        words.reset_index(inplace=True)
        
    else:
        phrase_list=[]
        results=pool.map_async(cdsm_phrase_reducer,df_split)

        
        pool.close()
        pool.join()
        print("Done parallelizing")
        print("Total time taken",round(time.time()-cur_time),"secs")
        phrase_list=results.get()
        phrases=pd.concat(phrase_list,ignore_index=True)
        phrases=phrases.groupby(['modifier','head','context','decade'])['count'].sum().to_frame()
        phrases.reset_index(inplace=True)
        
        if not isfile("/data/dharp/compounding/datasets/phrases.csv"):
            phrases.to_csv("/data/dharp/compounding/datasets/phrases.csv",sep="\t",index=False)
        else:
            phrases.to_csv("/data/dharp/compounding/datasets/phrases.csv", mode='a', header=False,index=False)
              
    print("Done concatenations \n")

    
    
def pkl_reader(files,size=None):
    cur_time=time.time()
    print("Reading the pickle files")
    pkl_list=[]
    if size is None:
        for f in files:
            temp_df=pd.read_pickle(f)
            if temp_df.empty==True:
                print("File ",f," is empty")
                continue
            pkl_list.append(temp_df)
    else:
        total_size=0
        for f in files:
            if total_size>size:
                break
            temp_df=pd.read_pickle(f)
            if temp_df.empty==True:
                print("File ",f," is empty")
                continue
            total_size+=temp_df.shape[0]
            pkl_list.append(temp_df)
    print("Done reading the files \n")
    entire_df=pd.concat(pkl_list,ignore_index=True)
    print("Done concatenating the datasets \n")
    print("Total time taken",round(time.time()-cur_time),"secs \n")
    return entire_df.head(size)


if __name__ == "__main__":
    #num_partitions=20
    num_cores=20
    #entire_df=pkl_reader(pkl_files)
    #partitioned_dfs = np.array_split(entire_df, num_partitions)
    #for num,curr_df in enumerate(partitioned_dfs):
    #    print("Current list = ",num+1)
    #    print("Compounds")
    #    parallelize_dataframe(curr_df,context_type="dependant",num_cores=num_cores)
    #    print("Phrases")
    #    parallelize_dataframe(curr_df,context_type="phrase",num_cores=num_cores)
    for f in files:
        print(f,'is read in')
        tmp_df=pd.read_hdf(f, key='df')
        print("Compounds and Phrases")
        parallelize_dataframe(tmp_df,context_type="dependant",num_cores=num_cores)
    print("Done writing to files")
