import pandas as pd
import os
import spacy
import re
import time
import torch
from thinc.api import set_gpu_allocator, require_gpu
import gc
import pickle
import argparse

parser = argparse.ArgumentParser(description='Program to process the coha pickle files and store the 5-gram files akin to google N-grams V3')

parser.add_argument('--input', type=str,
                    help='location of the directory with the coha pickle files')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')

parser.add_argument('--gpuid', type=int,
                    help='which gpu to use')
#parser.add_argument('--start_from', type=str,
#                    help='which decade to resume from')

args = parser.parse_args()

word='.*'

nn='(?!(?:NOUN|PROPN)).*'
comp='(?:NOUN|PROPN)\s(?:NOUN|PROPN)'

n1=f'^{comp}\s{nn}\s{comp}$'
n2=f'^{comp}\s{nn}\s{word}\s{word}$'
n3=f'^{nn}\s{comp}\s{nn}\s{word}$'
n4=f'^{word}\s{nn}\s{comp}\s{nn}$'
n5=f'^{word}\s{word}\s{nn}\s{comp}$'


spacy.prefer_gpu()
set_gpu_allocator("pytorch")
require_gpu(args.gpuid)


_dir = args.input

coha_files = sorted(os.listdir(_dir))
print(len(coha_files))



def year_processor(file_id,parser):
    print(file_id)

    cur_year=int(file_id.split('_')[1])
    sents=cur_decade_pkl[file_id]
    print(f'Number of sentences {len(sents)}')
    print("Running parser")
    
    docs = list(parser.pipe(sents))
    print("Done running parser")
    torch.cuda.empty_cache()
    
    tokens=[]
    lemmas=[]
    pos=[]
    deps=[]
    is_stop=[]
    ner=[]
    for doc in docs:
        for token in doc:
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            pos.append(token.pos_)
            deps.append(token.dep_)
            is_stop.append(token.is_stop)
            ner.append(token.ent_type_)

    df_dicts={'token':tokens,'lemma':lemmas,'pos':pos,'dep':deps,'ner':ner,'year':cur_year}
    cur_df=pd.DataFrame.from_dict(df_dicts)
    if cur_df.shape[0]==0:
        return None
    cur_df=cur_df.loc[~(cur_df.pos=="SPACE")].reset_index(drop=True)
    cur_df['lem_pos']=cur_df.lemma.str.lower()+"_"+cur_df.pos
    cur_df['n_comp']=False
    cur_df.loc[(cur_df.pos.isin(['PROPN','NOUN'])) & (cur_df.dep=='compound'),'n_comp']=True
    cur_df['lemma_pos']=cur_df.lem_pos.shift(0)+ ' ' + cur_df.lem_pos.shift(-1)+ ' ' + cur_df.lem_pos.shift(-2)+ ' ' + cur_df.lem_pos.shift(-3)+ ' ' + cur_df.lem_pos.shift(-4)
    cur_df['pos_sent']=cur_df.pos.shift(0)+ ' ' + cur_df.pos.shift(-1)+ ' ' + cur_df.pos.shift(-2)+ ' ' + cur_df.pos.shift(-3)+ ' ' + cur_df.pos.shift(-4)
    cur_df['num_comp']=False
    cur_df.loc[cur_df.n_comp.shift(0)|cur_df.n_comp.shift(-1)|cur_df.n_comp.shift(-2)|cur_df.n_comp.shift(-3)|cur_df.n_comp.shift(-4),'num_comp']=True
    cur_df['tokens']=cur_df.token.shift(0)+ ' ' + cur_df.token.shift(-1)+ ' ' + cur_df.token.shift(-2)+ ' ' + cur_df.token.shift(-3)+ ' ' + cur_df.token.shift(-4)
    cur_df['c_ner_sent']=""
    mask=(cur_df.ner!="") & (cur_df.num_comp==True)
    cur_df.loc[mask,'c_ner_sent']=cur_df.loc[mask,'ner']
    cur_df['comp_ner_sent']=cur_df.c_ner_sent.shift(0)+ ' ' + cur_df.c_ner_sent.shift(-1)+ ' ' + cur_df.c_ner_sent.shift(-2)+ ' ' + cur_df.c_ner_sent.shift(-3)+ ' ' + cur_df.c_ner_sent.shift(-4)
    cur_df.comp_ner_sent=cur_df.comp_ner_sent.str.strip()
    cur_df.dropna(inplace=True)
    cur_df['nX']=cur_df.lemma_pos.str.count('_X')-cur_df.lemma_pos.str.count('_AUX')
    cur_df=cur_df.loc[cur_df.nX!=5]
    cur_df['comp_class']=0
    cur_df.loc[cur_df.pos_sent.str.contains(n1),'comp_class']=1
    cur_df.loc[~(cur_df.pos_sent.str.contains(n1))& cur_df.pos_sent.str.contains(n2),'comp_class']=2
    cur_df.loc[cur_df.pos_sent.str.contains(n3),'comp_class']=3
    cur_df.loc[cur_df.pos_sent.str.contains(n4),'comp_class']=4
    cur_df.loc[~(cur_df.pos_sent.str.contains(n1))& cur_df.pos_sent.str.contains(n5),'comp_class']=5
    cur_df['count']=1
    cur_df=cur_df.groupby(['lemma_pos','tokens','year','comp_class','num_comp','comp_ner_sent'])['count'].sum().to_frame().reset_index()
    print("\n")
    return cur_df



def part_processor(df,cur_decade):
    file_list=df.fname.to_list()
    cur_time=time.time()
    parser = spacy.load('en_core_web_trf')
    parser.add_pipe("doc_cleaner")
    parser.max_length=10_000_000
    
    
    df_list=[]
    
    for i,cur_file in enumerate(file_list):
        print(f'File {i+1} out of {len(file_list)}')
        df_list.append(year_processor(cur_file,parser))
    
    del parser
    torch.cuda.empty_cache()
    gc.collect()
    cur_decade_df=pd.concat(df_list,ignore_index=True,sort=True)
    print(f'Shape of dataframe before grouping :{cur_decade_df.shape}')
    cur_decade_df=cur_decade_df.groupby(['lemma_pos','tokens','year','comp_class','num_comp','comp_ner_sent'])['count'].sum().to_frame().reset_index()

    print(f'Shape of dataframe after grouping :{cur_decade_df.shape}')

    print(f"Done processing for decade {cur_decade}, group {df.iloc[0].fcat}")
    print(f"Total time taken for decade {cur_decade} : {round(time.time()-cur_time)} secs")
    return cur_decade_df



for cur_pkl in coha_files:
    dec_time=time.time()
    cur_decade=cur_pkl.split('.')[0]
    print(cur_decade)
    
    with open(f"{_dir}/{cur_pkl}", "rb" ) as f:
        cur_decade_pkl = pickle.load(f)

    
    df_list=[]
    names=[]
    sents=[]

    for key,val in cur_decade_pkl.items():
        names.append(key)
        sents.append(len(val))
    cur_decade_df=pd.DataFrame({'fname':names,'flen':sents})
    cur_decade_df=cur_decade_df.loc[cur_decade_df.flen>0]
    cur_decade_df.sort_values(by=['flen'],ascending=False,inplace=True,ignore_index=True)
    
    maxvalue = 300_000

    lastvalue = 0
    newcum = []
    labels=[]
    cur_label=1
    for row in cur_decade_df.itertuples():
        thisvalue =  row.flen + lastvalue
        if thisvalue > maxvalue:
            thisvalue = 0
            cur_label+=1
        newcum.append(thisvalue)
        labels.append(cur_label)
        lastvalue = thisvalue
    cur_decade_df['fcat']=labels
    
    print(f'Number of interations {cur_decade_df.fcat.nunique()}')
    
    
    cur_decade_df=cur_decade_df.groupby('fcat').apply(lambda x: part_processor(x,cur_decade))
    cur_decade_df.reset_index(drop=True,inplace=True)
    print(f'Shape of dataframe before grouping :{cur_decade_df.shape}')

    cur_decade_df=cur_decade_df.groupby(['lemma_pos','tokens','year','comp_class','num_comp','comp_ner_sent'])['count'].sum().to_frame().reset_index()

    print(f'Shape of dataframe after grouping :{cur_decade_df.shape}')

    print(f"Done processing for decade {cur_decade}")
    
    cur_decade_df.to_pickle(f'{args.output}/{cur_decade}.pkl')
    print(f"Total time taken for decade {cur_decade} : {round(time.time()-dec_time)} secs")