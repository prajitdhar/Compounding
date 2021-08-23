import pandas as pd
import fasttext
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import csv
import fastparquet
import spacy
import glob, os
import re


import argparse


parser = argparse.ArgumentParser(description='Program to prepare Reddit 5-grams for extracting compounds')

parser.add_argument('--infile', type=str, help='name of file for which to extract data')
parser.add_argument('--outfilename', type=str, help='name of the output file to store')

args = parser.parse_args()

to_save_path='/mnt/dhr/CreateChallenge_ICC_0821/reddit-5-grams-compound-type'
#keep_string=r"(.+_(NOUN|ADV|VERB|ADJ|X|PRT|CONJ|PRON|DET|ADP|NUM|\.)|_END_)\s*"
nn='(?!(?:NOUN|PROPN)).*'
comp='(?:ADJ|NOUN|PROPN)\s(?:NOUN|PROPN)'
word='.*'
n1=f'^{comp}\s{nn}\s{comp}$'
n2=f'^{comp}\s{nn}\s{word}\s{word}$'
n3=f'^{nn}\s{comp}\s{nn}\s{word}$'
n4=f'^{word}\s{nn}\s{comp}\s{nn}$'
n5=f'^{word}\s{word}\s{nn}\s{comp}$'

fmodel = fasttext.load_model('/mnt/dhr/CreateChallenge_ICC_0821/lid.176.bin')
nlp = spacy.load('en_core_web_lg')

def sent_maker(sent_lst):
    ret_sents=[]
    g_pos=[]
    for sent in sent_lst:
        cur_words=[]
        pos_sent=[]
        # probs don't need this
        sent=sent.replace(' _SPACE','x_SPACE')
        for word_pos in sent.split(' '):
            try:
                word,pos=word_pos.rsplit('_',1)
            except:
                print("No POS tag", sent)
                continue
            cur_words.append(word)
            pos_sent.append(pos)
            cur_sent=' '.join(cur_words)
            cur_pos=' '.join(pos_sent)
        ret_sents.append(cur_sent)
        g_pos.append(cur_pos)
    return ret_sents,g_pos

def ner_lemma_reducer(sent):
    ner_sent=[]
    lemma=[]
    pos=[]
    #parse=[]
    is_comp=False
    ner_token=[]
    ner_length=[]
    ner=[]
    parsed_sent=nlp(sent)
    for token in parsed_sent:
        #parse.append(token.text)
        lemma.append(token.lemma_)
        pos.append(token.pos_)
        if token.ent_type_=="":
            to_add="NONNER"
        else:
            to_add=token.ent_type_
        ner_token.append(to_add)
        if token.dep_=="compound":
            is_comp=True
    #print(parse)
    #parse_sent=' '.join(parse)
    lemma_sent=' '.join(lemma)
    pos_sent=' '.join(pos)
    ner_token_sent=' '.join(ner_token)
    #dep_sent=' '.join(dep)
    ner_length=0
    if parsed_sent.ents:
        for ent in parsed_sent.ents:
            #cur_ner=
            #cur_ner='_'.join([str(ent.start_char), str(ent.end_char), ent.label_])
            ner_length+=ent.end_char-ent.start_char
            #ner.append(cur_ner)
    #else:
        #ner.append("")
    ner_sent=' '.join(ner)
    
    return ner_token_sent,ner_length,lemma_sent,pos_sent,is_comp

def str_joiner(df):
    #print(df)
    new_df=pd.DataFrame()
    try:
        new_df[['l1','l2','l3','l4','l5']]=df.lemma_sent.str.split(" ",expand=True)
        new_df[['p1','p2','p3','p4','p5']]=df.pos_sent.str.split(" ",expand=True)
    except:
        print(df)
    new_df['lemma_pos']=new_df.l1+"_"+new_df.p1+" "+\
                        new_df.l2+"_"+new_df.p2+" "+\
                        new_df.l3+"_"+new_df.p3+" "+\
                        new_df.l4+"_"+new_df.p4+" "+\
                        new_df.l5+"_"+new_df.p5
    return new_df['lemma_pos']

def index_processor(df):
    df.reset_index(inplace=True,drop=True)
    ret_lst=sent_maker(df.old_index)
    
    df['sent']=ret_lst[0]
    df['g_pos']=ret_lst[1]
    
    results=np.vectorize(ner_lemma_reducer)(df.sent.values)
    results_df=pd.DataFrame(results)
    results_df=results_df.transpose()
    #results_df.columns=ner_token_sent,ner_length,lemma_sent,pos_sent,is_comp
    results_df.columns=['ner_token_sent','ner_length','lemma_sent','pos_sent','is_comp']

    results_df=results_df.loc[~results_df.ner_token_sent.str.contains("PERSON PERSON")]

    index_df=pd.concat([df,results_df],axis=1,ignore_index=False)

    index_df['nwords']=index_df.pos_sent.str.count(' ').add(1)
    index_df=index_df.loc[index_df.nwords==5]
    
    index_df.lemma_sent=index_df.lemma_sent.str.lower()
    #index_df.pos_sent=index_df.pos_sent.str.replace('PROPN','NOUN',regex=False)
    #index_df.pos_sent=index_df.pos_sent.str.replace('AUX','VERB',regex=False)
    #index_df.pos_sent=index_df.pos_sent.str.replace('CCONJ','CONJ',regex=False)
    #index_df.g_pos=index_df.g_pos.str.replace('.','PUNCT',regex=False)
    #index_df.g_pos=index_df.g_pos.str.replace('PRT','ADP',regex=False)
    if index_df.shape[0]==0:
        return index_df
    index_df['lemma_pos']=str_joiner(index_df)
    index_df['nX']=index_df.pos_sent.str.count('X')-index_df.pos_sent.str.count('AUX')
    index_df=index_df.loc[~(index_df.nX>1)]
    
    index_df['ner_perc']=index_df.ner_length/index_df.sent.str.len()
   
    index_df['comp_class']=0

    index_df.loc[index_df.pos_sent.str.contains(n1),'comp_class']=1
    index_df.loc[~(index_df.pos_sent.str.contains(n1))& index_df.pos_sent.str.contains(n2),'comp_class']=2
    index_df.loc[index_df.pos_sent.str.contains(n3),'comp_class']=3
    index_df.loc[index_df.pos_sent.str.contains(n4),'comp_class']=4
    index_df.loc[~(index_df.pos_sent.str.contains(n1))& index_df.pos_sent.str.contains(n5),'comp_class']=5
    index_df.drop(['g_pos','nwords','nX','lemma_sent','ner_length'],axis=1,inplace=True)

    return index_df

CHUNKSIZE=10_000_000
dfs = pd.read_csv(args.infile, chunksize=CHUNKSIZE) #fivegram_pos, year, count

for i,df in enumerate(dfs):
    index_df=df.groupby(['fivegram_pos'])['count'].sum().reset_index()
    index_df.columns=['old_index','total_count']

    with open(f'{to_save_path}/reddit_stats.txt','a') as f:
        f.write(str(index_df.memory_usage()[0]/(1024*1024))+'\n')

    if index_df.shape[0]!=0:
        num_partitions=round(0.95*mp.cpu_count())
        cur_time=time.time()
        df_split = np.array_split(index_df, num_partitions)
        pool = Pool(num_partitions)
        print('Started parallelization')
        results=pool.map_async(index_processor,df_split)
        pool.close()
        pool.join()

        print(f'Total time taken {round(time.time()-cur_time)/60} min')
        curr_df_list=results.get()
        new_index_df=pd.concat(curr_df_list,ignore_index=True)
        if new_index_df.shape[0]!=0:
            words_df = new_index_df.loc[new_index_df.pos_sent.str.contains('(?:NOUN|PROPN)')].reset_index(drop=True)
            # words_df['nner']=words_df.ner_sent.str.count(' ').add(1)
            words_df.comp_class = words_df.comp_class.astype('int32')

            #print(df)
            #print(words_df)

            words = pd.merge(df, words_df, left_on='fivegram_pos', right_on='old_index', how='right')
            words = words.groupby(['lemma_pos', 'pos_sent', 'year', 'comp_class', 'ner_token_sent', 'ner_perc', 'is_comp'])[
                'count'].sum().to_frame()
            words.reset_index(inplace=True)

            outfilename = args.infile.split("/")[-1].split(".")[0]
            words.to_pickle(f'{to_save_path}/{outfilename}_{i}.pkl')