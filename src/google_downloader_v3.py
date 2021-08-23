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


parser = argparse.ArgumentParser(description='Program to download google ngram data for version 3')

parser.add_argument('--file', type=int,
                    help='number of file for which to extract data')


args = parser.parse_args()


to_save_path='/data/dharp/compounds/datasets/'
keep_string=r"(.+_(NOUN|ADV|VERB|ADJ|X|PRT|CONJ|PRON|DET|ADP|NUM|\.)|_END_)\s*"
nn='(?!(?:NOUN|PROPN)).*'
comp='(?:ADJ|NOUN|PROPN)\s(?:NOUN|PROPN)'
word='.*'
n1=f'^{comp}\s{nn}\s{comp}$'
n2=f'^{comp}\s{nn}\s{word}\s{word}$'
n3=f'^{nn}\s{comp}\s{nn}\s{word}$'
n4=f'^{word}\s{nn}\s{comp}\s{nn}$'
n5=f'^{word}\s{word}\s{nn}\s{comp}$'

fmodel = fasttext.load_model('/data/dharp/packages/lid.176.bin')
nlp = spacy.load('en_core_web_lg')


def delist_lang(lst):
    lang_lst=[]
    for i,lang in enumerate(lst):
        if not lang:
            lang_lst.append(None)
        else:
            lang_lst.append(lang[0])
    return lang_lst


def significance(lst):
    significance_list=[]
    for l in lst:
        if len(l)>1:
            significance_list.append(abs(l[0]-l[1])/np.mean(l[0]+l[1])>0.1)
            #print(f'{conf[0]} {conf[1]} {abs(conf[0]-conf[1])/np.mean(conf[0]+conf[1])>0.1}')
        else:
            significance_list.append(True)
    return significance_list


def sent_maker(sent_lst):
    ret_sents=[]
    g_pos=[]
    for sent in sent_lst:
        cur_words=[]
        pos_sent=[]
        sent=sent.replace('_END_','@@@_.')
        for word_pos in sent.split(' '):
            word,pos=word_pos.rsplit('_',1)
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

def lang_tagger(parsed_sent):
    labels,confs=fmodel.predict(parsed_sent,k=-1,threshold=0.1)
    lang_list=delist_lang(labels)    
    significance_list=significance(confs)
    assert len(lang_list)==len(significance_list)
    return lang_list,significance_list

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


def year_count_split(df):
    trial_df=pd.concat([df.lemma_pos, df.year_counts.str.split("\t", expand=True)], axis=1)
    trial_df=pd.melt(trial_df, id_vars=["lemma_pos"], value_vars=list(range(len(trial_df.columns)-1))).dropna().drop("variable", axis = 1)
    trial_df[['year','count']] = trial_df.value.str.split(",", n=3, expand=True)[[0,1]]
    return trial_df.drop(['value'],axis=1).reset_index(drop=True)


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

    lang_list,significance_list=lang_tagger(index_df.sent.values.tolist())
    index_df['lang']=lang_list
    index_df['lang_conf']=significance_list
    index_df.lang=index_df.lang.str.split('_',n=4).str[-1]
    index_df=index_df.loc[(index_df.lang=='en') &(index_df.lang_conf==True)]

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
    index_df.drop(['old_index','g_pos','lang','lang_conf','nwords','nX','lemma_sent','ner_length'],axis=1,inplace=True)
    index_year_df=year_count_split(index_df)
    index_df=index_df.merge(index_year_df, on='lemma_pos',how='right')
    index_df=index_df.groupby(['lemma_pos','pos_sent','year','comp_class'])['count'].sum().to_frame().reset_index()
    return index_df



i=args.file
print(i)

lnk=f'http://storage.googleapis.com/books/ngrams/books/20200217/eng/5-{i:05}-of-19423.gz'
print(lnk)
index_df   = pd.read_csv(lnk, compression='gzip', header=None, sep="\n", quoting=csv.QUOTE_NONE)
    
with open(f'{to_save_path}/V3stats.txt','a') as f:
    f.write(str(index_df.memory_usage()[0]/(1024*1024))+'\n')
    
index_df[['old_index','year_counts']]=index_df[0].str.split('\t',n=1,expand=True)
index_df=index_df.loc[index_df.old_index.str.match("^"+keep_string*5+"$",na=False)]
index_df.drop(0,axis=1,inplace=True)

if index_df.shape[0]!=0:
    num_partitions=round(0.95*mp.cpu_count())
    cur_time=time.time()
    df_split = np.array_split(index_df, num_partitions)
    pool = Pool(num_partitions)
    print('Started parallelization')
    results=pool.map_async(index_processor,df_split)
    pool.close()
    pool.join()




    print(f'Total time taken {round(time.time()-cur_time)} secs')
    curr_df_list=results.get()
    new_index_df=pd.concat(curr_df_list,ignore_index=True)
    if new_index_df.shape[0]!=0:
        new_index_df.to_pickle(f'{to_save_path}/googleV3/{i}.pkl')

    


