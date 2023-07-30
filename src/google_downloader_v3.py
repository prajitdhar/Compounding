import pandas as pd
import fasttext
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import csv
import spacy
import re
import warnings
import os

from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
detokenizer = Detok()


fasttext.FastText.eprint = lambda x: None
import argparse


parser = argparse.ArgumentParser(description='Program to download google ngram data for version 3')

parser.add_argument('--file', type=int,
                    help='File from google V3, for which data needs to be extracted')
parser.add_argument('--spath', type=str,
                    help='directory where to save output')

args = parser.parse_args()



to_save_path=args.spath

keep_string=r"(.+_(NOUN|ADV|VERB|ADJ|X|PRT|CONJ|PRON|DET|ADP|NUM|\.)|_END_|_START_)\s*"
try_keep_string=r"(.+_(NOUN|ADV|VERB|ADJ|X|PRT|CONJ|PRON|DET|ADP|NUM|\.)|_NOUN_|_ADV_|_VERB_|_ADJ_|_X_|_PRT_|_CONJ_|_PRON_|_DET_|_ADP_|_NUM_|_\._)"

word='.*'

nn='(?!(?:NOUN|PROPN)).*'
nn_comp='(?:NOUN|PROPN)\s(?:NOUN|PROPN)'
an_comp='ADJ\s(?:NOUN|PROPN)'

ner_cats=['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
n1=f'^{nn_comp}\s{nn}\s{nn_comp}$'
n2=f'^{nn_comp}\s{nn}\s{word}\s{word}$'
n3=f'^{nn}\s{nn_comp}\s{nn}\s{word}$'
n4=f'^{word}\s{nn}\s{nn_comp}\s{nn}$'
n5=f'^{word}\s{word}\s{nn}\s{nn_comp}$'

a1=f'^{an_comp}\s{nn}\s{an_comp}$'
a2=f'^{an_comp}\s{nn}\s{word}\s{word}$'
a3=f'^{nn}\s{an_comp}\s{nn}\s{word}$'
a4=f'^{word}\s{nn}\s{an_comp}\s{nn}$'
a5=f'^{word}\s{word}\s{nn}\s{an_comp}$'


c1=f'^{nn_comp}\s{nn}\s{an_comp}$'
c2=f'^{an_comp}\s{nn}\s{nn_comp}$'


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
    dep=[]
    comp_ner_type=[]
    parsed_sent=nlp(sent)
    for token in parsed_sent:
        lemma.append(token.lemma_)
        pos.append(token.pos_)
        dep.append(token.dep_)
        if token.dep_=="compound":
            if token.ent_type_!="":
                comp_ner_type.append(token.ent_type_)

    comp_ner_sent=' '.join(comp_ner_type)
    if len(parsed_sent)<5:
        new_lemma_list=["eos"]*(5-len(parsed_sent))
        new_pos_list=["X"]*(5-len(parsed_sent))
        lemma.extend(new_lemma_list)
        pos.extend(new_pos_list)
        
    comp_ner_sent=' '.join(comp_ner_type)
    lemma_sent=' '.join(lemma)
    pos_sent=' '.join(pos)
    
    dep_sent=' '.join(dep)
        
    num_count=len(re.findall("compound\s(?!compound)", dep_sent))
   
    return lemma_sent,pos_sent,num_count,comp_ner_sent

def year_binner(year,val=10):
    return year - year%val

def lang_tagger(parsed_sent):
    labels,confs=fmodel.predict(parsed_sent,k=-1,threshold=0.1)
    lang_list=delist_lang(labels)    
    significance_list=significance(confs)
    assert len(lang_list)==len(significance_list)
    return lang_list,significance_list


def year_count_split(df):
    trial_df=pd.concat([df.lemma_pos, df.year_counts.str.split("\t", expand=True)], axis=1)
    trial_df=pd.melt(trial_df, id_vars=["lemma_pos"], value_vars=list(range(len(trial_df.columns)-1))).dropna().drop("variable", axis = 1)
    trial_df[['year','count']] = trial_df.value.str.split(",", n=3, expand=True)[[0,1]]
    return trial_df.drop(['value'],axis=1).reset_index(drop=True)

def str_joiner(df):
    #print(df)
    new_df=pd.DataFrame()
    try:
        new_df[['l1','l2','l3','l4','l5']]=df.lemma_sent.str.split(" ",expand=True,n=4)
        new_df[['p1','p2','p3','p4','p5']]=df.pos_sent.str.split(" ",expand=True,n=4)
    except:
        return pd.DataFrame()
    new_df['lemma_pos']=new_df.l1+"_"+new_df.p1+" "+\
                        new_df.l2+"_"+new_df.p2+" "+\
                        new_df.l3+"_"+new_df.p3+" "+\
                        new_df.l4+"_"+new_df.p4+" "+\
                        new_df.l5+"_"+new_df.p5
    return new_df['lemma_pos']

def year_count_split(df):
    trial_df=pd.concat([df.old_index, df.year_counts.str.split("\t", expand=True)], axis=1)
    trial_df=pd.melt(trial_df, id_vars=["old_index"], value_vars=list(range(len(trial_df.columns)-1))).dropna().drop("variable", axis = 1)
    trial_df[['year','count']] = trial_df.value.str.split(",", n=3, expand=True)[[0,1]]
    return trial_df.drop(['value'],axis=1).reset_index(drop=True)


def index_processor(df):
    
    df['sent']=np.vectorize(detokenizer.detokenize)(df.old_index.str.split(" ").values)
    df['sent']=df.sent.str.replace('\s*,\s*',', ',regex=False).copy()
    df['sent']=df.sent.str.replace('\s*\.\s*','. ',regex=False).copy()
    df['sent']=df.sent.str.replace('\s*\?\s*','? ',regex=False).copy()
    df['sent']=df.sent.str.replace('__',' ',regex=False).copy()

    df['sent']=df.sent.str.replace('_START_ ','',regex=False).copy()
    df['sent']=df['sent'].str.replace(' _END_','',regex=False).copy()
     
    #df['sent']=df['sent'].str.replace(r"(.+)'\s(.+)",r"\1'\2",regex=True).copy()
    #df['sent']=df['sent'].str.replace(r"(.+)\s'(.+)",r"\1'\2",regex=True).copy()

    lang_list,significance_list=lang_tagger(df.sent.values.tolist())
    df['lang']=lang_list
    df['lang_conf']=significance_list
    df.lang=df.lang.str.split('_',n=4).str[-1]
    
    df=df.loc[(df.lang=='en') &(df.lang_conf==True)]

    lemma_sent,pos_sent,comp_count,comp_ner_sent=np.vectorize(ner_lemma_reducer)(df.sent.values)
    pd.options.mode.chained_assignment = None
    df['lemma_sent']=lemma_sent
    df['pos_sent']=pos_sent
    df['comp_count']=comp_count
    df['comp_ner_sent']=comp_ner_sent
    
    df['is_comp']=False
    df.loc[df.comp_count!=0,'is_comp']=True
    #results_df=results_df.loc[~results_df.ner_token_sent.str.contains("PERSON PERSON")]

    #index_df=pd.concat([df,results_df],axis=1,ignore_index=True)

    #return results_df,df

    #index_df=index_df.loc[(index_df.lang=='en') &(index_df.lang_conf==True)]

    df['nwords']=df.pos_sent.str.count(' ').add(1).copy()
    
    pd.options.mode.chained_assignment = 'warn'
    df=df.loc[df.nwords==5]
    
    df.lemma_sent=df.lemma_sent.str.lower()

    #index_df.pos_sent=index_df.pos_sent.str.replace('PROPN','NOUN',regex=False)
    #index_df.pos_sent=index_df.pos_sent.str.replace('AUX','VERB',regex=False)
    #index_df.pos_sent=index_df.pos_sent.str.replace('CCONJ','CONJ',regex=False)
    #index_df.g_pos=index_df.g_pos.str.replace('.','PUNCT',regex=False)
    #index_df.g_pos=index_df.g_pos.str.replace('PRT','ADP',regex=False)
    if df.shape[0]==0:
        return pd.DataFrame()
    
    df['lemma_pos']=str_joiner(df)
    df['nX']=df.pos_sent.str.count('X')-df.pos_sent.str.count('AUX')
    df=df.loc[~(df.nX==5)]
       
    df['comp_class']=0

    df.loc[df.pos_sent.str.contains(n1),'comp_class']=1
    df.loc[~(df.pos_sent.str.contains(n1))& df.pos_sent.str.contains(n2),'comp_class']=2
    df.loc[df.pos_sent.str.contains(n3),'comp_class']=3
    df.loc[df.pos_sent.str.contains(n4),'comp_class']=4
    df.loc[~(df.pos_sent.str.contains(n1))& df.pos_sent.str.contains(n5),'comp_class']=5
    
    df.loc[df.pos_sent.str.contains(a1),'comp_class']=6
    df.loc[~(df.pos_sent.str.contains(a1))& df.pos_sent.str.contains(a2),'comp_class']=7
    df.loc[df.pos_sent.str.contains(a3),'comp_class']=8
    df.loc[df.pos_sent.str.contains(a4),'comp_class']=9
    df.loc[~(df.pos_sent.str.contains(a1))& df.pos_sent.str.contains(a5),'comp_class']=10

    df.loc[df.pos_sent.str.contains(c1),'comp_class']=11
    df.loc[df.pos_sent.str.contains(c2),'comp_class']=12

    df.drop(['sent','pos_sent','lang','lang_conf','nwords','nX','lemma_sent'],axis=1,inplace=True)

    index_year_df=year_count_split(df)
    index_df=df.merge(index_year_df, on='old_index',how='right')

    index_df['count']=index_df['count'].astype("int64")
    index_df['year']=index_df['year'].astype("int64")

    index_df['decade']=year_binner(index_df['year'].values,10)
    index_df['decade']=index_df['decade'].astype("int64")
    index_df=index_df.loc[index_df.decade>=1800]
    
    index_df=index_df.groupby(['lemma_pos','decade','comp_class','is_comp','comp_ner_sent'])['count'].sum().to_frame().reset_index()
    return index_df



i=args.file
print(i)

lnk=f'http://storage.googleapis.com/books/ngrams/books/20200217/eng/5-{i:05}-of-19423.gz'
print(lnk)
index_df   = pd.read_csv(lnk, compression='gzip', header=None, sep=u"\u0001", quoting=csv.QUOTE_NONE)    

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=UserWarning)
    index_df[['old_index','year_counts']]=index_df[0].str.split('\t',n=1,expand=True)
    index_df=index_df.loc[~index_df.old_index.str.contains(try_keep_string,na=False,regex=True)]
    index_df.drop(0,axis=1,inplace=True)
    index_df.reset_index(drop=True,inplace=True)

if index_df.shape[0]==0:
    print(f'{i} is empty')
    
else:

    if index_df.shape[0]<10_000:
    
        cur_time=time.time()
        new_index_df=index_processor(index_df)
        print(f'Total time taken {round(time.time()-cur_time)} secs')
    
    else:
        num_partitions=round(0.95*mp.cpu_count())
        cur_time=time.time()
        df_split = np.array_split(index_df, num_partitions)
        pool = Pool(num_partitions)
        print('Started parallelization')
        results=pool.map_async(index_processor,df_split)
        pool.close()
        pool.join()
        
        
        curr_df_list=results.get()
        new_index_df=pd.concat(curr_df_list,ignore_index=True)
        print(f'Total time taken {round(time.time()-cur_time)} secs')

    
    if new_index_df.shape[0]!=0:
        
        decade_lists=new_index_df.decade.unique().tolist()

        for decade in decade_lists:

            path = f"{to_save_path}/{decade}"

            if not os.path.exists(path):
                os.makedirs(path)
            new_index_df.loc[new_index_df.decade==decade].reset_index(drop=True).to_pickle(f'{path}/{i}.pkl')
        
    else:
        print(f'{i} is empty')

    


