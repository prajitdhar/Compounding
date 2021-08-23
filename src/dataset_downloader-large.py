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
from os.path import isfile

import argparse


parser = argparse.ArgumentParser(description='Program to download google ngram data for a particular bigram and setting')

parser.add_argument('--letter', type=str,
                    help='bigram or letter for which to extract data')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')

#parser.add_argument('--cores', type=int,default=100,
#                    help='Number of cpu cores to use')

parser.add_argument('--chunksize', type=int,default=100_000_000,
                    help='Value of chunksize to read datasets in')


args = parser.parse_args()




fmodel = fasttext.load_model('/data/dharp/packages/lid.176.bin')
nlp = spacy.load('en_core_web_lg')
#nlp = spacy.load('en_core_web_trf')


keep_string=r"(.+_(NOUN|ADV|VERB|ADJ|X|PRT|CONJ|PRON|DET|ADP|NUM|\.)|_END_)\s*"
large_files=['a_','an','of','to','in','ad','wh','be','ha','is','co','wa','he','no','it','wi','fo','re','as','on','we','th','ma','pr','ar','ip','sh','ca','so','hi','bu','al','se','de','by','wo','st','fr','di','mo','su','at','or','yo','me','li','pa','do','ex','le','pe','po','if','ne','fi','un','fa','sa','ch','la','lo','ac','ho','mu','go','si','en','ev','tr']
nn='(?!(?:NOUN|PROPN)).*'
comp='(?:NOUN|PROPN)\s(?:NOUN|PROPN)'
word='.*'
ner_cats=['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
n1=f'^{comp}\s{nn}\s{comp}$'
n2=f'^{comp}\s{nn}\s{word}\s{word}$'
n3=f'^{nn}\s{comp}\s{nn}\s{word}$'
n4=f'^{word}\s{nn}\s{comp}\s{nn}$'
n5=f'^{word}\s{word}\s{nn}\s{comp}$'


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

def index_parser(df):
    df.reset_index(inplace=True,drop=True)
    ret_lst=sent_maker(df.old_index)
    
    df['sent']=ret_lst[0]
    df['g_pos']=ret_lst[1]
    
    results=np.vectorize(ner_lemma_reducer)(df.sent.values)
    results_df=pd.DataFrame(results)
    results_df=results_df.transpose()
    #results_df.columns=ner_token_sent,ner_length,lemma_sent,pos_sent,is_comp
    results_df.columns=['ner_token_sent','ner_length','lemma_sent','pos_sent','is_comp']


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
    index_df.drop(['lang','lang_conf','nwords','nX','lemma_sent','ner_length'],axis=1,inplace=True)
    #print(index_df)
    return index_df


def large_df_processor(letter):
    num_partitions=round(0.9*mp.cpu_count())
    CHUNKSIZE = args.chunksize
    total_df_shape=0
    df_list=[]
    path_loc="http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-5gram-20120701-"+letter+".gz"
    dfs   = pd.read_csv(path_loc, compression='gzip', header=None, sep="\t", quoting=csv.QUOTE_NONE,usecols=[0,1,2],chunksize=CHUNKSIZE)    
    for i,df in enumerate(dfs):


        print(f'Split num {i+1}')        
        cur_time=time.time()
        df.columns=['fivegram_pos','year','count']
        #df=df.loc[df.year>=1800]
        index_df=df.groupby(['fivegram_pos'])['count'].sum().reset_index()
        index_df.columns=['old_index','total_count']
        index_df=index_df.loc[index_df.old_index.str.match("^"+keep_string*5+"$",na=False)]

        df_split = np.array_split(index_df, num_partitions)
        pool = Pool(num_partitions)
        print('Started parallelization')
        results=pool.map_async(index_parser,df_split)
        pool.close()
        pool.join()
        
        
        curr_df_list=results.get()
        new_index_df=pd.concat(curr_df_list,ignore_index=True)
        
        
        print(f'Total time taken for split num {0+1}: {round(time.time()-cur_time)} secs')        
    
        ntypes=new_index_df.shape[0]
        ntokens=new_index_df.total_count.sum()

        types_perc=round(ntypes/df.shape[0]*100,3)
        print(f'Number of types: {ntypes}, perc. of unique types (decade agnostic): {types_perc}%')

        print(f'Number of tokens: {ntokens}, ratio of tokens to types: {round(ntokens/ntypes,3)}')

        ncomptypes=np.sum(new_index_df.comp_class!=0)
        ncomptypes_perc=round(ncomptypes/ntypes*100,3)
        print(f'Number of compounds types: {ncomptypes}, perc. of compound types: {ncomptypes_perc}%')

        comp_count=new_index_df.loc[new_index_df.comp_class!=0,'total_count'].sum()
        comp_count_perc=round(comp_count/ntokens*100,3)
        print(f'Compound count: {comp_count}, perc. of compound tokens: {comp_count_perc}%')

        words_df=new_index_df.loc[new_index_df.pos_sent.str.contains('(?:NOUN|PROPN)')].reset_index(drop=True)
        #words_df['nner']=words_df.ner_sent.str.count(' ').add(1)
        words_df.comp_class=words_df.comp_class.astype('int32')

        words=pd.merge(df,words_df,left_on='fivegram_pos',right_on='old_index',how='right')
        words=words.groupby(['lemma_pos','pos_sent','year','comp_class','ner_token_sent','ner_perc','is_comp'])['count'].sum().to_frame()
        words.reset_index(inplace=True)

        words.to_pickle(f'/data/dharp/compounds/datasets/google/{letter}{i+1}.pkl')
        #phrases_df=words_df.loc[words_df.pos_sent.str.contains('NOUN NOUN')].reset_index(drop=True)
        #phrases=pd.merge(df,phrases_df,left_on='fivegram_pos',right_on='old_index',how='right')
        #phrases=phrases.groupby(['lemma_sent','year','pos_sent','comp_class','ner_sent'])['count'].sum().to_frame()
        #phrases.reset_index(inplace=True)

        #comp_df=phrases_df.loc[phrases_df.comp_class!=0].reset_index(drop=True)
        #compounds=pd.merge(df,comp_df,left_on='fivegram_pos',right_on='old_index',how='right')
        #compounds=compounds.groupby(['lemma_sent','year','pos_sent','comp_class','ner_sent'])['count'].sum().to_frame()
        #compounds.reset_index(inplace=True)

        print(f'Total time taken for letter {letter}: {round(time.time()-cur_time)} secs')
        with open(f'/data/dharp/compounds/datasets/stats/{letter}{i+1}.txt','w') as f:
            f.write(f'{letter}\t{i+1}\t{ntypes}\t{ntokens}\t{ncomptypes}\t{comp_count}\n')
            

    
large_df_processor(args.letter)