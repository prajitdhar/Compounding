#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
#import fasttext
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
import seaborn as sns



# In[5]:


# fmodel = fasttext.load_model('C:/Users/glori/Documents/Persönliches/#PhD_local/codeCreateChallenge_Lonneke/lid.176.bin')
nlp = spacy.load('en_core_web_lg')


# In[3]:


keep_string=r"(.+_(NOUN|ADV|VERB|ADJ|X|PRT|CONJ|PRON|DET|ADP|NUM|\.)|_END_)\s*"
large_files=['a_','an','of','to','in','ad','wh','be','ha','is','co','wa','he','no','it','wi','fo','re','as','on','we','punctuation','th','ma','pr','ar','ip','sh','ca','so','hi','bu','al','se','de','by','wo','st','fr','di','mo','su','at','or','yo','me','li','pa','do','ex','le','pe','po','if','ne','fi','un','fa','sa','ch','la','lo','ac','ho','mu','go','si','en','ev','tr']
nn='(?!(?:NOUN|PROPN)).*'
comp='(?:NOUN|PROPN)\s(?:NOUN|PROPN)'
word='.*'
ner_cats=['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
n1=f'^{comp}\s{nn}\s{comp}$'
n2=f'^{comp}\s{nn}\s{word}\s{word}$'
n3=f'^{nn}\s{comp}\s{nn}\s{word}$'
n4=f'^{word}\s{nn}\s{comp}\s{nn}$'
n5=f'^{word}\s{word}\s{nn}\s{comp}$'


# In[4]:


done_files=[]
cur_dir='/data/dharp/compounds/datasets/google/'
for file in glob.glob(f"{cur_dir}*.pkl"):
    done_files.append(file.split('.')[0].split('/')[-1][:2])

done_bigrams=list(set(done_files))
fivegram_list='0 1 2 3 4 5 6 7 8 9 a_ aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar as at au av aw ax ay az b_ ba bb bc bd be bf bg bh bi bj bk bl bm bn bo bp bq br bs bt bu bv bw bx by bz c_ ca cb cc cd ce cf cg ch ci cj ck cl cm cn co cp cq cr cs ct cu cv cw cx cy cz d_ da db dc dd de df dg dh di dj dk dl dm dn do dp dq dr ds dt du dv dw dx dy dz e_ ea eb ec ed ee ef eg eh ei ej ek el em en eo ep eq er es et eu ev ew ex ey ez f_ fa fb fc fd fe ff fg fh fi fj fk fl fm fn fo fp fq fr fs ft fu fv fw fx fy fz g_ ga gb gc gd ge gf gg gh gi gj gk gl gm gn go gp gq gr gs gt gu gv gw gx gy gz h_ ha hb hc hd he hf hg hh hi hj hk hl hm hn ho hp hq hr hs ht hu hv hw hx hy hz i_ ia ib ic id ie if ig ih ii ij ik il im in io ip iq ir is it iu iv iw ix iy iz j_ ja jb jc jd je jf jg jh ji jj jk jl jm jn jo jp jq jr js jt ju jv jw jx jy jz k_ ka kb kc kd ke kf kg kh ki kj kk kl km kn ko kp kq kr ks kt ku kv kw kx ky kz l_ la lb lc ld le lf lg lh li lj lk ll lm ln lo lp lq lr ls lt lu lv lw lx ly lz m_ ma mb mc md me mf mg mh mi mj mk ml mm mn mo mp mq mr ms mt mu mv mw mx my mz n_ na nb nc nd ne nf ng nh ni nj nk nl nm nn no np nq nr ns nt nu nv nw nx ny nz o_ oa ob oc od oe of og oh oi oj ok ol om on oo op oq or os ot other ou ov ow ox oy oz p_ pa pb pc pd pe pf pg ph pi pj pk pl pm pn po pp pq pr ps pt pu punctuation pv pw px py pz q_ qa qb qc qd qe qf qg qh qi qj ql qm qn qo qp qq qr qs qt qu qv qw qx qy qz r_ ra rb rc rd re rf rg rh ri rj rk rl rm rn ro rp rq rr rs rt ru rv rw rx ry rz s_ sa sb sc sd se sf sg sh si sj sk sl sm sn so sp sq sr ss st su sv sw sx sy sz t_ ta tb tc td te tf tg th ti tj tk tl tm tn to tp tq tr ts tt tu tv tw tx ty tz u_ ua ub uc ud ue uf ug uh ui uj uk ul um un uo up uq ur us ut uu uv uw ux uy uz v_ va vb vc vd ve vf vg vh vi vj vk vl vm vn vo vp vq vr vs vt vu vv vw vx vy vz w_ wa wb wc wd we wf wg wh wi wj wk wl wm wn wo wp wq wr ws wt wu wv ww wx wy wz x_ xa xb xc xd xe xf xg xh xi xj xk xl xm xn xo xp xq xr xs xt xu xv xw xx xy xz y_ ya yb yc yd ye yf yg yh yi yj yk yl ym yn yo yp yq yr ys yt yu yv yw yx yy yz z_ za zb zc zd ze zf zg zh zi zj zk zl zm zn zo zp zq zr zs zt zu zv zw zx zy zz'
fivegram_list=fivegram_list.split(" ")
to_do_list=set(fivegram_list).difference(done_bigrams)
done_list='0 1 2 3 4 5 6 7 8 9 other punctuation'
done_list=done_list.split(" ")
to_do_list=list(to_do_list.difference(done_list))
to_do_list


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


def lang_tagger(parsed_sent):
    labels,confs=fmodel.predict(parsed_sent,k=-1,threshold=0.1)
    lang_list=delist_lang(labels)    
    significance_list=significance(confs)
    assert len(lang_list)==len(significance_list)
    return lang_list,significance_list


# In[9]:


def str_joiner(df):
    #print(df)
    new_df=pd.DataFrame()
    try:
        new_df[['l1','l2','l3','l4','l5']]=df.lemma_sent.str.split(" ",expand=True)
        new_df[['p1','p2','p3','p4','p5']]=df.pos_sent.str.split(" ",expand=True)
    except:
        print(df)
    new_df['lemma_pos']=new_df.l1+"_"+new_df.p1+" "+                        new_df.l2+"_"+new_df.p2+" "+                        new_df.l3+"_"+new_df.p3+" "+                        new_df.l4+"_"+new_df.p4+" "+                        new_df.l5+"_"+new_df.p5
    return new_df['lemma_pos']


# In[10]:


def trial(df):
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


# In[11]:


def large_df_processor(letter,num_partitions):
    
    #CHUNKSIZE = 1_000_000_000
    CHUNKSIZE=10_000_000
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
        results=pool.map_async(trial,df_split)
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
        return words_df
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


# In[12]:


num_partitions=round(0.9*mp.cpu_count())
cur_df=large_df_processor('co',num_partitions)


# In[16]:


cur_df.loc[(cur_df.is_comp==True )& (cur_df.lemma_pos.str.contains('coffee'))].sample(30)


# Keep PROP NOUN
# 
# Change Dep to Is_Comp
# 
# (later) remove compounds based on PROPN, NER

# In[19]:


cur_df.is_comp.value_counts(normalize=True)


# In[21]:


cur_df.comp_class.value_counts(normalize=True)


# cur_df.loc[(cur_df.comp_class==0) & (cur_df.is_comp==True)].sample(30)

# new_index_df['sent_len']=new_index_df.parse_sent.str.len()
# new_index_df['ner_ratio']=new_index_df.ner_length/new_index_df.sent_len
# new_index_df.loc[new_index_df.ner_ratio>=0.5]

# In[20]:


for letter in to_do_list:
    #num_partitions=250
    path_loc="http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-5gram-20120701-"+letter+".gz"
    df   = pd.read_csv(path_loc, compression='gzip', header=None, sep="\t", quoting=csv.QUOTE_NONE,usecols=[0,1,2])    
   
    cur_time=time.time()
    df.columns=['fivegram_pos','year','count']
    #df=df.loc[df.year>=1800]
    index_df=df.groupby(['fivegram_pos'])['count'].sum().reset_index()
    index_df.columns=['old_index','total_count']
    index_df=index_df.loc[index_df.old_index.str.match("^"+keep_string*5+"$",na=False)]

    old_types=index_df.shape[0]
    old_tokens=index_df.total_count.sum()
    #df_split = np.array_split(index_df, num_partitions)
    #pool = Pool(num_partitions)
    #print('Started parallelization')
    #results=pool.map_async(trial,df_split)
    #pool.close()
    #pool.join()
        
        
    #  curr_df_list=results.get()
    #new_index_df=pd.concat(curr_df_list,ignore_index=True)
    new_index_df=trial(index_df)
    new_types=new_index_df.shape[0]
    new_tokens=new_index_df.total_count.sum()

    print(f'Change in number of types: {(new_types-old_types)/old_types*100}')
    print(f'Change in number of tokens: {(new_tokens-old_tokens)/old_tokens*100}')
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

    words.to_pickle(f'/data/dharp/compounds/datasets/google/{letter}{0+1}.pkl')
        #phrases_df=words_df.loc[words_df.pos_sent.str.contains('NOUN NOUN')].reset_index(drop=True)
        #phrases=pd.merge(df,phrases_df,left_on='fivegram_pos',right_on='old_index',how='right')
        #phrases=phrases.groupby(['lemma_sent','year','pos_sent','comp_class','ner_sent'])['count'].sum().to_frame()
        #phrases.reset_index(inplace=True)

        #comp_df=phrases_df.loc[phrases_df.comp_class!=0].reset_index(drop=True)
        #compounds=pd.merge(df,comp_df,left_on='fivegram_pos',right_on='old_index',how='right')
        #compounds=compounds.groupby(['lemma_sent','year','pos_sent','comp_class','ner_sent'])['count'].sum().to_frame()
        #compounds.reset_index(inplace=True)

    print(f'Total time taken for letter {letter}: {round(time.time()-cur_time)} secs')
    with open(f'/data/dharp/compounds/datasets/stats/{letter}{0+1}.txt','w') as f:
        f.write(f'{letter}\t{0+1}\t{ntypes}\t{ntokens}\t{ncomptypes}\t{comp_count}\n')


# In[ ]:


words.is_comp.value_counts()


# In[16]:


words.loc[words.year<1600]


# In[14]:


words.loc[words.comp_class==3].sample(30)


# In[31]:


new_index_df.loc[new_index_df.comp_class!=0].ner_sent.value_counts(normalize=True)*100


# In[29]:


new_index_df.loc[(new_index_df.ner_length>1)&(new_index_df.comp_class!=0)]


# In[28]:


for ner in ner_cats:
    print(ner)
    print(new_index_df.shape[0])
    display(new_index_df.loc[(new_index_df.ner_sent.str.contains(ner))&(new_index_df.comp_class!=0)].sample(30))

