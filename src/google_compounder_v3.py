import pandas as pd
import os
import numpy as np
import time
import fastparquet
import pickle
import argparse
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat


parser = argparse.ArgumentParser(description='Program to run google compounder for a particular file and setting')

parser.add_argument('--data', type=str,
                    help='location of the parquet file')

parser.add_argument('--word', action='store_true',
                    help='extracting context for words only?')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')
parser.add_argument('--decade', type=str,
                    help='decade to process')


args = parser.parse_args()

context_list = pickle.load( open( f'{args.output}/word_lists/contexts_top50k_old.pkl', "rb" ) )

constituent_list = pickle.load( open( f'{args.output}/word_lists/constituents.pkl', "rb" ) )


compound_id_vars=['modifier','head','count','is_comp','comp_ner_sent']
modifier_id_vars=['modifier','count']
head_id_vars=['head','count']
word_id_vars=['word','count']

compound_id_context_vars=['modifier','head','context','is_comp','comp_ner_sent']
modifier_id_context_vars=['modifier','context']
head_id_context_vars=['head','context']
word_id_context_vars=['word','context']

word='.+_.+'
comp='.+_(?:NOUN|PROPN)\s.+_(?:NOUN|PROPN)'


p2=f'^{comp}\s{word}\s{word}\s{word}$'
p3=f'^{word}\s{comp}\s{word}\s{word}$'
p4=f'^{word}\s{word}\s{comp}\s{word}$'
p5=f'^{word}\s{word}\s{word}\s{comp}$'

adj_noun='.+_(?:ADJ|NOUN|PROPN)'

an1=f'^{adj_noun}\s{word}\s{word}\s{word}\s{word}$'
an2=f'^{word}\s{adj_noun}\s{word}\s{word}\s{word}$'
an3=f'^{word}\s{word}\s{adj_noun}\s{word}\s{word}$'
an4=f'^{word}\s{word}\s{word}\s{adj_noun}\s{word}$'
an5=f'^{word}\s{word}\s{word}\s{word}\s{adj_noun}$'

word_dict={1:an1,2:an2,3:an3,4:an4,5:an5}

### ADJ NOUNs

an_comp='.+_ADJ\s.+_(?:NOUN|PROPN)'
adj='.+_ADJ'


ap2=f'^{an_comp}\s{word}\s{word}\s{word}$'
ap3=f'^{word}\s{an_comp}\s{word}\s{word}$'
ap4=f'^{word}\s{word}\s{an_comp}\s{word}$'
ap5=f'^{word}\s{word}\s{word}\s{an_comp}$'

phrase_dict={2:p2,3:p3,4:p4,5:p5,7:ap2,8:ap3,9:ap4,10:ap5}

    
def left_side_parser(df,phrases=False): # N N _ _ _ 
    cur_df=df.copy()
    if not phrases:

        try:
            cur_df[['modifier','head','w1','w2','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            compound_df=pd.DataFrame()
            modifier_df=pd.DataFrame()
            head_df=pd.DataFrame()
            return compound_df,modifier_df,head_df

        compound_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')

        modifier_df=pd.melt(cur_df,id_vars=modifier_id_vars,value_vars=['head','w1','w2'],value_name='context')

        head_df=pd.melt(cur_df,id_vars=head_id_vars,value_vars=['modifier','w1','w2','w3'],value_name='context')

        return compound_df,modifier_df,head_df
    else:
        try:
            cur_df[['modifier','head','w1','w2','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            phrase_df=pd.DataFrame()
            return phrase_df
        
        phrase_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')
        
        return phrase_df
        

def mid1_parser(df,phrases=False): # _ N N _ _
    cur_df=df.copy()
    
    if not phrases:
        try:
            cur_df[['w1','modifier','head','w2','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            compound_df=pd.DataFrame()
            modifier_df=pd.DataFrame()
            head_df=pd.DataFrame()
            return compound_df,modifier_df,head_df

        compound_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')

        modifier_df=pd.melt(cur_df,id_vars=modifier_id_vars,value_vars=['head','w1','w2','w3'],value_name='context')

        head_df=pd.melt(cur_df,id_vars=head_id_vars,value_vars=['modifier','w1','w2','w3'],value_name='context')

        return compound_df,modifier_df,head_df
    else:
        try:
            cur_df[['w1','modifier','head','w2','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            phrase_df=pd.DataFrame()
            return phrase_df
        
        phrase_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')
        
        return phrase_df

def mid2_parser(df,phrases=False): # _ _ N N _
    cur_df=df.copy()
    
    if not phrases:
        try:
            cur_df[['w1','w2','modifier','head','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            compound_df=pd.DataFrame()
            modifier_df=pd.DataFrame()
            head_df=pd.DataFrame()
            return compound_df,modifier_df,head_df

        compound_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')

        modifier_df=pd.melt(cur_df,id_vars=modifier_id_vars,value_vars=['head','w1','w2','w3'],value_name='context')

        head_df=pd.melt(cur_df,id_vars=head_id_vars,value_vars=['modifier','w1','w2','w3'],value_name='context')

        return compound_df,modifier_df,head_df
    else:
        try:
            cur_df[['w1','w2','modifier','head','w3']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            phrase_df=pd.DataFrame()
            return phrase_df
        
        phrase_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')
        
        return phrase_df

def right_side_parser(df,phrases=False): # _ _ _ N N
    cur_df=df.copy()
    
    if not phrases:
        try:
            cur_df[['w1','w2','w3','modifier','head']]=cur_df.lemma_pos.str.split(' ',expand=True,n=4)
        except ValueError:
            compound_df=pd.DataFrame()
            modifier_df=pd.DataFrame()
            head_df=pd.DataFrame()
            return compound_df,modifier_df,head_df

        compound_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')

        modifier_df=pd.melt(cur_df,id_vars=modifier_id_vars,value_vars=['head','w1','w2','w3'],value_name='context')

        head_df=pd.melt(cur_df,id_vars=head_id_vars,value_vars=['modifier','w2','w3'],value_name='context')

        return compound_df,modifier_df,head_df

    else:
        try:
            cur_df[['w1','w2','w3','modifier','head']]=cur_df.lemma_pos.str.split(' ',expand=True)
        except ValueError:
            phrase_df=pd.DataFrame()
            return phrase_df
        
        phrase_df=pd.melt(cur_df,id_vars=compound_id_vars,value_vars=['w1','w2','w3'],value_name='context')
        
        return phrase_df



def syntactic_reducer(df,phrases=False):
    pattern=df.iloc[0].comp_class
    if pattern in [1,6,11,12]: # N N _ N N | A N _ A N | A N _  N N | N N _ A N
        compound_left_df,modifier_left_df,head_left_df=left_side_parser(df)
        compound_right_df,modifier_right_df,head_right_df=right_side_parser(df)
        
        final_compound_df=pd.concat([compound_left_df,compound_right_df],ignore_index=True)
        final_modifier_df=pd.concat([modifier_left_df,modifier_right_df],ignore_index=True)
        final_head_df=pd.concat([head_left_df,head_right_df],ignore_index=True)
           
    elif pattern in [2,7]: # N N _ _ _ | A N _ _ _
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=left_side_parser(df)
        else:
            final_phrases_df=left_side_parser(df,phrases=True)

    elif pattern in [3,8]: # _ N N _ _ | _ A N _ _
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=mid1_parser(df)
        else:
            final_phrases_df=mid1_parser(df,phrases=True)
    
    elif pattern in [4,9]: # _ _ N N _ | _ _ A N _
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=mid2_parser(df)
        else:
            final_phrases_df=mid2_parser(df,phrases=True)
        
    elif pattern in [5,10]: # _ _ _ N N | _ _ _ A N
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=right_side_parser(df)   
        else:
            final_phrases_df=right_side_parser(df,phrases=True)

            

    if not phrases:
        return final_compound_df,final_modifier_df,final_head_df
    else:
        return final_phrases_df



def compound_extracter(df,phrases=False):
    
    comp_df_list=[]
    head_df_list=[]
    mod_df_list=[]
    phrase_df_list=[]
    if not phrases:
        
        for i in range(1,13):
            
            if df.loc[df.comp_class==i].shape[0]!=0:
                cur_comp_df,cur_mod_df,cur_head_df=syntactic_reducer(df.loc[df.comp_class==i])

                comp_df_list.append(cur_comp_df)
                mod_df_list.append(cur_mod_df)
                head_df_list.append(cur_head_df)
    
        compounds=pd.concat(comp_df_list,ignore_index=True,sort=False)
        modifiers=pd.concat(mod_df_list,ignore_index=True,sort=False)
        heads=pd.concat(head_df_list,ignore_index=True,sort=False)

        
        compounds.dropna(inplace=True)
        compounds=compounds.loc[compounds.context.isin(context_list)]
        compounds=compounds.groupby(compound_id_context_vars,observed=True)['count'].sum().to_frame()
        compounds.reset_index(inplace=True)

        modifiers.dropna(inplace=True)
        modifiers=modifiers.loc[modifiers.context.isin(context_list)]
        modifiers=modifiers.groupby(modifier_id_context_vars,observed=True)['count'].sum().to_frame()
        modifiers.reset_index(inplace=True)

        heads.dropna(inplace=True)
        heads=heads.loc[heads.context.isin(context_list)]
        heads=heads.groupby(head_id_context_vars,observed=True)['count'].sum().to_frame()
        heads.reset_index(inplace=True)

        return compounds,modifiers,heads
    
    else:
        phrase_list=[2,3,4,5,7,8,9,10]
        for i in phrase_list:
            cur_df=df.loc[df.lemma_pos.str.contains(phrase_dict[i])].copy()
            if cur_df.shape[0]!=0:
                cur_df.comp_class=i
                cur_phrase_df=syntactic_reducer(cur_df,phrases=True)
                phrase_df_list.append(cur_phrase_df)
                
                
        if phrase_df_list==[]:
            return None
        else:
            phrases=pd.concat(phrase_df_list,ignore_index=True,sort=False)

            phrases.dropna(inplace=True)
            phrases=phrases.loc[phrases.context.isin(context_list)]
            phrases=phrases.groupby(compound_id_context_vars,observed=True)['count'].sum().to_frame()
            phrases.reset_index(inplace=True)

            return phrases
    
    
def word_reducer(df):
    pattern=df.iloc[0].comp_class
    if pattern==1: # A|N _ _ _ _
        df[['word','w1','w2','w3','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3'],value_name='context')
        
    elif pattern==2: # _ A|N _ _ _
        df[['w1','word','w2','w3','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3','w4'],value_name='context')

    elif pattern==3: # _ _ A|N _ _
        df[['w1','w2','word','w3','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3','w4'],value_name='context')

    elif pattern==4: # _ _ _ A|N _
        df[['w1','w2','w3','word','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3','w4'],value_name='context')

    elif pattern==5: # _ _ _ _ A|N
        df[['w1','w2','w3','w4','word']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w2','w3','w4'],value_name='context')

    return word_df


def word_extractor(df):
    word_df_list=[]

    for i in range(1,6):
        cur_df=df.loc[df.lemma_pos.str.contains(word_dict[i])].copy()
        if cur_df.shape[0]!=0:
            cur_df.comp_class=i
            cur_word_df=word_reducer(cur_df)
            word_df_list.append(cur_word_df)

    words=pd.concat(word_df_list,ignore_index=True,sort=False)
        
    words.dropna(inplace=True)
    words=words.loc[words.context.isin(context_list)]
    words=words.loc[words.word.isin(constituent_list)]
    words=words.groupby(word_id_context_vars,observed=True)['count'].sum().to_frame()
    words.reset_index(inplace=True)
    
    return words
        

def parallelize_dataframe(df,phrases=False):
    num_partitions=round(0.95*mp.cpu_count())
    df_split = np.array_split(df, num_partitions)
    print("Done splitting the datasets")
    pool = Pool(num_partitions)

    cur_time=time.time()
    print("Starting parallelizing")
    if not args.word:
        
        if not phrases:
            #Processing heads, modifiers and compounds for Compound Aware

            results=pool.map_async(compound_extracter,df_split)
            pool.close()
            pool.join()

            results=results.get()

            print("Done parallelizing")
            compound_list = [ result[0] for result in results]
            compounds=pd.concat(compound_list,ignore_index=True)
            compounds=compounds.groupby(compound_id_context_vars,observed=True)['count'].sum().to_frame()
            compounds.reset_index(inplace=True)

            modifier_list = [ result[1] for result in results]
            modifiers=pd.concat(modifier_list,ignore_index=True)
            modifiers=modifiers.groupby(modifier_id_context_vars,observed=True)['count'].sum().to_frame()
            modifiers.reset_index(inplace=True)

            head_list = [ result[2] for result in results]
            heads=pd.concat(head_list,ignore_index=True)
            heads=heads.groupby(head_id_context_vars,observed=True)['count'].sum().to_frame()
            heads.reset_index(inplace=True)
            print("Total time taken",round(time.time()-cur_time),"secs")

            return compounds,modifiers,heads
        else:
            
            #Processing phrases for Compound Agnostic
            results=pool.starmap_async(compound_extracter,zip(df_split,repeat(phrases)))
            pool.close()
            pool.join()

            phrase_list=results.get()
            
            print("Done parallelizing")
            
            phrases=pd.concat(phrase_list,ignore_index=True)
            phrases=phrases.groupby(compound_id_context_vars,observed=True)['count'].sum().to_frame()
            phrases.reset_index(inplace=True)

            print("Total time taken",round(time.time()-cur_time),"secs")

            return phrases
        
    else:
        
        #Processing words for Compound Agnostic

        words_list=[]
        results=pool.map_async(word_extractor,df_split)

        pool.close()
        pool.join()
        words_list=results.get()

        print("Done parallelizing")
        
        words = pd.concat(words_list,ignore_index=True)
        words=words.groupby(word_id_context_vars,observed=True)['count'].sum().to_frame()
        words.reset_index(inplace=True)
        print("Total time taken",round(time.time()-cur_time),"secs")
        
        return words
    

    
    
def parquet_processor(f):   
    cur_fname=f.split('.')[0].split('/')[-1]
    print(f'Current parquet file: {f}')
    cur_parq=fastparquet.ParquetFile(f)

    print(f'Number of partitions: {len(cur_parq.row_groups)}')
    compounds_list=[]
    modifiers_list=[]
    heads_list=[]
    phrases_list=[]
    words_list=[]

    
    for i,cur_df in enumerate(cur_parq.iter_row_groups()):
        print(f'Partition {i+1} out of {len(cur_parq.row_groups)}\n')
        cur_df['nwords']=cur_df.lemma_pos.str.count(' ').add(1).copy()
        pd.options.mode.chained_assignment = 'warn'
        cur_df=cur_df.loc[cur_df.nwords==5]
        cur_df.drop(['nwords'],axis=1,inplace=True)
        
        if not args.word:
            reduced_df=cur_df.loc[cur_df.comp_class!=0].reset_index(drop=True)
            cur_compounds,cur_modifiers,cur_heads=parallelize_dataframe(reduced_df)
            compounds_list.append(cur_compounds)
            modifiers_list.append(cur_modifiers)
            heads_list.append(cur_heads)

            #Gathering phrases
            #Removing compound classes as they are already gathered
            #Use old comp_class to now store phrase_class
            print("Phrases")


            cur_phrases=parallelize_dataframe(cur_df,phrases=True)
            #print(cur_phrases.shape[0])
            phrases_list.append(cur_phrases)
            
        else:
            print("Words")
            cur_df.drop(['is_comp','comp_ner_sent'],axis=1,inplace=True)
            cur_words=parallelize_dataframe(cur_df)
            words_list.append(cur_words)

    if not args.word:

        compounds=pd.concat(compounds_list,ignore_index=True)
        comp_before=compounds.shape[0]
        compounds=compounds.groupby(compound_id_context_vars,observed=True)['count'].sum().to_frame()
        comp_after=compounds.shape[0]

        print(f"Compound before : {comp_before}, after : {comp_after} Change in percentage : {(comp_before-comp_after)/comp_before*100:0.2f}%")

        compounds.reset_index(inplace=True)
        compounds.to_pickle(f'{args.output}/compounds/{cur_fname}.pkl')

        modifiers=pd.concat(modifiers_list,ignore_index=True)
        mod_before=modifiers.shape[0]
        modifiers=modifiers.groupby(modifier_id_context_vars,observed=True)['count'].sum().to_frame()
        mod_after=modifiers.shape[0]

        print(f"Modifier before : {mod_before}, after : {mod_after} Change in percentage : {(mod_before-mod_after)/mod_before*100:0.2f}%")

        modifiers.reset_index(inplace=True)
        modifiers.to_pickle(f'{args.output}/modifiers/{cur_fname}.pkl')

        heads=pd.concat(heads_list,ignore_index=True)
        head_before=heads.shape[0]
        heads=heads.groupby(head_id_context_vars,observed=True)['count'].sum().to_frame()
        head_after=heads.shape[0]

        print(f"Head before : {head_before}, after : {head_after} Change in percentage : {(head_before-head_after)/head_before*100:0.2f}%")

        heads.reset_index(inplace=True)
        heads.to_pickle(f'{args.output}/heads/{cur_fname}.pkl')

        phrases=pd.concat(phrases_list,ignore_index=True)
        phr_before=phrases.shape[0]
        phrases=phrases.groupby(compound_id_context_vars,observed=True)['count'].sum().to_frame()
        phr_after=phrases.shape[0]

        print(f"Phrase before : {phr_before}, after : {phr_after} Change in percentage : {(phr_before-phr_after)/phr_before*100:0.2f}%")

        phrases.reset_index(inplace=True)
        phrases.to_pickle(f'{args.output}/phrases/{cur_fname}.pkl')
        
    else:
        
        words=pd.concat(words_list,ignore_index=True)
        words_before=words.shape[0]
        words=words.groupby(word_id_context_vars,observed=True)['count'].sum().to_frame()
        words_after=words.shape[0]

        print(f"Words before : {words_before}, after : {words_after} Change in percentage : {(words_before-words_after)/words_before*100:0.2f}%")

        words.reset_index(inplace=True)
        words.to_pickle(f'{args.output}/words/{cur_fname}.pkl')
        

    print("Done with file \n")
    
f=args.data+'/'+args.decade+'.parq'

parquet_processor(f)





