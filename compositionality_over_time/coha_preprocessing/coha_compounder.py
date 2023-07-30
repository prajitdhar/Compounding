import pandas as pd
import os
import numpy as np
import time
import argparse
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat


parser = argparse.ArgumentParser(description='Program to run compounder for the coha files')

parser.add_argument('--input', type=str,
                    help='location of the pickle data files')

parser.add_argument('--word', action='store_true',
                    help='Extracting context for words only?')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')


args = parser.parse_args()






compound_id_vars=['modifier','head','year','count','num_comp','comp_ner_sent']
modifier_id_vars=['modifier','year','count','num_comp','comp_ner_sent']
head_id_vars=['head','year','count','num_comp','comp_ner_sent']
word_id_vars=['word','year','count','num_comp','comp_ner_sent']

word='.+_.+'
comp='.+_(?:NOUN|PROPN)\s.+_(?:NOUN|PROPN)'


p2=f'^{comp}\s{word}\s{word}\s{word}$'
p3=f'^{word}\s{comp}\s{word}\s{word}$'
p4=f'^{word}\s{word}\s{comp}\s{word}$'
p5=f'^{word}\s{word}\s{word}\s{comp}$'

phrase_dict={2:p2,3:p3,4:p4,5:p5}

noun='.+_(?:NOUN|PROPN)'

n1=f'^{noun}\s{word}\s{word}\s{word}\s{word}$'
n2=f'^{word}\s{noun}\s{word}\s{word}\s{word}$'
n3=f'^{word}\s{word}\s{noun}\s{word}\s{word}$'
n4=f'^{word}\s{word}\s{word}\s{noun}\s{word}$'
n5=f'^{word}\s{word}\s{word}\s{word}\s{noun}$'

word_dict={1:n1,2:n2,3:n3,4:n4,5:n5}


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
            cur_df[['w1','w2','w3','modifier','head']]=cur_df.lemma_pos.str.split(' ',expand=True)
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
    if pattern==1: # N N _ N N
        compound_left_df,modifier_left_df,head_left_df=left_side_parser(df)
        compound_right_df,modifier_right_df,head_right_df=right_side_parser(df)
        
        final_compound_df=pd.concat([compound_left_df,compound_right_df],ignore_index=True)
        final_modifier_df=pd.concat([modifier_left_df,modifier_right_df],ignore_index=True)
        final_head_df=pd.concat([head_left_df,head_right_df],ignore_index=True)
           
    elif pattern==2: # N N _ _ _
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=left_side_parser(df)
        else:
            final_phrases_df=left_side_parser(df,phrases=True)

    elif pattern==3: # _ N N _ _
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=mid1_parser(df)
        else:
            final_phrases_df=mid1_parser(df,phrases=True)
    
    elif pattern==4: # _ _ N N _
        if not phrases:
            final_compound_df,final_modifier_df,final_head_df=mid2_parser(df)
        else:
            final_phrases_df=mid2_parser(df,phrases=True)
        
    elif pattern==5: # _ _ _ N N
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
        
        for i in range(1,6):
            
            if df.loc[df.comp_class==i].shape[0]!=0:
                cur_comp_df,cur_mod_df,cur_head_df=syntactic_reducer(df.loc[df.comp_class==i])

                comp_df_list.append(cur_comp_df)
                mod_df_list.append(cur_mod_df)
                head_df_list.append(cur_head_df)
    
        compounds=pd.concat(comp_df_list,ignore_index=True,sort=False)
        modifiers=pd.concat(mod_df_list,ignore_index=True,sort=False)
        heads=pd.concat(head_df_list,ignore_index=True,sort=False)

        
        compounds.dropna(inplace=True)
        compounds=compounds.groupby(['modifier','head','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)

        modifiers.dropna(inplace=True)
        modifiers=modifiers.groupby(['modifier','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
        modifiers.reset_index(inplace=True)

        heads.dropna(inplace=True)
        heads=heads.groupby(['head','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
        heads.reset_index(inplace=True)

        return compounds,modifiers,heads
    
    else:
        for i in range(2,6):
            cur_df=df.loc[df.lemma_pos.str.contains(phrase_dict[i])].copy()
            if cur_df.shape[0]!=0:
                cur_df.comp_class=i
                cur_phrase_df=syntactic_reducer(cur_df,phrases=True)
                phrase_df_list.append(cur_phrase_df)

        phrases=pd.concat(phrase_df_list,ignore_index=True,sort=False)
        
        phrases.dropna(inplace=True)
        phrases=phrases.groupby(['modifier','head','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
        phrases.reset_index(inplace=True)
        
        return phrases
        

def word_reducer(df):
    pattern=df.iloc[0].comp_class
    if pattern==1: # N _ _ _ _
        df[['word','w1','w2','w3','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3'],value_name='context')
        
    elif pattern==2: # _ N _ _ _
        df[['w1','word','w2','w3','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3','w4'],value_name='context')

    elif pattern==3: # _ _ N _ _
        df[['w1','w2','word','w3','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3','w4'],value_name='context')

    elif pattern==4: # _ _ _ N _
        df[['w1','w2','w3','word','w4']]=df.lemma_pos.str.split(' ',expand=True)
        word_df=pd.melt(df,id_vars=word_id_vars,value_vars=['w1','w2','w3','w4'],value_name='context')

    elif pattern==5: # _ _ _ _ N
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
    words=words.groupby(['word','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
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
            compounds=compounds.groupby(['modifier','head','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
            compounds.reset_index(inplace=True)

            modifier_list = [ result[1] for result in results]
            modifiers=pd.concat(modifier_list,ignore_index=True)
            modifiers=modifiers.groupby(['modifier','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
            modifiers.reset_index(inplace=True)

            head_list = [ result[2] for result in results]
            heads=pd.concat(head_list,ignore_index=True)
            heads=heads.groupby(['head','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
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
            phrases=phrases.groupby(['modifier','head','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
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
        words=words.groupby(['word','num_comp','context','year','comp_ner_sent'])['count'].sum().to_frame()
        words.reset_index(inplace=True)
        print("Total time taken",round(time.time()-cur_time),"secs")
        
        return words
    

def dec_df_processor(f):
    t1=time.time()
    cur_decade=f.split('.')[0]
    print(f'Current decade: {cur_decade}')

    cur_dec_df=pd.read_pickle(args.input+f)
    if not args.word:
        reduced_df=cur_dec_df.loc[cur_dec_df.comp_class!=0].reset_index(drop=True)
        compound_df,modifier_df,head_df=parallelize_dataframe(reduced_df)


        #Gathering phrases
        #Removing compound classes as they are already gathered
        #Use old comp_class to now store phrase_class
        print("Phrases")

        phrase_df=parallelize_dataframe(cur_dec_df,phrases=True)

        print(f"Total time for decade {cur_decade} taken {round(time.time()-t1)} secs")
        return compound_df,modifier_df,head_df,phrase_df
    else:
        print("Words")
        
        word_df=parallelize_dataframe(cur_dec_df)
        print(f"Total time for decade {cur_decade} taken {round(time.time()-t1)} secs")
        return word_df
    
    
_dir = args.input

files_orig = sorted(os.listdir(_dir))#[1:]
print(len(files_orig))

comp_df_list=[]
mod_df_list=[]
head_df_list=[]
phrase_df_list=[]

word_df_list=[]

if not args.word:
    for cur_file in files_orig: 
        comp_df,mod_df,head_df,phrase_df=dec_df_processor(cur_file)
        comp_df_list.append(comp_df)
        mod_df_list.append(mod_df)
        head_df_list.append(head_df)
        phrase_df_list.append(phrase_df)
    
    compounds=pd.concat(comp_df_list,ignore_index=True)
    print(f'Compounds shape {compounds.shape[0]}')
    compounds.to_pickle(f'{args.output}/compounds.pkl')

    modifiers=pd.concat(mod_df_list,ignore_index=True)
    print(f'Modifiers shape {modifiers.shape[0]}')
    modifiers.to_pickle(f'{args.output}/modifiers.pkl')

    heads=pd.concat(head_df_list,ignore_index=True)
    print(f'Heads shape {heads.shape[0]}')
    heads.to_pickle(f'{args.output}/heads.pkl')

    phrases=pd.concat(phrase_df_list,ignore_index=True)
    print(f'Phrase shape {phrases.shape[0]}')
    phrases.to_pickle(f'{args.output}/phrases.pkl')
    
else:
    for cur_file in files_orig:
        word_df=dec_df_processor(cur_file)
        word_df_list.append(word_df)
    
    words=pd.concat(word_df_list,ignore_index=True)
    print(f'Words shape {words.shape[0]}')
    words.to_pickle(f'{args.output}/words.pkl')
