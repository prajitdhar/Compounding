import pandas as pd
import numpy as np
import argparse
import time
import pickle as pkl

from itertools import product
from functools import reduce
import glob
import os

import seaborn as sns
sns.set(style="whitegrid", font_scale = 2.5)
sns.set_context(rc={"lines.markersize": 17, "lines.linewidth": 2})

import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser(description='Compute features from sparse dataset')

parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--reddy90', type=str,
                    help='Path to Reddy data')
parser.add_argument('--cordeiro90', type=str,
                    help='Path to Cordeiro 90 data')
parser.add_argument('--cordeiro100', type=str,
                    help='Path to Cordeiro 100 data')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')
parser.add_argument('--tag', action='store_true',
                    help='Should the POS tag be kept?')
parser.add_argument('--ppmi', action='store_true',
                    help='Should co-occurence matrix be converted to PPMI values')
parser.add_argument('--plot', action='store_true',
                    help='Should plots be saved')

args = parser.parse_args()


reddy_df=pd.read_csv(args.reddy90,sep='\t')
reddy_df['source']='reddy'
cordeiro90_df=pd.read_csv(args.cordeiro90,sep='\t')
cordeiro90_df['source']='cordeiro90'
cordeiro100_df=pd.read_csv(args.cordeiro100,sep='\t')
cordeiro100_df['source']='cordeiro100'


def testset_tagger(df):

    #### NOUN NOUN
    
    copy_df_1=df.copy()
    copy_df_1.modifier=copy_df_1.modifier+'_NOUN'
    copy_df_1['head']=copy_df_1['head']+'_NOUN'


    
    ### PROPN PROPN    

    copy_df_4=df.copy()
    copy_df_4.modifier=copy_df_4.modifier+'_PROPN'
    copy_df_4['head']=copy_df_4['head']+'_PROPN'
    
   
    ### ADJ/NOUN NOUN
    
    copy_df_5=df.copy()
    
    copy_df_5.loc[copy_df_5.is_adj==True,"modifier"]+="_ADJ"
    copy_df_5.loc[copy_df_5.is_adj==False,"modifier"]+="_NOUN"
    copy_df_5['head']=copy_df_5['head']+'_NOUN'   
    
    

    
    #### ADJ/PROPN PROPN
    
    copy_df_8=df.copy()
    copy_df_8.loc[copy_df_8.is_adj==True,"modifier"]+="_ADJ"
    copy_df_8.loc[copy_df_8.is_adj==False,"modifier"]+="_PROPN"
    copy_df_8['head']=copy_df_8['head']+'_PROPN' 
    
    
    complete_df=pd.concat([copy_df_1,copy_df_4,copy_df_5,copy_df_8],ignore_index=True)
                           
    return complete_df
    
comp_ratings_df=pd.concat([reddy_df,cordeiro90_df,cordeiro100_df])
#comp_ratings_df.drop_duplicates(inplace=True)
if args.tag:
    comp_ratings_df=testset_tagger(comp_ratings_df)


def process_time_compound(df):

    print(f'Temporal information is stored with intervals {temporal}')

    df['time']=df['year'] - df['year']%temporal
    
    df=df.groupby(['modifier','head','time','context'])['count'].sum().to_frame()
    df.reset_index(inplace=True)
    return df
        
def process_cutoff_compound(df):

    df=df.loc[df.groupby(['modifier','head','time'])['count'].transform('sum').gt(cutoff)]
    return df


def process_cutoff_constituent(df,ctype='word'):

    df=df.loc[df.groupby([ctype,'time'])['count'].transform('sum').gt(cutoff)]    
    return df


def process_constituent(df,ctype='word'):
            
    df['time']=df['year'] - df['year']%temporal
          
    df=df.groupby([ctype,'time','context'])['count'].sum().to_frame()

    df.reset_index(inplace=True)
    
    return df


def ppmi(ppmi_df):
    
    ppmi_cols=ppmi_df.columns.tolist()
    ppmi_cols=['XY' if 'count' in x else x for x in ppmi_cols]
    ppmi_df.columns=ppmi_cols

    ppmi_time_counts=ppmi_df.groupby('time')['XY'].sum().to_frame()
    ppmi_time_counts.columns=['N']


    Y_star=ppmi_df.groupby(['context','time'])['XY'].sum().to_frame()
    Y_star.columns=['Y']

    ppmi_df=pd.merge(ppmi_df,Y_star.reset_index(),on=['context','time'])
    
    X_cols=[x for x in ppmi_cols if x not in ['context','XY'] ]


    X_star=ppmi_df.groupby(X_cols)['XY'].sum().to_frame()
    X_star.columns=['X']

    ppmi_df=pd.merge(ppmi_df,X_star.reset_index(),on=X_cols)
    ppmi_df=pd.merge(ppmi_df,ppmi_time_counts.reset_index(),on=['time'])
    ppmi_df['count']=np.log2((ppmi_df['XY']*ppmi_df['N'])/(ppmi_df['X']*ppmi_df['Y']))
    ppmi_df=ppmi_df.loc[ppmi_df['count']>=0]
    ppmi_df.drop(['XY','X','Y','N'],axis=1,inplace=True)
    
    return ppmi_df



def temporal_features(compounds,modifiers,heads,compound_list_df):
    
    compounds_pivot=pd.pivot_table(compounds, values='count', index=['modifier','head', 'time'],
                       columns=['context'], aggfunc="sum",fill_value=0)
    modifiers_pivot=pd.pivot_table(modifiers, values='count', index=['modifier','time'],
                       columns=['context'], aggfunc="sum",fill_value=0)
    heads_pivot=pd.pivot_table(heads, values='count', index=['head','time'],
                       columns=['context'], aggfunc="sum",fill_value=0)
    
    change_compounds_df=compounds_pivot.groupby(level=[0,1]).apply(cosine_bw_rows)
    change_compounds_df.columns=['change_comp']

    change_modifiers_df=modifiers_pivot.groupby(level=[0]).apply(cosine_bw_rows)
    change_modifiers_df.columns=['change_mod']
    change_heads_df=heads_pivot.groupby(level=[0]).apply(cosine_bw_rows)
    change_heads_df.columns=['change_head']
    
    
    changed_df=pd.merge(change_compounds_df.reset_index(),compound_list_df,on=['modifier','head','time'],how='right')
    changed_df=pd.merge(changed_df,change_modifiers_df.reset_index(),on=['modifier','time'],how='right')
    
    changed_df=pd.merge(changed_df,change_heads_df.reset_index(),on=['head','time'])
    return changed_df





def merge_comp_ratings(features_df):

    features_df=pd.pivot_table(features_df, index=['modifier','head'], columns=['time'])
    features_df_columns_1=features_df.columns.get_level_values(0)
    features_df_columns_2=features_df.columns.get_level_values(1)

    cur_year=0
    new_columns=[]
    for year in features_df_columns_2:
        new_columns.append(features_df_columns_1[cur_year]+":"+str(year))
        cur_year+=1

    features_df.columns=new_columns
    cur_ratings_df_na=features_df.reset_index().merge(comp_ratings_df,on=['modifier','head'])


    imputer= SimpleImputer(strategy="median")
    features_df = features_df.reset_index().merge(comp_ratings_df[['modifier', 'head']], on=['modifier','head'], how='right')
    features_df.set_index(['modifier', 'head'], inplace=True)
    df_med=pd.DataFrame(imputer.fit_transform(features_df))
    df_med.columns=features_df.columns
    df_med.index=features_df.index

    cur_ratings_df_med=df_med.reset_index().merge(comp_ratings_df,on=['modifier','head'])
    
    return cur_ratings_df_na,cur_ratings_df_med


cutoff_list=[0,10,50,100,500,1000]
temporal_list=[10,20,50,100]


if args.ppmi:
    ppmi_str="PPMI"
else:
    ppmi_str="RAW"
    
if args.tag:
    tag_str='Tagged'
else:
    tag_str='UnTagged'

    
unique_mod_list=comp_ratings_df[['modifier']].drop_duplicates()['modifier'].to_list()
unique_head_list=comp_ratings_df[['head']].drop_duplicates()['head'].to_list() 
unique_constituent_list=list(set(unique_mod_list+unique_head_list))
    
complete_phrases=pd.read_pickle(args.inputdir+"/phrases.pkl")
complete_phrases=complete_phrases.loc[(complete_phrases.modifier.isin(unique_mod_list))&(complete_phrases['head'].isin(unique_head_list))]            


complete_words=pd.read_pickle(args.inputdir+"/words.pkl")
complete_words=complete_words.loc[complete_words.word.isin(unique_constituent_list)]

    
    
complete_compounds=pd.read_pickle(args.inputdir+"/compounds.pkl")
complete_compounds=complete_compounds.loc[(complete_compounds.modifier.isin(unique_mod_list))&(complete_compounds['head'].isin(unique_head_list))]            


complete_modifiers=pd.read_pickle(args.inputdir+"/modifiers.pkl")
complete_modifiers=complete_modifiers.loc[complete_modifiers.modifier.isin(unique_mod_list)]


complete_heads=pd.read_pickle(args.inputdir+"/heads.pkl")
complete_heads=complete_heads.loc[complete_heads['head'].isin(unique_head_list)]

    
if not args.tag:
    print('Removing tags')
    complete_phrases['head']=complete_phrases['head'].str.replace('_NOUN|_PROPN','',regex=True)
    complete_phrases.modifier=complete_phrases.modifier.str.replace('_NOUN|_PROPN|_ADJ','',regex=True)

    complete_words.word=complete_words.word.str.replace('_NOUN|_PROPN|_ADJ','',regex=True)
        
        
    complete_compounds['head']=complete_compounds['head'].str.replace('_NOUN|_PROPN','',regex=True)
    complete_compounds.modifier=complete_compounds.modifier.str.replace('_NOUN|_PROPN|_ADJ','',regex=True)

    complete_modifiers.modifier=complete_modifiers.modifier.str.replace('_NOUN|_PROPN|_ADJ','',regex=True)

    complete_heads['head']=complete_heads['head'].str.replace('_NOUN|_PROPN','',regex=True)
    
    
    
for temporal in temporal_list:
    
    print(f'Time span:  {temporal}')
        
    temporal_compounds=process_time_compound(complete_compounds)
        
        
    temporal_phrases=process_time_compound(complete_phrases)

    constituents=process_constituent(complete_words,'word')
    print('Done reading words')
        
    modifiers_aware=process_constituent(complete_modifiers,'modifier')
    print('Done reading modifiers')

    heads_aware=process_constituent(complete_heads,'head')
    print('Done reading heads')
        
    for cutoff in cutoff_list:
        
        print(f'Cutoff: {cutoff}')
        print(f'Time span:  {temporal}')
        temp_cutoff_str=str(temporal)+'_'+str(cutoff)
            
        if cutoff==0:
            compounds_aware=temporal_compounds.copy()
            compounds_agnostic=temporal_phrases.copy()
                
        else:
            compounds_aware=process_cutoff_compound(temporal_compounds)
            compounds_agnostic=process_cutoff_compound(temporal_phrases)
        
            constituents=process_cutoff_constituent(constituents,ctype='word')
            modifiers_aware=process_cutoff_constituent(modifiers_aware,ctype='modifier')
            heads_aware=process_cutoff_constituent(heads_aware,ctype='head')

        print('Done reading compounds')
        
        if args.ppmi:
            compounds_aware=ppmi(compounds_aware)
            heads_aware=ppmi(heads_aware)
            modifiers_aware=ppmi(modifiers_aware)
                        
            compounds_agnostic=ppmi(compounds_agnostic)
            constituents=ppmi(constituents)
            
        heads_agnostic=constituents.copy()
        heads_agnostic_cols=heads_agnostic.columns
        heads_agnostic_cols=['head' if 'word' in x else x for x in heads_agnostic_cols]
        heads_agnostic.columns=heads_agnostic_cols

        modifiers_agnostic=constituents.copy()
        modifiers_agnostic_cols=modifiers_agnostic.columns
        modifiers_agnostic_cols=['modifier' if 'word' in x else x for x in modifiers_agnostic_cols]
        modifiers_agnostic.columns=modifiers_agnostic_cols
            
            
        timespan_list_aware_df=pd.DataFrame(compounds_aware.time.unique())
        timespan_list_aware_df.columns=['time']

        compound_list_aware_df=comp_ratings_df[['modifier','head']].copy()
        compound_list_aware_df=compound_list_aware_df.merge(timespan_list_aware_df,how='cross')

        modifier_list_aware_df=comp_ratings_df[['modifier']].drop_duplicates().copy()
        modifier_list_aware_df=modifier_list_aware_df.merge(timespan_list_aware_df,how='cross')

        head_list_aware_df=comp_ratings_df[['head']].drop_duplicates().copy()
        head_list_aware_df=head_list_aware_df.merge(timespan_list_aware_df,how='cross')

        all_comps_aware=compounds_aware[['modifier','head','time']].copy()
        all_comps_aware.drop_duplicates(inplace=True)

        all_mods_aware=compounds_aware[['modifier','time']].copy()
        all_mods_aware.drop_duplicates(inplace=True)

        all_heads_aware=compounds_aware[['head','time']].copy()
        all_heads_aware.drop_duplicates(inplace=True)

        not_found_compounds_aware_df=compound_list_aware_df.merge(all_comps_aware, on=['modifier','head','time'], how='outer', suffixes=['', '_'], indicator=True)
        not_found_compounds_aware_df=not_found_compounds_aware_df.loc[not_found_compounds_aware_df['_merge']=='left_only']
        not_found_compounds_aware_df.drop('_merge',axis=1,inplace=True)


        not_found_modifiers_aware_df=modifier_list_aware_df.merge(all_mods_aware, on=['modifier','time'], how='outer', suffixes=['', '_'], indicator=True)
        not_found_modifiers_aware_df=not_found_modifiers_aware_df.loc[not_found_modifiers_aware_df['_merge']=='left_only']
        not_found_modifiers_aware_df.drop('_merge',axis=1,inplace=True)

        not_found_heads_aware_df=head_list_aware_df.merge(all_heads_aware, on=['head','time'], how='outer', suffixes=['', '_'], indicator=True)
        not_found_heads_aware_df=not_found_heads_aware_df.loc[not_found_heads_aware_df['_merge']=='left_only']
        not_found_heads_aware_df.drop('_merge',axis=1,inplace=True)



        timespan_list_agnostic_df=pd.DataFrame(compounds_agnostic.time.unique())
        timespan_list_agnostic_df.columns=['time']

        compound_list_agnostic_df=comp_ratings_df[['modifier','head']].copy()
        compound_list_agnostic_df=compound_list_agnostic_df.merge(timespan_list_agnostic_df,how='cross')

        modifier_list_agnostic_df=comp_ratings_df[['modifier']].drop_duplicates().copy()
        modifier_list_agnostic_df=modifier_list_agnostic_df.merge(timespan_list_agnostic_df,how='cross')

        head_list_agnostic_df=comp_ratings_df[['head']].drop_duplicates().copy()
        head_list_agnostic_df=head_list_agnostic_df.merge(timespan_list_agnostic_df,how='cross')

        all_comps_agnostic=compounds_agnostic[['modifier','head','time']].copy()
        all_comps_agnostic.drop_duplicates(inplace=True)

        all_mods_agnostic=compounds_agnostic[['modifier','time']].copy()
        all_mods_agnostic.drop_duplicates(inplace=True)

        all_heads_agnostic=compounds_agnostic[['head','time']].copy()
        all_heads_agnostic.drop_duplicates(inplace=True)

        not_found_compounds_agnostic_df=compound_list_agnostic_df.merge(all_comps_agnostic, on=['modifier','head','time'], how='outer', suffixes=['', '_'], indicator=True)
        not_found_compounds_agnostic_df=not_found_compounds_agnostic_df.loc[not_found_compounds_agnostic_df['_merge']=='left_only']
        not_found_compounds_agnostic_df.drop('_merge',axis=1,inplace=True)

        not_found_modifiers_agnostic_df=modifier_list_agnostic_df.merge(all_mods_agnostic, on=['modifier','time'], how='outer', suffixes=['', '_'], indicator=True)
        not_found_modifiers_agnostic_df=not_found_modifiers_agnostic_df.loc[not_found_modifiers_agnostic_df['_merge']=='left_only']
        not_found_modifiers_agnostic_df.drop('_merge',axis=1,inplace=True)

        not_found_heads_agnostic_df=head_list_agnostic_df.merge(all_heads_agnostic, on=['head','time'], how='outer', suffixes=['', '_'], indicator=True)
        not_found_heads_agnostic_df=not_found_heads_agnostic_df.loc[not_found_heads_agnostic_df['_merge']=='left_only']
        not_found_heads_agnostic_df.drop('_merge',axis=1,inplace=True)

        print('Calculating features')
        unique_mod_list=comp_ratings_df[['modifier']].drop_duplicates()['modifier'].to_list()
        unique_head_list=comp_ratings_df[['head']].drop_duplicates()['head'].to_list() 
        print('CompoundAware features')

        change_aware_df=temporal_features(compounds_aware,modifiers_aware,heads_aware,all_comps_aware)


        print('CompoundAgnostic features')

        change_agnostic_df=temporal_features(compounds_agnostic,modifiers_agnostic,heads_agnostic,all_comps_agnostic)


        cur_ratings_aware_df_na,cur_ratings_aware_df_med=merge_comp_ratings(change_aware_df)
        cur_ratings_agnostic_df_na,cur_ratings_agnostic_df_med=merge_comp_ratings(change_agnostic_df)


        print(cur_ratings_aware_df_na.shape[0])
        print(cur_ratings_aware_df_na..drop_duplicates().shape[0])

        print('Saving feature datasets')

        cur_ratings_aware_df_na.drop_duplicates().to_csv(f'{args.outputdir}/temporal_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
        cur_ratings_aware_df_med.drop_duplicates().to_csv(f'{args.outputdir}/temporal_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)

        cur_ratings_agnostic_df_na.drop_duplicates().to_csv(f'{args.outputdir}/temporal_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
        cur_ratings_agnostic_df_med.drop_duplicates()to_csv(f'{args.outputdir}/temporal_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)
