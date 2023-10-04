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



parser = argparse.ArgumentParser(description='Compute temporal variation features from sparse dataset for google version')

parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')
parser.add_argument('--tag', action='store_true',
                    help='Should the POS tag be kept?')
parser.add_argument('--ppmi', action='store_true',
                    help='Should co-occurence matrix be converted to PPMI values')
parser.add_argument('--temporal',  type=int,
                    help='Value to bin the temporal information: 10000 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')


args = parser.parse_args()


reddy_df=pd.read_csv('/data/dharp/compounds/Compounding/data/reddy_90.txt',sep='\t')
reddy_df['source']='reddy'
cordeiro90_df=pd.read_csv('/data/dharp/compounds/Compounding/data/cordeiro_90.txt',sep='\t')
cordeiro90_df['source']='cordeiro90'
cordeiro100_df=pd.read_csv('/data/dharp/compounds/Compounding/data/cordeiro_100.txt',sep='\t')
cordeiro100_df['source']='cordeiro100'

    
comp_ratings_df=pd.concat([reddy_df,cordeiro90_df,cordeiro100_df])
#comp_ratings_df.drop_duplicates(inplace=True)


def testset_tagger(df):

    #### NOUN NOUN
    
    copy_df_1=df.copy()
    copy_df_1.modifier=copy_df_1.modifier+'_NOUN'
    copy_df_1['head']=copy_df_1['head']+'_NOUN'

    ### PROPN NOUN

    copy_df_2=df.copy()
    copy_df_2.modifier=copy_df_2.modifier+'_PROPN'
    copy_df_2['head']=copy_df_2['head']+'_NOUN'
    
    ### NOUN PROPN

    copy_df_3=df.copy()
    copy_df_3.modifier=copy_df_3.modifier+'_NOUN'
    copy_df_3['head']=copy_df_3['head']+'_PROPN'
    
    ### PROPN PROPN    

    copy_df_4=df.copy()
    copy_df_4.modifier=copy_df_4.modifier+'_PROPN'
    copy_df_4['head']=copy_df_4['head']+'_PROPN'
    
   
    ### ADJ/NOUN NOUN
    
    copy_df_5=df.copy()
    
    copy_df_5.loc[copy_df_5.is_adj==True,"modifier"]+="_ADJ"
    copy_df_5.loc[copy_df_5.is_adj==False,"modifier"]+="_NOUN"
    copy_df_5['head']=copy_df_5['head']+'_NOUN'   
    
    
    ### ADJ/NOUN PROPN
    
    copy_df_6=df.copy()
    copy_df_6.loc[copy_df_6.is_adj==True,"modifier"]+="_ADJ"
    copy_df_6.loc[copy_df_6.is_adj==False,"modifier"]+="_NOUN"
    copy_df_6['head']=copy_df_6['head']+'_PROPN'  

    
    #### ADJ/PROPN NOUN
    
    copy_df_7=df.copy()
    copy_df_7.loc[copy_df_7.is_adj==True,"modifier"]+="_ADJ"
    copy_df_7.loc[copy_df_7.is_adj==False,"modifier"]+="_PROPN"
    copy_df_7['head']=copy_df_7['head']+'_NOUN' 
    
    
    #### ADJ/PROPN PROPN
    
    copy_df_8=df.copy()
    copy_df_8.loc[copy_df_8.is_adj==True,"modifier"]+="_ADJ"
    copy_df_8.loc[copy_df_8.is_adj==False,"modifier"]+="_PROPN"
    copy_df_8['head']=copy_df_8['head']+'_PROPN' 
    
    
    complete_df=pd.concat([copy_df_1,copy_df_2,copy_df_3,copy_df_4,copy_df_5,copy_df_6,copy_df_7,copy_df_8],ignore_index=True)
                           
    return complete_df     

if args.tag:
    comp_ratings_df=testset_tagger(comp_ratings_df)
comp_ratings_df


def process_decades_compound(dec_list,input_dir,unique_mod_list,unique_head_list,ctype='compound'):

    if os.path.exists(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl"):
        print('Reading file')
        complete_df=pd.read_pickle(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl")
        
    elif os.path.exists(f"{input_dir}/{ctype}s/10_{dec_list[0]}_{tag_str}.pkl") and args.temporal!=10000:
        print(f'Reading decades file {ctype}s/10_{dec_list[0]}_{tag_str}.pkl')
        complete_df=pd.read_pickle(f"{input_dir}/{ctype}s/10_{dec_list[0]}_{tag_str}.pkl")
        
        print(f'Reducing to {args.temporal}')
        complete_df['time']=complete_df['time']-complete_df['time']%args.temporal

        complete_df=complete_df.groupby(['modifier','head','time','context'])['count'].sum().to_frame().reset_index()
        
        print("Saving file")
        complete_df.to_pickle(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl")


    else:

        df_list=[]

        for dec in dec_list:
            print(dec)
            cur_df=pd.read_pickle(f'{input_dir}/{ctype}s/{dec}.pkl')
            
            if not args.tag:
                cur_df=compound_tag_remover(cur_df)
            cur_df['time']=dec
            cur_df['time']=cur_df['time']-cur_df['time']%args.temporal
            df_list.append(cur_df)

        print('Done reading compound dataframes')
        complete_df=pd.concat(df_list,ignore_index=True)

        if args.temporal!=10:
            complete_df=complete_df.groupby(['modifier','head','time','context'])['count'].sum().to_frame().reset_index()
        
        print("Saving file")
        complete_df.to_pickle(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl")
    
    reduced_complete_df=complete_df.loc[(complete_df.modifier.isin(unique_mod_list))&(complete_df['head'].isin(unique_head_list))]            
    return reduced_complete_df


def process_decades_constituent(dec_list,input_dir,unique_constituent_list,ctype='word'):
        
    if os.path.exists(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl"):
        print('Reading file')
        complete_df=pd.read_pickle(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl")
        
    elif os.path.exists(f"{input_dir}/{ctype}s/10_{dec_list[0]}_{tag_str}.pkl") and args.temporal!=10000:
        print(f'Reading decades file {ctype}s/10_{dec_list[0]}_{tag_str}.pkl')
        complete_df=pd.read_pickle(f"{input_dir}/{ctype}s/10_{dec_list[0]}_{tag_str}.pkl")
        
        print(f'Reducing to {args.temporal}')
        complete_df['time']=complete_df['time']-complete_df['time']%args.temporal
        complete_df=complete_df.groupby([ctype,'time','context'])['count'].sum().to_frame().reset_index()
        
        print("Saving file")
        complete_df.to_pickle(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl")


    else:

        df_list=[]

        for dec in dec_list:
            cur_df=pd.read_pickle(f'{input_dir}/{ctype}s/{dec}.pkl')
            if not args.tag:
                cur_df=constituent_tag_remover(cur_df,ctype)
            cur_df['time']=dec
            cur_df['time']=cur_df['time']-cur_df['time']%args.temporal
            df_list.append(cur_df)

        print(f'Done reading {ctype} dataframes')
        complete_df=pd.concat(df_list,ignore_index=True)
        
        if args.temporal!=10:
            complete_df=complete_df.groupby([ctype,'time','context'])['count'].sum().to_frame().reset_index()
        
        print("Saving file")
        complete_df.to_pickle(f"{input_dir}/{ctype}s/{args.temporal}_{dec_list[0]}_{tag_str}.pkl")

    if ctype=='modifier':
        reduced_complete_df=complete_df.loc[complete_df.modifier.isin(unique_constituent_list)]
    elif ctype=='head':
        reduced_complete_df=complete_df.loc[complete_df['head'].isin(unique_constituent_list)]
    else:
        reduced_complete_df=complete_df.loc[complete_df.word.isin(unique_constituent_list)]

    return reduced_complete_df


def compound_tag_remover(compounds):
    
    print('Removing tags for compound dataset')
    compounds['head']=compounds['head'].str.replace('_NOUN|_PROPN','',regex=True)
    compounds.modifier=compounds.modifier.str.replace('_NOUN|_PROPN|_ADJ','',regex=True)
    
    compounds=compounds.groupby(['modifier','head','context'])['count'].sum().to_frame().reset_index()

    return compounds


def constituent_tag_remover(constituents,ctype='word'):
    
    print(f'Removing tags for {ctype} dataset')
    constituents[ctype]=constituents[ctype].str.replace('_NOUN|_PROPN|_ADJ','',regex=True)
    
    constituents=constituents.groupby([ctype,'context'])['count'].sum().to_frame().reset_index()

    return constituents


def process_cutoff_compound(df,cutoff):

    df=df.loc[df.groupby(['modifier','head','time'])['count'].transform('sum').gt(cutoff)]
    
    return df


def process_cutoff_constituent(df,cutoff,ctype='word'):

    df=df.loc[df.groupby([ctype,'time'])['count'].transform('sum').gt(cutoff)]
    
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
    cur_ratings_df_na=features_df.reset_index().merge(comp_ratings_df,on=['modifier','head'],how='right')


    imputer= SimpleImputer(strategy="median")
    df_med=pd.DataFrame(imputer.fit_transform(features_df))
    df_med.columns=features_df.columns
    df_med.index=features_df.index

    cur_ratings_df_med=df_med.reset_index().merge(comp_ratings_df,on=['modifier','head'],how='right')
    
    return cur_ratings_df_na,cur_ratings_df_med

def cosine_bw_rows(df):
    df_orig=df.copy()
    df_shifted=df.shift().copy()
    denom_df_orig=(df_orig**2).sum(axis=1)
    denom_df_shifted=(df_shifted**2).sum(axis=1)
    denominator=np.sqrt(denom_df_orig*denom_df_shifted)
    numerator=(df_orig*df_shifted).sum(axis=1)
    if df.index.nlevels==3:
        cosine_sim_df=(numerator/denominator).reset_index(level=[0,1],drop=True)
    else:
        cosine_sim_df=(numerator/denominator).reset_index(level=[0],drop=True)        
    cosine_sim_df.dropna(inplace=True)
    cosine_sim_df=cosine_sim_df.to_frame()
    return cosine_sim_df


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



total_dec_list=[[1820,1830,1840,1850,1860,1870,1880,1890],[1900,1910,1920,1930,1940,1950,1960,1970,1980,1990],[2000,2010]]
    
    
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

compounds_agnostic_list=[]
constituents_list=[]
compounds_aware_list=[]
modifiers_aware_list=[]
heads_aware_list=[]


for dec_list in total_dec_list:
    
    print(f'Current dec list {dec_list}')
    
    cur_compounds_agnostic=process_decades_compound(dec_list,f'{args.inputdir}',unique_mod_list,unique_head_list,ctype="phrase")
    cur_constituents=process_decades_constituent(dec_list,f'{args.inputdir}',unique_constituent_list,ctype='word')
    
    cur_compounds_aware=process_decades_compound(dec_list,f'{args.inputdir}',unique_mod_list,unique_head_list,ctype="compound")

    cur_modifiers_aware=process_decades_constituent(dec_list,f'{args.inputdir}',unique_mod_list,ctype='modifier')

    cur_heads_aware=process_decades_constituent(dec_list,f'{args.inputdir}',unique_head_list,ctype='head')
    
    compounds_agnostic_list.append(cur_compounds_agnostic)
    constituents_list.append(cur_constituents)
    
    compounds_aware_list.append(cur_compounds_aware)
    modifiers_aware_list.append(cur_modifiers_aware)
    heads_aware_list.append(cur_heads_aware)
    
    
compounds_agnostic=pd.concat(compounds_agnostic_list,ignore_index=True)
constituents=pd.concat(constituents_list,ignore_index=True)

compounds_aware=pd.concat(compounds_aware_list,ignore_index=True)
modifiers_aware=pd.concat(modifiers_aware_list,ignore_index=True)
heads_aware=pd.concat(heads_aware_list,ignore_index=True)


cutoff_list=[0,10,50,100,500,1000]

for cutoff in cutoff_list:
    
    temp_cutoff_str=str(args.temporal)+'_'+str(cutoff)
    
    if os.path.exists(f'{args.outputdir}/temporal_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv'):
        print('File exists')
        continue
        
        
    if cutoff==0:
        print('No cut-off applied')          
    else:
        print(f'Cut-off: {cutoff}')

        compounds_aware=process_cutoff_compound(compounds_aware,cutoff)

        compounds_agnostic=process_cutoff_compound(compounds_agnostic,cutoff)


        constituents=process_cutoff_constituent(constituents,cutoff,ctype='word')
        modifiers_aware=process_cutoff_constituent(modifiers_aware,cutoff,ctype='modifier')
        heads_aware=process_cutoff_constituent(heads_aware,cutoff,ctype='head')



    if args.ppmi:
        print('Applying PPMI')
        compounds_aware=ppmi(compounds_aware)
        modifiers_aware=ppmi(modifiers_aware)
        heads_aware=ppmi(heads_aware)

        compounds_agnostic=ppmi(compounds_agnostic)
        constituents=ppmi(constituents)

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

    compounds_aware=compounds_aware.merge(comp_ratings_df[['modifier','head']],on=['modifier','head'])

    compounds_agnostic=compounds_agnostic.merge(comp_ratings_df[['modifier','head']],on=['modifier','head'])

    heads_agnostic=constituents.copy()
    heads_agnostic_cols=heads_agnostic.columns
    heads_agnostic_cols=['head' if 'word' in x else x for x in heads_agnostic_cols]
    heads_agnostic.columns=heads_agnostic_cols
    heads_agnostic=heads_agnostic.loc[heads_agnostic['head'].isin(unique_head_list)]


    modifiers_agnostic=constituents.copy()
    modifiers_agnostic_cols=modifiers_agnostic.columns
    modifiers_agnostic_cols=['modifier' if 'word' in x else x for x in modifiers_agnostic_cols]
    modifiers_agnostic.columns=modifiers_agnostic_cols
    modifiers_agnostic=modifiers_agnostic.loc[modifiers_agnostic.modifier.isin(unique_mod_list)]

    print('Calculating features')



    print('CompoundAware features')

    change_aware_df=temporal_features(compounds_aware,modifiers_aware,heads_aware,all_comps_aware)


    print('CompoundAgnostic features')

    change_agnostic_df=temporal_features(compounds_agnostic,modifiers_agnostic,heads_agnostic,all_comps_agnostic)


    cur_ratings_aware_df_na,cur_ratings_aware_df_med=merge_comp_ratings(change_aware_df)
    cur_ratings_agnostic_df_na,cur_ratings_agnostic_df_med=merge_comp_ratings(change_agnostic_df)


    print(cur_ratings_aware_df_na.shape[0])

    print('Saving feature datasets')


    cur_ratings_aware_df_na.to_csv(f'{args.outputdir}/temporal_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
    cur_ratings_aware_df_med.to_csv(f'{args.outputdir}/temporal_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)

    cur_ratings_agnostic_df_na.to_csv(f'{args.outputdir}/temporal_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
    cur_ratings_agnostic_df_med.to_csv(f'{args.outputdir}/temporal_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)