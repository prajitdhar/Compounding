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

    
comp_ratings_df=pd.concat([reddy_df,cordeiro90_df,cordeiro100_df])
#comp_ratings_df.drop_duplicates(inplace=True)
if args.tag:
    comp_ratings_df=testset_tagger(comp_ratings_df)
    

    
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

def calculate_compound_features(compounds,modifiers,heads,all_comps,not_found_compounds_df,not_found_modifiers_df,not_found_heads_df):
    
    mod_cols=modifiers.columns.tolist()
    mod_cols=['count' if 'count' in x else x for x in mod_cols]
    modifiers.columns=mod_cols

    head_cols=heads.columns.tolist()
    head_cols=['count' if 'count' in x else x for x in head_cols]
    heads.columns=head_cols

    comp_cols=compounds.columns.tolist()
    comp_cols=['count' if 'count' in x else x for x in comp_cols]
    compounds.columns=comp_cols

    print('Calculating productivity features')
    
    compound_types=compounds.groupby(['time']).size().to_frame()
    compound_types.columns=['comp_size']
    
    modifier_types=modifiers.groupby(['time']).size().to_frame()
    modifier_types.columns=['mod_size']
    
    head_types=heads.groupby(['time']).size().to_frame()
    head_types.columns=['head_size']

    mod_prod=compounds.groupby(['modifier','time']).size().to_frame()
    mod_prod.columns=['mod_prod']
    mod_prod=pd.merge(mod_prod.reset_index(),compound_types.reset_index(),on=['time'])
    mod_prod=pd.merge(mod_prod,modifier_types.reset_index(),on=['time'])

    mod_prod['mod_family_size']=np.log2(mod_prod.mod_prod/mod_prod.comp_size)
    mod_prod['mod_family_size_new']=np.log2(mod_prod.mod_prod/mod_prod.mod_size)


    not_found_mod_prod=not_found_modifiers_df.copy()
    not_found_mod_prod['mod_prod']=0
    not_found_mod_prod=pd.merge(not_found_mod_prod,compound_types.reset_index(),on=['time'])
    not_found_mod_prod=pd.merge(not_found_mod_prod,modifier_types.reset_index(),on=['time'])
    not_found_mod_prod['mod_family_size']=0
    not_found_mod_prod['mod_family_size_new']=0


    head_prod=compounds.groupby(['head','time']).size().to_frame()
    head_prod.columns=['head_prod']
    head_prod=pd.merge(head_prod.reset_index(),compound_types.reset_index(),on=['time'])
    head_prod=pd.merge(head_prod,head_types.reset_index(),on=['time'])

    head_prod['head_family_size']=np.log2(head_prod.head_prod/head_prod.comp_size)
    head_prod['head_family_size_new']=np.log2(head_prod.head_prod/head_prod.head_size)


    not_found_head_prod=not_found_heads_df.copy()
    not_found_head_prod['head_prod']=0
    not_found_head_prod=pd.merge(not_found_head_prod,compound_types.reset_index(),on=['time'])
    not_found_head_prod=pd.merge(not_found_head_prod,head_types.reset_index(),on=['time'])

    not_found_head_prod['head_family_size']=0
    not_found_head_prod['head_family_size_new']=0

    
    mod_prod=pd.concat([mod_prod,not_found_mod_prod],ignore_index=True)
    head_prod=pd.concat([head_prod,not_found_head_prod],ignore_index=True)


    prod1=pd.merge(mod_prod.drop(['mod_size','comp_size'],axis=1),all_comps,on=['modifier','time'])
    productivity=pd.merge(head_prod.drop('head_size',axis=1),prod1,on=['head','time'])


    print('Calculating information theory features')
    
    compound_time_counts=compounds.groupby('time')['count'].sum().to_frame()
    
    compound_time_counts.columns=['N']
    XY=compounds.groupby(['modifier','head','time'])['count'].sum().to_frame()    

    XY.columns=['a']
    
    not_found_XY=not_found_compounds_df.copy()
    not_found_XY['count']=0
    not_found_XY=not_found_XY.groupby(['modifier','head','time'])['count'].sum().to_frame()
    not_found_XY.columns=['a']
    
    
    X_star=compounds.groupby(['modifier','time'])['count'].sum().to_frame()
    X_star.columns=['x_star']
    
    not_found_X_star=not_found_modifiers_df.copy()
    not_found_X_star['count']=0
    not_found_X_star=not_found_X_star.groupby(['modifier','time'])['count'].sum().to_frame()
    not_found_X_star.columns=['x_star']

    Y_star=compounds.groupby(['head','time'])['count'].sum().to_frame()
    Y_star.columns=['star_y']

    not_found_Y_star=not_found_heads_df.copy()
    not_found_Y_star['count']=0    
    not_found_Y_star=not_found_Y_star.groupby(['head','time'])['count'].sum().to_frame()
    not_found_Y_star.columns=['star_y']

    XY=pd.concat([XY,not_found_XY])
    X_star=pd.concat([X_star,not_found_X_star])
    Y_star=pd.concat([Y_star,not_found_Y_star])

    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])    

    information_feat['b']=information_feat['x_star']-information_feat['a']
    information_feat['c']=information_feat['star_y']-information_feat['a']

    information_feat=pd.merge(information_feat,compound_time_counts.reset_index(),on=['time'])

    information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
    information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
    information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']
    information_feat['overflow_check']=np.log2((information_feat['d']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y_bar']+1))
    information_feat['overflow_check'] = information_feat['overflow_check'].fillna(0)
    information_feat['log_ratio']=2*(\
    information_feat['a']*np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))+\
    information_feat['b']*np.log2((information_feat['b']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y_bar']+1))+\
    information_feat['c']*np.log2((information_feat['c']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y']+1))+\
    information_feat['d']*information_feat['overflow_check'])
    information_feat['ppmi']=np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))
    information_feat['local_mi']=information_feat['a']*information_feat['ppmi']
    information_feat.loc[information_feat.ppmi<=0,'ppmi']=0
    information_feat.drop(['a','x_star','star_y','b','c','N','d','x_bar_star','star_y_bar','overflow_check'],axis=1,inplace=True)

    
    compound_features=pd.merge(productivity,information_feat,on=['modifier','head','time'])
    
    print('Frequency features')
            
    modifier_time_counts=modifiers.groupby(['time'])['count'].sum().to_frame()
    modifier_time_counts.columns=['mod_time_count']
    
    head_time_counts=heads.groupby(['time'])['count'].sum().to_frame()
    head_time_counts.columns=['head_time_count']
    
    
    
    
    frequency_feat=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])
    frequency_feat=frequency_feat.merge(Y_star.reset_index(),on=['head','time'])

    frequency_feat=frequency_feat.merge(compound_time_counts.reset_index(),on='time')
    frequency_feat=frequency_feat.merge(modifier_time_counts.reset_index(),on='time')
    frequency_feat=frequency_feat.merge(head_time_counts.reset_index(),on='time')

    frequency_feat.set_index(['modifier','head','time'],inplace=True)
    frequency_feat.columns=['comp_freq','mod_freq','head_freq','N','mod_time_count','head_time_count']
    frequency_feat['comp_tf']=np.log2(1+frequency_feat.comp_freq)
    
    frequency_feat['log_comp_freq']=np.log2(frequency_feat.comp_freq/frequency_feat.N)

    frequency_feat['mod_tf']=np.log2(1+frequency_feat.mod_freq)
    frequency_feat['log_mod_freq']=np.log2(frequency_feat.mod_freq/frequency_feat.N)
    frequency_feat['log_mod_freq_new']=np.log2(frequency_feat.mod_freq/frequency_feat.mod_time_count)

    frequency_feat['head_tf']=np.log2(1+frequency_feat.head_freq)
    frequency_feat['log_head_freq']=np.log2(frequency_feat.head_freq/frequency_feat.N)
    frequency_feat['log_head_freq_new']=np.log2(frequency_feat.head_freq/frequency_feat.head_time_count)
    frequency_feat.fillna(0,inplace=True)
    frequency_feat.drop(['mod_time_count','head_time_count','N'],axis=1,inplace=True)

    
    compound_features=compound_features.merge(frequency_feat.reset_index(),on=['modifier','head','time'])
    
    return compound_features
    
    
def calculate_cosine_features(compounds,modifiers,heads,not_found_compounds_df):
    
    mod_cols=modifiers.columns.tolist()
    mod_cols=['count' if 'count' in x else x for x in mod_cols]
    modifiers.columns=mod_cols

    head_cols=heads.columns.tolist()
    head_cols=['count' if 'count' in x else x for x in head_cols]
    heads.columns=head_cols

    comp_cols=compounds.columns.tolist()
    comp_cols=['count' if 'count' in x else x for x in comp_cols]
    compounds.columns=comp_cols
        
    compound_denom=compounds.copy()
    compound_denom['count']=compound_denom['count']**2
    compound_denom=compound_denom.groupby(['modifier','head','time'])['count'].sum().to_frame()
    compound_denom['count']=np.sqrt(compound_denom['count'])
    compound_denom.columns=['compound_denom']

    modifier_denom=modifiers.copy()
    modifier_denom['count']=modifier_denom['count']**2
    modifier_denom=modifier_denom.groupby(['modifier','time'])['count'].sum().to_frame()
    modifier_denom['count']=np.sqrt(modifier_denom['count'])
    modifier_denom.columns=['modifier_denom']

    head_denom=heads.copy()
    head_denom['count']=head_denom['count']**2
    head_denom=head_denom.groupby(['head','time'])['count'].sum().to_frame()
    head_denom['count']=np.sqrt(head_denom['count'])
    head_denom.columns=['head_denom']

    mod_cols=modifiers.columns.tolist()
    mod_cols=['mod_count' if 'count' in x else x for x in mod_cols]
    modifiers.columns=mod_cols

    head_cols=heads.columns.tolist()
    head_cols=['head_count' if 'count' in x else x for x in head_cols]
    heads.columns=head_cols

    comp_cols=compounds.columns.tolist()
    comp_cols=['comp_count' if 'count' in x else x for x in comp_cols]
    compounds.columns=comp_cols
    
    print('Calculating cosine features')

    print('compound_modifier_sim')
    compound_modifier_sim=pd.merge(compounds,modifiers,on=["modifier","context",'time'])
    compound_modifier_sim['numerator']=compound_modifier_sim['comp_count']*compound_modifier_sim['mod_count']
    compound_modifier_sim=compound_modifier_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    compound_modifier_sim=pd.merge(compound_modifier_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_modifier_sim=pd.merge(compound_modifier_sim,modifier_denom.reset_index(),on=['modifier','time'])
    compound_modifier_sim['sim_with_modifier']=compound_modifier_sim['numerator']/(compound_modifier_sim['compound_denom']*compound_modifier_sim['modifier_denom'])
    compound_modifier_sim.drop(['numerator','compound_denom','modifier_denom'],axis=1,inplace=True)

    print('compound_head_sim')
    compound_head_sim=pd.merge(compounds,heads,on=["head","context",'time'])
    compound_head_sim['numerator']=compound_head_sim['comp_count']*compound_head_sim['head_count']
    compound_head_sim=compound_head_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    compound_head_sim=pd.merge(compound_head_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_head_sim=pd.merge(compound_head_sim,head_denom.reset_index(),on=['head','time'])
    compound_head_sim['sim_with_head']=compound_head_sim['numerator']/(compound_head_sim['compound_denom']*compound_head_sim['head_denom'])
    compound_head_sim.drop(['numerator','compound_denom','head_denom'],axis=1,inplace=True)
    
    cosine_sim_feat=pd.merge(compound_modifier_sim,compound_head_sim,on=['modifier','head','time'])
    
    print('constituent_sim')

    constituent_sim=pd.merge(heads,compounds,on=["head","context","time"])
    #constituent_sim.drop('comp_count',axis=1,inplace=True)
    constituent_sim=pd.merge(constituent_sim,modifiers,on=["modifier","context","time"])
    constituent_sim['numerator']=constituent_sim['head_count']*constituent_sim['mod_count']
    constituent_sim=constituent_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    constituent_sim=pd.merge(constituent_sim.reset_index(),head_denom.reset_index(),on=["head","time"])
    constituent_sim=pd.merge(constituent_sim,modifier_denom.reset_index(),on=["modifier","time"])
    constituent_sim['sim_bw_constituents']=constituent_sim['numerator']/(constituent_sim['head_denom']*constituent_sim['modifier_denom'])
    constituent_sim.drop(['numerator','modifier_denom','head_denom'],axis=1,inplace=True)
    
    
    not_found_constituent_sim=pd.merge(not_found_compounds_df,heads,on=["head",'time'])
    not_found_constituent_sim=pd.merge(not_found_constituent_sim,modifiers,on=["modifier",'context','time'])
    not_found_constituent_sim['numerator']=not_found_constituent_sim['head_count']*not_found_constituent_sim['mod_count']
    not_found_constituent_sim=not_found_constituent_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    not_found_constituent_sim=pd.merge(not_found_constituent_sim.reset_index(),head_denom.reset_index(),on=["head",'time'])
    not_found_constituent_sim=pd.merge(not_found_constituent_sim,modifier_denom.reset_index(),on=["modifier",'time'])
    not_found_constituent_sim['sim_bw_constituents']=not_found_constituent_sim['numerator']/(not_found_constituent_sim['head_denom']*not_found_constituent_sim['modifier_denom'])
    not_found_constituent_sim.drop(['numerator','modifier_denom','head_denom'],axis=1,inplace=True)
    
    constituent_sim=pd.concat([constituent_sim,not_found_constituent_sim])

    
    cosine_sim_feat=pd.merge(cosine_sim_feat,constituent_sim,on=['modifier','head','time'],how='right')
    print('Cordeiro features')

    cosine_sim_feat['beta']=(cosine_sim_feat['sim_with_modifier']-cosine_sim_feat['sim_with_head']*cosine_sim_feat['sim_bw_constituents'])/\
    ((cosine_sim_feat['sim_with_modifier']+cosine_sim_feat['sim_with_head'])*(1-cosine_sim_feat['sim_bw_constituents']))
    cosine_sim_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    na_values = {"beta": 0.5}
    cosine_sim_feat.fillna(value=na_values,inplace=True)

    cosine_sim_feat['geom_mean_sim']=np.sqrt(cosine_sim_feat['sim_with_modifier']*cosine_sim_feat['sim_with_head'])
    cosine_sim_feat['arith_mean_sim']=cosine_sim_feat[['sim_with_modifier', 'sim_with_head']].mean(axis=1)
    
    cpf_head_df=pd.merge(heads,head_denom.reset_index(),on=["head",'time'])
    cpf_head_df['head_value']=cpf_head_df['head_count']/cpf_head_df['head_denom']
    cpf_head_df.drop(['head_count','head_denom'],axis=1,inplace=True)

    cpf_modifier_df=pd.merge(modifiers,modifier_denom.reset_index(),on=["modifier",'time'])
    cpf_modifier_df['modifier_value']=cpf_modifier_df['mod_count']/cpf_modifier_df['modifier_denom']
    cpf_modifier_df.drop(['mod_count','modifier_denom'],axis=1,inplace=True)
    
    cpf_sim=pd.merge(cpf_head_df,compounds,on=["head","context","time"])
    cpf_sim=pd.merge(cpf_sim,cpf_modifier_df,on=["modifier","context","time"])
    cpf_sim=pd.merge(cpf_sim,cosine_sim_feat[['modifier','head','beta','time']],on=["modifier",'head','time'])

    beta=0.0
    cpf_sim['cp_0']=(beta*(cpf_sim['head_value'])/((cpf_sim['head_value']**2).sum()))+((1-beta)*cpf_sim['modifier_value'])

    beta=0.25
    cpf_sim['cp_25']=(beta*cpf_sim['head_value'])+((1-beta)*cpf_sim['modifier_value'])

    beta=0.5
    cpf_sim['cp_50']=(beta*cpf_sim['head_value'])+((1-beta)*cpf_sim['modifier_value'])

    beta=0.75
    cpf_sim['cp_75']=(beta*cpf_sim['head_value'])+((1-beta)*cpf_sim['modifier_value'])

    beta=1
    cpf_sim['cp_100']=(beta*cpf_sim['head_value'])+((1-beta)*cpf_sim['modifier_value'])

    cpf_sim['cp_beta']=(cpf_sim['beta']*cpf_sim['head_value'])+((1-cpf_sim['beta'])*cpf_sim['modifier_value'])

    temp_cdf_df=cpf_sim[['modifier','head','time','cp_0','cp_25','cp_50','cp_75','cp_100','cp_beta']].copy()
    temp_cdf_df['denom_cp_0']=temp_cdf_df['cp_0']**2
    temp_cdf_df['denom_cp_25']=temp_cdf_df['cp_25']**2
    temp_cdf_df['denom_cp_50']=temp_cdf_df['cp_50']**2
    temp_cdf_df['denom_cp_75']=temp_cdf_df['cp_75']**2
    temp_cdf_df['denom_cp_100']=temp_cdf_df['cp_100']**2
    temp_cdf_df['denom_cp_beta']=temp_cdf_df['cp_beta']**2

    cdf_denom=temp_cdf_df.groupby(['modifier','head','time'])[['denom_cp_0','denom_cp_25','denom_cp_50','denom_cp_75','denom_cp_100','denom_cp_beta']].sum()
    cdf_denom['denom_cp_0']=np.sqrt(cdf_denom['denom_cp_0'])
    cdf_denom['denom_cp_25']=np.sqrt(cdf_denom['denom_cp_25'])
    cdf_denom['denom_cp_50']=np.sqrt(cdf_denom['denom_cp_50'])
    cdf_denom['denom_cp_75']=np.sqrt(cdf_denom['denom_cp_75'])
    cdf_denom['denom_cp_100']=np.sqrt(cdf_denom['denom_cp_100'])
    cdf_denom['denom_cp_beta']=np.sqrt(cdf_denom['denom_cp_beta'])

    cpf_sim['num_cp_0']=cpf_sim['comp_count']*cpf_sim['cp_0']
    cpf_sim['num_cp_25']=cpf_sim['comp_count']*cpf_sim['cp_25']
    cpf_sim['num_cp_50']=cpf_sim['comp_count']*cpf_sim['cp_50']
    cpf_sim['num_cp_75']=cpf_sim['comp_count']*cpf_sim['cp_75']
    cpf_sim['num_cp_100']=cpf_sim['comp_count']*cpf_sim['cp_100']
    cpf_sim['num_cp_beta']=cpf_sim['comp_count']*cpf_sim['cp_beta']

    cpf_sim=cpf_sim.groupby(['modifier','head','time'])[['num_cp_0','num_cp_25','num_cp_50','num_cp_75','num_cp_100','num_cp_beta']].sum()
    cpf_sim=pd.merge(cpf_sim,cdf_denom,on=['modifier','head','time'])
    cpf_sim=pd.merge(cpf_sim,compound_denom.reset_index(),on=["modifier","head","time"])
    cpf_sim['sim_cpf_0']=cpf_sim['num_cp_0']/(cpf_sim['denom_cp_0']*cpf_sim['compound_denom'])
    cpf_sim['sim_cpf_25']=cpf_sim['num_cp_25']/(cpf_sim['denom_cp_25']*cpf_sim['compound_denom'])
    cpf_sim['sim_cpf_50']=cpf_sim['num_cp_50']/(cpf_sim['denom_cp_50']*cpf_sim['compound_denom'])
    cpf_sim['sim_cpf_75']=cpf_sim['num_cp_75']/(cpf_sim['denom_cp_75']*cpf_sim['compound_denom'])
    cpf_sim['sim_cpf_100']=cpf_sim['num_cp_100']/(cpf_sim['denom_cp_100']*cpf_sim['compound_denom'])
    cpf_sim['sim_cpf_beta']=cpf_sim['num_cp_beta']/(cpf_sim['denom_cp_beta']*cpf_sim['compound_denom'])

    cpf_sim=cpf_sim[['modifier','head','time','sim_cpf_0','sim_cpf_25','sim_cpf_50','sim_cpf_75','sim_cpf_100','sim_cpf_beta']].copy()
    
    cosine_sim_feat=cosine_sim_feat.merge(cpf_sim,on=["modifier",'head','time'],how='left')
    
    return cosine_sim_feat


def calculate_setting_similarity(compounds,modifiers,heads,compounds_agnostic,modifiers_agnostic,heads_agnostic,compound_list_df):
    
    
    mod_cols=modifiers.columns.tolist()
    mod_cols=['count' if 'count' in x else x for x in mod_cols]
    modifiers.columns=mod_cols

    head_cols=heads.columns.tolist()
    head_cols=['count' if 'count' in x else x for x in head_cols]
    heads.columns=head_cols

    comp_cols=compounds.columns.tolist()
    comp_cols=['count' if 'count' in x else x for x in comp_cols]
    compounds.columns=comp_cols

    
    mod_agn_cols=modifiers_agnostic.columns.tolist()
    mod_agn_cols=['count' if 'count' in x else x for x in mod_agn_cols]
    modifiers_agnostic.columns=mod_agn_cols
    
    head_agn_cols=heads_agnostic.columns.tolist()
    head_agn_cols=['count' if 'count' in x else x for x in head_agn_cols]
    heads_agnostic.columns=head_agn_cols
    
    comp_agn_cols=compounds_agnostic.columns.tolist()
    comp_agn_cols=['count' if 'count' in x else x for x in comp_agn_cols]
    compounds_agnostic.columns=comp_agn_cols
    
    print('Calculating denominator values')

    compound_denom=compounds.copy()
    compound_denom['count']=compound_denom['count']**2
    compound_denom=compound_denom.groupby(['modifier','head','time'])['count'].sum().to_frame()
    compound_denom['count']=np.sqrt(compound_denom['count'])
    compound_denom.columns=['compound_denom']
    
    compound_agnostic_denom=compounds_agnostic.copy()
    compound_agnostic_denom['count']=compound_agnostic_denom['count']**2
    compound_agnostic_denom=compound_agnostic_denom.groupby(['modifier','head','time'])['count'].sum().to_frame()
    compound_agnostic_denom['count']=np.sqrt(compound_agnostic_denom['count'])
    compound_agnostic_denom.columns=['compound_agn_denom']
    

    modifier_denom=modifiers.copy()
    modifier_denom['count']=modifier_denom['count']**2
    modifier_denom=modifier_denom.groupby(['modifier','time'])['count'].sum().to_frame()
    modifier_denom['count']=np.sqrt(modifier_denom['count'])
    modifier_denom.columns=['modifier_denom']
    
    modifier_agnostic_denom=modifiers_agnostic.copy()
    modifier_agnostic_denom['count']=modifier_agnostic_denom['count']**2
    modifier_agnostic_denom=modifier_agnostic_denom.groupby(['modifier','time'])['count'].sum().to_frame()
    modifier_agnostic_denom['count']=np.sqrt(modifier_agnostic_denom['count'])
    modifier_agnostic_denom.columns=['modifier_agn_denom']
    
    
    head_denom=heads.copy()
    head_denom['count']=head_denom['count']**2
    head_denom=head_denom.groupby(['head','time'])['count'].sum().to_frame()
    head_denom['count']=np.sqrt(head_denom['count'])
    head_denom.columns=['head_denom']
    
    head_agnostic_denom=heads_agnostic.copy()
    head_agnostic_denom['count']=head_agnostic_denom['count']**2
    head_agnostic_denom=head_agnostic_denom.groupby(['head','time'])['count'].sum().to_frame()
    head_agnostic_denom['count']=np.sqrt(head_agnostic_denom['count'])
    head_agnostic_denom.columns=['head_agn_denom'] 
    
    
    mod_cols=modifiers.columns.tolist()
    mod_cols=['mod_count' if 'count' in x else x for x in mod_cols]
    modifiers.columns=mod_cols

    head_cols=heads.columns.tolist()
    head_cols=['head_count' if 'count' in x else x for x in head_cols]
    heads.columns=head_cols

    comp_cols=compounds.columns.tolist()
    comp_cols=['comp_count' if 'count' in x else x for x in comp_cols]
    compounds.columns=comp_cols

    
    mod_agn_cols=modifiers_agnostic.columns.tolist()
    mod_agn_cols=['mod_agn_count' if 'count' in x else x for x in mod_agn_cols]
    modifiers_agnostic.columns=mod_agn_cols
    
    head_agn_cols=heads_agnostic.columns.tolist()
    head_agn_cols=['head_agn_count' if 'count' in x else x for x in head_agn_cols]
    heads_agnostic.columns=head_agn_cols
    
    comp_agn_cols=compounds_agnostic.columns.tolist()
    comp_agn_cols=['comp_agn_count' if 'count' in x else x for x in comp_agn_cols]
    compounds_agnostic.columns=comp_agn_cols

    print('Calculating cosine features')

    compound_setting_sim=pd.merge(compounds,compounds_agnostic,on=["modifier",'head',"context",'time'])
    compound_setting_sim['numerator']=compound_setting_sim['comp_count']*compound_setting_sim['comp_agn_count']
    compound_setting_sim=compound_setting_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()

    compound_setting_sim=pd.merge(compound_setting_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_setting_sim=pd.merge(compound_setting_sim,compound_agnostic_denom.reset_index(),on=["modifier","head",'time'])

    compound_setting_sim['sim_bw_settings_comp']=compound_setting_sim['numerator']/(compound_setting_sim['compound_denom']*compound_setting_sim['compound_agn_denom'])
    
    compound_setting_sim=pd.merge(compound_setting_sim,compound_list_df,on=["modifier",'head','time'],how='outer')

    compound_setting_sim.set_index(['modifier','head','time'],inplace=True)
    compound_setting_sim.drop(['numerator','compound_denom','compound_agn_denom'],axis=1,inplace=True)

    head_setting_sim=pd.merge(heads,heads_agnostic,on=['head',"context",'time'])
    head_setting_sim['numerator']=head_setting_sim['head_count']*head_setting_sim['head_agn_count']
    head_setting_sim=head_setting_sim.groupby(['head','time'])['numerator'].sum().to_frame()

    head_setting_sim=pd.merge(head_setting_sim.reset_index(),head_denom.reset_index(),on=["head",'time'])
    head_setting_sim=pd.merge(head_setting_sim,head_agnostic_denom.reset_index(),on=["head",'time'])

    head_setting_sim['sim_bw_settings_head']=head_setting_sim['numerator']/(head_setting_sim['head_denom']*head_setting_sim['head_agn_denom'])
    head_setting_sim.set_index(['head','time'],inplace=True)
    head_setting_sim.drop(['numerator','head_denom','head_agn_denom'],axis=1,inplace=True)
    
    
    
    compound_setting_sim=pd.merge(compound_setting_sim.reset_index(),head_setting_sim.reset_index(),on=["head",'time'])


    modifier_setting_sim=pd.merge(modifiers,modifiers_agnostic,on=['modifier',"context",'time'])
    modifier_setting_sim['numerator']=modifier_setting_sim['mod_count']*modifier_setting_sim['mod_agn_count']
    modifier_setting_sim=modifier_setting_sim.groupby(['modifier','time'])['numerator'].sum().to_frame()

    modifier_setting_sim=pd.merge(modifier_setting_sim.reset_index(),modifier_denom.reset_index(),on=["modifier",'time'])
    modifier_setting_sim=pd.merge(modifier_setting_sim,modifier_agnostic_denom.reset_index(),on=["modifier",'time'])

    modifier_setting_sim['sim_bw_settings_modifier']=modifier_setting_sim['numerator']/(modifier_setting_sim['modifier_denom']*modifier_setting_sim['modifier_agn_denom'])
    modifier_setting_sim.set_index(['modifier','time'],inplace=True)
    modifier_setting_sim.drop(['numerator','modifier_denom','modifier_agn_denom'],axis=1,inplace=True)

    compound_setting_sim=pd.merge(compound_setting_sim,modifier_setting_sim.reset_index(),on=["modifier",'time'])
    
    return compound_setting_sim


def plotting(compound_df,sim_df):
    
    print('Plotting')
    plotdir=args.plotdir
        
    compounds_complete_index=compound_df.index
    print(len(compounds_complete_index))

    compound_pivot=pd.pivot_table(sim_df,columns='time',index=['modifier','head'],values='sim_bw_constituents')
    compounds_decades_all_index=compound_pivot.dropna().index
    print(len(compounds_decades_all_index))

    columns_names_1900_end=compound_pivot.columns[compound_pivot.columns>=1900]
    compounds_1900_end_index=compound_pivot.loc[:,columns_names_1900_end].dropna().index
    print(len(compounds_1900_end_index))

    columns_names_1950_end=compound_pivot.columns[compound_pivot.columns>=1950]
    compounds_1950_end_index=compound_pivot.loc[:,columns_names_1950_end].dropna().index
    print(len(compounds_1950_end_index))

    columns_names_2000_end=compound_pivot.columns[compound_pivot.columns>=2000]
    compounds_2000_end_index=compound_pivot.loc[:,columns_names_2000_end].dropna().index
    print(len(compounds_2000_end_index))

    compound_index_lst=[compounds_decades_all_index,compounds_1900_end_index,compounds_1950_end_index,compounds_2000_end_index]
    tags_lst=['all','1900','1950','2000']


    for cur_index_lst,cur_tag in zip(compound_index_lst,tags_lst):
        if cur_tag=='1950' and temporal==100:
            continue

        print(cur_tag)
        if str(cur_tag).isdigit():
            if cur_tag=='1950' and temporal==20:
                cur_decades = [str(year) for year in list(range(1940, 2010, temporal))]
            else:
                cur_decades = [str(year) for year in list(range(int(cur_tag), 2010, temporal))]
            cur_columns_regex = re.compile(".+({})$".format("|".join(cur_decades)))
            cur_df = compound_df.loc[cur_index_lst,[col for col in compound_df if re.search(cur_columns_regex, col)]]
        else:
            cur_df=compound_df.loc[cur_index_lst]
        
        if cur_df.shape[0]==0:
            print("Nothing to plot")
            continue
        else:

            print('All compounds')
            print('Raw frequency features')
            comp_freq_cols=[col for col in cur_df if col.startswith('comp_freq')]
            mod_freq_cols=[col for col in cur_df if col.startswith('mod_freq')]
            head_freq_cols=[col for col in cur_df if col.startswith('head_freq')]
            raw_freq_cols=comp_freq_cols+mod_freq_cols+head_freq_cols

            plot_freq_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=raw_freq_cols)
            plot_freq_df[['variable','time']]=plot_freq_df['variable'].str.split(':',expand=True)


            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_freq_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper left')
            g.set_xlabel("Time")
            g.set_ylabel("Frequency")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/freq_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
            
            
            print('Log frequency features')

            comp_tf_cols=[col for col in cur_df if col.startswith('comp_tf')]
            mod_tf_cols=[col for col in cur_df if col.startswith('mod_tf')]
            head_tf_cols=[col for col in cur_df if col.startswith('head_tf')]
            log_freq_cols=comp_tf_cols+mod_tf_cols+head_tf_cols

            plot_tf_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=log_freq_cols)
            plot_tf_df[['variable','time']]=plot_tf_df['variable'].str.split(':',expand=True)

            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_tf_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper left')
            g.set_xlabel("Time")
            g.set_ylabel("Log Frequency")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/log_freq_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
            
            print('Family size')

            mod_family_size_cols=[col for col in cur_df if col.startswith('mod_family_size')]
            head_family_size_cols=[col for col in cur_df if col.startswith('head_family_size')]
            fam_size_cols=mod_family_size_cols+head_family_size_cols

            plot_family_size_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=fam_size_cols)
            plot_family_size_df[['variable','time']]=plot_family_size_df['variable'].str.split(':',expand=True)

            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_family_size_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper left')
            g.set_xlabel("Time")
            g.set_ylabel("Family Size")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/family_size_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
            
            print('Productivity')

            mod_prod_cols=[col for col in cur_df if col.startswith('mod_prod')]
            head_prod_cols=[col for col in cur_df if col.startswith('head_prod')]
            prod_cols=mod_prod_cols+head_prod_cols


            plot_prod_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=prod_cols)
            plot_prod_df[['variable','time']]=plot_prod_df['variable'].str.split(':',expand=True)

            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_prod_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper left')
            g.set_xlabel("Time")
            g.set_ylabel("Productivity")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/prod_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
            
            print('Information Theory')

            log_ratio_cols=[col for col in cur_df if col.startswith('log_ratio')]
            local_mi_cols=[col for col in cur_df if col.startswith('local_mi')]

            plot_log_ratio_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=log_ratio_cols)
            plot_log_ratio_df[['variable','time']]=plot_log_ratio_df['variable'].str.split(':',expand=True)

            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_log_ratio_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper left')
            g.set_xlabel("Time")
            g.set_ylabel("Log Ratio")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/log_ratio_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

            
            plot_lmi_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=local_mi_cols)
            plot_lmi_df[['variable','time']]=plot_lmi_df['variable'].str.split(':',expand=True)

            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_lmi_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper left')
            g.set_xlabel("Time")
            g.set_ylabel("Local MI")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/lmi_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
            
            
            print('Cosine')
            sim_with_modifier_cols=[col for col in cur_df if col.startswith('sim_with_modifier')]
            sim_with_head_cols=[col for col in cur_df if col.startswith('sim_with_head')]
            sim_bw_constituents_cols=[col for col in cur_df if col.startswith('sim_bw_constituents')]
            cosine_cols=sim_with_modifier_cols+sim_with_head_cols+sim_bw_constituents_cols

            plot_cosine_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=cosine_cols)
            plot_cosine_df[['variable','time']]=plot_cosine_df['variable'].str.split(':',expand=True)

            plt.figure(figsize=(15,15))
            g=sns.lineplot(x="time", y="value", hue="variable",data=plot_cosine_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
            g.legend(loc='upper right')
            g.set_xlabel("Time")
            g.set_ylabel("Cosine Similarity")
            plt.setp(g.get_xticklabels(), rotation=60)
            plt.savefig(f'{plotdir}/cosine_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
            
            
            
            print('Ratings only dataset')
            cur_ratings_df=cur_df.reset_index().merge(ratings_df,on=['modifier','head'])
            print(cur_ratings_df.shape)
            
            if cur_ratings_df.shape[0]==0:
                print("Nothing to plot")
                continue
                
            else:
                
                print('Raw frequency features')
                plot_freq_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=raw_freq_cols)
                plot_freq_ratings_df[['variable','time']]=plot_freq_ratings_df['variable'].str.split(':',expand=True)
                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_freq_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Frequency")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/freq_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                print('Log frequency features')

                plot_tf_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=log_freq_cols)
                plot_tf_ratings_df[['variable','time']]=plot_tf_ratings_df['variable'].str.split(':',expand=True)
                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_tf_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Log Frequency")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/log_freq_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                print('Family size')

                plot_family_size_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=fam_size_cols)
                plot_family_size_ratings_df[['variable','time']]=plot_family_size_ratings_df['variable'].str.split(':',expand=True)
                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_family_size_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Family Size")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/family_size_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                print('Productivity')

                plot_prod_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=prod_cols)
                plot_prod_ratings_df[['variable','time']]=plot_prod_ratings_df['variable'].str.split(':',expand=True)
                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_prod_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Productivity")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/prod_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                print('Information Theory')
                
                plot_log_ratio_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=log_ratio_cols)
                plot_log_ratio_ratings_df[['variable','time']]=plot_log_ratio_ratings_df['variable'].str.split(':',expand=True)

                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_log_ratio_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Log Ratio")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/log_ratio_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                
                plot_lmi_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=local_mi_cols)
                plot_lmi_ratings_df[['variable','time']]=plot_lmi_ratings_df['variable'].str.split(':',expand=True)

                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_lmi_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Local MI")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/lmi_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                print('Cosine features')

                plot_cosine_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=cosine_cols)
                plot_cosine_ratings_df[['variable','time']]=plot_cosine_ratings_df['variable'].str.split(':',expand=True)
                plt.figure(figsize=(15,15))
                g=sns.lineplot(x="time", y="value", hue="variable",data=plot_cosine_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                g.legend(loc='upper left')
                g.set_xlabel("Time")
                g.set_ylabel("Cosine Similarity")
                plt.setp(g.get_xticklabels(), rotation=60)
                plt.savefig(f'{plotdir}/cosine_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)
                
                print('Saving the dataset')

                cur_ratings_df.to_csv(f'{args.outputdir}/features_{cur_tag}_{comp_str}_{tag_str}_{temp_cutoff_str}.csv',sep='\t',index=False)


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
temporal_list=[10000,10,20,50,100]


if args.tag:
    tag_str='Tagged'
else:
    tag_str='UnTagged'


    
complete_phrases=pd.read_pickle(args.inputdir+"/phrases.pkl")


complete_words=pd.read_pickle(args.inputdir+"/words.pkl")
    
    
    
    
complete_compounds=pd.read_pickle(args.inputdir+"/compounds.pkl")
   

complete_modifiers=pd.read_pickle(args.inputdir+"/modifiers.pkl")

complete_heads=pd.read_pickle(args.inputdir+"/heads.pkl")

    
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
            compounds_aware=ppmi(compounds)
            heads_aware=ppmi(heads)
            modifiers_aware=ppmi(modifiers)
                        
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
        compound_features_aware=calculate_compound_features(compounds_aware,modifiers_aware,heads_aware,all_comps_aware,not_found_compounds_aware_df,not_found_modifiers_aware_df,not_found_heads_aware_df)
        compound_features_aware=compound_features_aware.loc[(compound_features_aware.modifier.isin(unique_mod_list))&(compound_features_aware['head'].isin(unique_head_list))]

        reduced_compounds_aware=compounds_aware.loc[(compounds_aware.modifier.isin(unique_mod_list))&(compounds_aware['head'].isin(unique_head_list))]
        reduced_modifiers_aware=modifiers_aware.loc[modifiers_aware.modifier.isin(unique_mod_list)]
        reduced_heads_aware=heads_aware.loc[heads_aware['head'].isin(unique_head_list)]

        cosine_sim_feat_aware=calculate_cosine_features(reduced_compounds_aware,reduced_modifiers_aware,reduced_heads_aware,not_found_compounds_aware_df)


        print('CompoundAgnostic features')

        compound_features_agnostic=calculate_compound_features(compounds_agnostic,modifiers_agnostic,heads_agnostic,all_comps_agnostic,not_found_compounds_agnostic_df,not_found_modifiers_agnostic_df,not_found_heads_agnostic_df)
        compound_features_agnostic=compound_features_agnostic.loc[(compound_features_agnostic.modifier.isin(unique_mod_list))&(compound_features_agnostic['head'].isin(unique_head_list))]


        reduced_compounds_agnostic=compounds_agnostic.loc[(compounds_agnostic.modifier.isin(unique_mod_list))&(compounds_agnostic['head'].isin(unique_head_list))]
        reduced_modifiers_agnostic=modifiers_agnostic.loc[modifiers_agnostic.modifier.isin(unique_mod_list)]
        reduced_heads_agnostic=heads_agnostic.loc[heads_agnostic['head'].isin(unique_head_list)]

        cosine_sim_feat_agnostic=calculate_cosine_features(reduced_compounds_agnostic,reduced_modifiers_agnostic,reduced_heads_agnostic,not_found_compounds_agnostic_df)



        print('Combined cosine features')
        compound_setting_sim=calculate_setting_similarity(reduced_compounds_aware,reduced_modifiers_aware,reduced_heads_aware,reduced_compounds_agnostic,reduced_modifiers_agnostic,reduced_heads_agnostic,compound_list_agnostic_df)


        print('Combining all compound aware features')

        features_aware_df=pd.merge(cosine_sim_feat_aware,compound_setting_sim,on=['modifier','head','time'],how='outer')
        features_aware_df=features_aware_df.merge(compound_features_aware,on=['modifier','head','time'],how='left')
        features_aware_df=features_aware_df.merge(compound_list_aware_df,on=['modifier','head','time'],how='right')


        print('Combining all compound agnostic features')

        features_agnostic_df=pd.merge(cosine_sim_feat_agnostic,compound_setting_sim,on=['modifier','head','time'],how='outer')
        features_agnostic_df=features_agnostic_df.merge(compound_features_agnostic,on=['modifier','head','time'],how='left')
        features_agnostic_df=features_agnostic_df.merge(compound_list_agnostic_df,on=['modifier','head','time'],how='right')

        if args.ppmi:
            ppmi_str="PPMI"
        else:
            ppmi_str="RAW"

        if args.tag:
            tag_str='Tagged'
        else:
            tag_str='UnTagged'

        temp_cutoff_str=str(temporal)+'_'+str(cutoff)

        cur_ratings_aware_df_na,cur_ratings_aware_df_med=merge_comp_ratings(features_aware_df)
        cur_ratings_agnostic_df_na,cur_ratings_agnostic_df_med=merge_comp_ratings(features_agnostic_df)
        cur_ratings_aware_df_na.loc[:,~cur_ratings_aware_df_na.columns.str.contains('setting')].to_csv(f'{args.outputdir}/features_CompoundAware_woSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
        cur_ratings_aware_df_med.loc[:,~cur_ratings_aware_df_med.columns.str.contains('setting')].to_csv(f'{args.outputdir}/features_CompoundAware_woSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)

        cur_ratings_agnostic_df_na.loc[:,~cur_ratings_agnostic_df_na.columns.str.contains('setting')].to_csv(f'{args.outputdir}/features_CompoundAgnostic_woSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
        cur_ratings_agnostic_df_med.loc[:,~cur_ratings_agnostic_df_med.columns.str.contains('setting')].to_csv(f'{args.outputdir}/features_CompoundAgnostic_woSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)


        cur_ratings_aware_df_na.to_csv(f'{args.outputdir}/features_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
        cur_ratings_aware_df_med.to_csv(f'{args.outputdir}/features_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)

        cur_ratings_agnostic_df_na.to_csv(f'{args.outputdir}/features_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
        cur_ratings_agnostic_df_med.to_csv(f'{args.outputdir}/features_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)
    
    
    
