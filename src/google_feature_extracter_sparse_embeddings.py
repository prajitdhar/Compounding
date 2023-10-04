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


parser = argparse.ArgumentParser(description='Compute features from sparse dataset for google version')

parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')
parser.add_argument('--tag', action='store_true',
                    help='Should the POS tag be kept?')
parser.add_argument('--ppmi', action='store_true',
                    help='Should co-occurence matrix be converted to PPMI values')
parser.add_argument('--plot', action='store_true',
                    help='Should plots be saved')
parser.add_argument('--temporal',  type=int,
                    help='Value to bin the temporal information: 10000 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')
parser.add_argument('--cutoff', type=int, default=0,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')


args = parser.parse_args()


reddy_df=pd.read_csv('/data/dharp/compounds/Compounding/data/reddy_90.txt',sep='\t')
reddy_df['source']='reddy'
cordeiro90_df=pd.read_csv('/data/dharp/compounds/Compounding/data/cordeiro_90.txt',sep='\t')
cordeiro90_df['source']='cordeiro90'
cordeiro100_df=pd.read_csv('/data/dharp/compounds/Compounding/data/cordeiro_100.txt',sep='\t')
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

    
def process_decades_compound(dec_list,input_dir,ctype='compound'):

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
        
        
    return complete_df


def process_decades_constituent(dec_list,input_dir,ctype='word'):
        
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

    return complete_df


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



def process_cutoff_compound(df):

    df=df.loc[df.groupby(['modifier','head','time'])['count'].transform('sum').gt(args.cutoff)]
    
    return df


def process_cutoff_constituent(df,ctype='word'):

    df=df.loc[df.groupby([ctype,'time'])['count'].transform('sum').gt(args.cutoff)]
    
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


def calculate_setting_similarity(compounds_aware,modifiers_aware,heads_aware,compounds_agnostic,modifiers_agnostic,heads_agnostic,compound_list_df):
    
    mod_awr_cols=modifiers_aware.columns.tolist()
    mod_awr_cols=['count' if 'count' in x else x for x in mod_awr_cols]
    modifiers_aware.columns=mod_awr_cols

    head_awr_cols=heads_aware.columns.tolist()
    head_awr_cols=['count' if 'count' in x else x for x in head_awr_cols]
    heads_aware.columns=head_awr_cols

    comp_awr_cols=compounds_aware.columns.tolist()
    comp_awr_cols=['count' if 'count' in x else x for x in comp_awr_cols]
    compounds_aware.columns=comp_awr_cols

    
    mod_agn_cols=modifiers_agnostic.columns.tolist()
    mod_agn_cols=['count' if 'count' in x else x for x in mod_agn_cols]
    modifiers_agnostic.columns=mod_agn_cols
    
    head_agn_cols=heads_agnostic.columns.tolist()
    head_agn_cols=['count' if 'count' in x else x for x in head_agn_cols]
    heads_agnostic.columns=head_agn_cols
    
    comp_agn_cols=compounds_agnostic.columns.tolist()
    comp_agn_cols=['count' if 'count' in x else x for x in comp_agn_cols]
    compounds_agnostic.columns=comp_agn_cols
    
    print('Calculating setting frequency values')
    
    
    num_modifiers_features_df=(modifiers_aware.groupby(['modifier','time'])['count'].agg(perc_token_modifier='sum', perc_type_modifier='size')/modifiers_agnostic.groupby(['modifier','time'])['count'].agg(perc_token_modifier='sum', perc_type_modifier='size')).reset_index()
    num_heads_features_df=(heads_aware.groupby(['head','time'])['count'].agg(perc_token_head='sum', perc_type_head='size')/heads_agnostic.groupby(['head','time'])['count'].agg(perc_token_head='sum', perc_type_head='size')).reset_index()

    num_compounds_features_df=(compounds_aware.groupby(['modifier','head','time'])['count'].agg(perc_token_comp='sum', perc_type_comp='size')/compounds_agnostic.groupby(['modifier','head','time'])['count'].agg(perc_token_comp='sum', perc_type_comp='size')).reset_index()
    num_compounds_features_df=pd.merge(num_compounds_features_df,num_modifiers_features_df,on=['modifier','time'])
    num_compounds_features_df=pd.merge(num_compounds_features_df,num_heads_features_df,on=['head','time'])
    
    print('Calculating denominator values')

    compound_aware_denom=compounds_aware.copy()
    compound_aware_denom['count']=compound_aware_denom['count']**2
    compound_aware_denom=compound_aware_denom.groupby(['modifier','head','time'])['count'].sum().to_frame()
    compound_aware_denom['count']=np.sqrt(compound_aware_denom['count'])
    compound_aware_denom.columns=['compound_awr_denom']
    
    compound_agnostic_denom=compounds_agnostic.copy()
    compound_agnostic_denom['count']=compound_agnostic_denom['count']**2
    compound_agnostic_denom=compound_agnostic_denom.groupby(['modifier','head','time'])['count'].sum().to_frame()
    compound_agnostic_denom['count']=np.sqrt(compound_agnostic_denom['count'])
    compound_agnostic_denom.columns=['compound_agn_denom']
    

    modifier_aware_denom=modifiers_aware.copy()
    modifier_aware_denom['count']=modifier_aware_denom['count']**2
    modifier_aware_denom=modifier_aware_denom.groupby(['modifier','time'])['count'].sum().to_frame()
    modifier_aware_denom['count']=np.sqrt(modifier_aware_denom['count'])
    modifier_aware_denom.columns=['modifier_awr_denom']
    
    modifier_agnostic_denom=modifiers_agnostic.copy()
    modifier_agnostic_denom['count']=modifier_agnostic_denom['count']**2
    modifier_agnostic_denom=modifier_agnostic_denom.groupby(['modifier','time'])['count'].sum().to_frame()
    modifier_agnostic_denom['count']=np.sqrt(modifier_agnostic_denom['count'])
    modifier_agnostic_denom.columns=['modifier_agn_denom']
    
    
    head_aware_denom=heads_aware.copy()
    head_aware_denom['count']=head_aware_denom['count']**2
    head_aware_denom=head_aware_denom.groupby(['head','time'])['count'].sum().to_frame()
    head_aware_denom['count']=np.sqrt(head_aware_denom['count'])
    head_aware_denom.columns=['head_awr_denom']
    
    head_agnostic_denom=heads_agnostic.copy()
    head_agnostic_denom['count']=head_agnostic_denom['count']**2
    head_agnostic_denom=head_agnostic_denom.groupby(['head','time'])['count'].sum().to_frame()
    head_agnostic_denom['count']=np.sqrt(head_agnostic_denom['count'])
    head_agnostic_denom.columns=['head_agn_denom'] 
    
    
    mod_awr_cols=modifiers_aware.columns.tolist()
    mod_awr_cols=['mod_awr_count' if 'count' in x else x for x in mod_awr_cols]
    modifiers_aware.columns=mod_awr_cols

    head_awr_cols=heads_aware.columns.tolist()
    head_awr_cols=['head_awr_count' if 'count' in x else x for x in head_awr_cols]
    heads_aware.columns=head_awr_cols

    comp_awr_cols=compounds_aware.columns.tolist()
    comp_awr_cols=['comp_awr_count' if 'count' in x else x for x in comp_awr_cols]
    compounds_aware.columns=comp_awr_cols

    
    mod_agn_cols=modifiers_agnostic.columns.tolist()
    mod_agn_cols=['mod_agn_count' if 'count' in x else x for x in mod_agn_cols]
    modifiers_agnostic.columns=mod_agn_cols
    
    head_agn_cols=heads_agnostic.columns.tolist()
    head_agn_cols=['head_agn_count' if 'count' in x else x for x in head_agn_cols]
    heads_agnostic.columns=head_agn_cols
    
    comp_agn_cols=compounds_agnostic.columns.tolist()
    comp_agn_cols=['comp_agn_count' if 'count' in x else x for x in comp_agn_cols]
    compounds_agnostic.columns=comp_agn_cols

    print('Calculating cosine setting features')

    compound_setting_sim=pd.merge(compounds_aware,compounds_agnostic,on=["modifier",'head',"context",'time'])
    compound_setting_sim['numerator']=compound_setting_sim['comp_awr_count']*compound_setting_sim['comp_agn_count']
    compound_setting_sim=compound_setting_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()

    compound_setting_sim=pd.merge(compound_setting_sim.reset_index(),compound_aware_denom.reset_index(),on=["modifier","head",'time'])
    compound_setting_sim=pd.merge(compound_setting_sim,compound_agnostic_denom.reset_index(),on=["modifier","head",'time'])

    compound_setting_sim['sim_bw_settings_comp']=compound_setting_sim['numerator']/(compound_setting_sim['compound_awr_denom']*compound_setting_sim['compound_agn_denom'])
    
    compound_setting_sim=pd.merge(compound_setting_sim,compound_list_df,on=["modifier",'head','time'],how='outer')

    compound_setting_sim.set_index(['modifier','head','time'],inplace=True)
    compound_setting_sim.drop(['numerator','compound_awr_denom','compound_agn_denom'],axis=1,inplace=True)

    head_setting_sim=pd.merge(heads_aware,heads_agnostic,on=['head',"context",'time'])
    head_setting_sim['numerator']=head_setting_sim['head_awr_count']*head_setting_sim['head_agn_count']
    head_setting_sim=head_setting_sim.groupby(['head','time'])['numerator'].sum().to_frame()

    head_setting_sim=pd.merge(head_setting_sim.reset_index(),head_aware_denom.reset_index(),on=["head",'time'])
    head_setting_sim=pd.merge(head_setting_sim,head_agnostic_denom.reset_index(),on=["head",'time'])

    head_setting_sim['sim_bw_settings_head']=head_setting_sim['numerator']/(head_setting_sim['head_awr_denom']*head_setting_sim['head_agn_denom'])
    head_setting_sim.set_index(['head','time'],inplace=True)
    head_setting_sim.drop(['numerator','head_awr_denom','head_agn_denom'],axis=1,inplace=True)
    
    compound_setting_sim=pd.merge(compound_setting_sim.reset_index(),head_setting_sim.reset_index(),on=["head",'time'])


    modifier_setting_sim=pd.merge(modifiers_aware,modifiers_agnostic,on=['modifier',"context",'time'])
    modifier_setting_sim['numerator']=modifier_setting_sim['mod_awr_count']*modifier_setting_sim['mod_agn_count']
    modifier_setting_sim=modifier_setting_sim.groupby(['modifier','time'])['numerator'].sum().to_frame()

    modifier_setting_sim=pd.merge(modifier_setting_sim.reset_index(),modifier_aware_denom.reset_index(),on=["modifier",'time'])
    modifier_setting_sim=pd.merge(modifier_setting_sim,modifier_agnostic_denom.reset_index(),on=["modifier",'time'])

    modifier_setting_sim['sim_bw_settings_modifier']=modifier_setting_sim['numerator']/(modifier_setting_sim['modifier_awr_denom']*modifier_setting_sim['modifier_agn_denom'])
    modifier_setting_sim.set_index(['modifier','time'],inplace=True)
    modifier_setting_sim.drop(['numerator','modifier_awr_denom','modifier_agn_denom'],axis=1,inplace=True)

    compound_setting_sim=pd.merge(compound_setting_sim,modifier_setting_sim.reset_index(),on=["modifier",'time'])
    
    compound_setting_sim=pd.merge(compound_setting_sim,num_compounds_features_df,on=["modifier","head",'time'])
    
    return compound_setting_sim


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


def feature_extractor_dec(dec_list):
    
    print(f'Current dec list {dec_list}')
    
    compounds_agnostic=process_decades_compound(dec_list,f'{args.inputdir}',ctype="phrase")

    constituents=process_decades_constituent(dec_list,f'{args.inputdir}',ctype='word')
    
    
    compounds_aware=process_decades_compound(dec_list,f'{args.inputdir}',ctype="compound")

    modifiers_aware=process_decades_constituent(dec_list,f'{args.inputdir}',ctype='modifier')

    heads_aware=process_decades_constituent(dec_list,f'{args.inputdir}',ctype='head')
    
    
    if args.cutoff==0:
        print('No cut-off applied')          
    else:
        print(f'Cut-off: {args.cutoff}')
        compounds_aware=process_cutoff_compound(compounds_aware)
        compounds_agnostic=process_cutoff_compound(compounds_agnostic)
        
        constituents=process_cutoff_constituent(constituents,ctype='word')
        modifiers_aware=process_cutoff_constituent(modifiers_aware,ctype='modifier')
        heads_aware=process_cutoff_constituent(heads_aware,ctype='head')

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
    
    
    heads_agnostic=constituents.copy()
    heads_agnostic_cols=heads_agnostic.columns
    heads_agnostic_cols=['head' if 'word' in x else x for x in heads_agnostic_cols]
    heads_agnostic.columns=heads_agnostic_cols

    modifiers_agnostic=constituents.copy()
    modifiers_agnostic_cols=modifiers_agnostic.columns
    modifiers_agnostic_cols=['modifier' if 'word' in x else x for x in modifiers_agnostic_cols]
    modifiers_agnostic.columns=modifiers_agnostic_cols

    
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

    
    print('Combining all compound agnostic features')
    
    features_agnostic_df=pd.merge(cosine_sim_feat_agnostic,compound_features_agnostic,on=['modifier','head','time'],how='outer')
    
    return features_aware_df,features_agnostic_df


if args.temporal!=10000:
    total_dec_list=[[1820,1830,1840,1850,1860,1870,1880,1890],[1900,1910,1920,1930,1940,1950,1960,1970,1980,1990],[2000,2010]]
    
else:
    total_dec_list=[[1820,1830,1840,1850,1860,1870,1880,1890,1900,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]]
    
    
if args.ppmi:
    ppmi_str="PPMI"
else:
    ppmi_str="RAW"
    
if args.tag:
    tag_str='Tagged'
else:
    tag_str='UnTagged'
    
temp_cutoff_str=str(args.temporal)+'_'+str(args.cutoff)


features_aware_df_list=[]
features_agnostic_df_list=[]

for dec_list in total_dec_list:
    dfs=feature_extractor_dec(dec_list)
    features_aware_df_list.append(dfs[0])
    features_agnostic_df_list.append(dfs[1])
    
features_aware_df=pd.concat(features_aware_df_list)
features_agnostic_df=pd.concat(features_agnostic_df_list)


cur_ratings_aware_df_na,cur_ratings_aware_df_med=merge_comp_ratings(features_aware_df)
cur_ratings_agnostic_df_na,cur_ratings_agnostic_df_med=merge_comp_ratings(features_agnostic_df)
print(cur_ratings_aware_df_na.shape[0])

print('Saving feature datasets')

      
cur_ratings_aware_df_na.to_csv(f'{args.outputdir}/features_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
cur_ratings_aware_df_med.to_csv(f'{args.outputdir}/features_CompoundAware_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)

cur_ratings_agnostic_df_na.to_csv(f'{args.outputdir}/features_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_na.csv',sep='\t',index=False)
cur_ratings_agnostic_df_med.to_csv(f'{args.outputdir}/features_CompoundAgnostic_withSetting_{ppmi_str}_{tag_str}_{temp_cutoff_str}_med.csv',sep='\t',index=False)