import pandas as pd
import numpy as np
import argparse
import time
import pickle as pkl

from itertools import product
from functools import reduce


def year_binner(year,val=10):
    return year - year%val

parser = argparse.ArgumentParser(description='Compute features from sparse dataset')

parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')
parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')
parser.add_argument('--tag', action='store_true',
                    help='Should the POS tag be kept?')


args = parser.parse_args()



test_df=pd.read_csv('/data/dharp/compounds/Compounding/data/all_compounds.txt',sep='\t')


copy_df=test_df.copy()
copy_df.modifier=copy_df.modifier.str.split('_').str[0]
copy_df['head']=copy_df['head'].str.split('_').str[0]

if args.tag:
    copy_df_1=copy_df.copy()
    copy_df_1.modifier=copy_df_1.modifier+'_NOUN'
    copy_df_1['head']=copy_df_1['head']+'_NOUN'

    copy_df_2=copy_df.copy()
    copy_df_2.modifier=copy_df_2.modifier+'_PROPN'
    copy_df_2['head']=copy_df_2['head']+'_NOUN'

    copy_df_3=copy_df.copy()
    copy_df_3.modifier=copy_df_3.modifier+'_NOUN'
    copy_df_3['head']=copy_df_3['head']+'_PROPN'

    copy_df_4=copy_df.copy()
    copy_df_4.modifier=copy_df_4.modifier+'_PROPN'
    copy_df_4['head']=copy_df_4['head']+'_PROPN'


    compound_df=pd.concat([copy_df_1,copy_df_2,copy_df_3,copy_df_4],ignore_index=True)

else:
    compound_df=copy_df.copy()
    
def process_compound(df):
     
    if temporal==0:
        print('No temporal information is stored')
        if cutoff!=0:
            df=df.loc[df.groupby(['modifier','head'],observed=True)['count'].transform('sum').gt(cutoff)]
        df=df.groupby(['modifier','head','context'],observed=True)['count'].sum().to_frame()

    else:
        print(f'Temporal information is stored with intervals {temporal}')
        df['time']=year_binner(df['year'].values,temporal)
        if cutoff!=0:
            df=df.loc[df.groupby(['modifier','head','time'],observed=True)['count'].transform('sum').gt(cutoff)]
        df=df.groupby(['modifier','head','time','context'],observed=True)['count'].sum().to_frame()
    
    df.reset_index(inplace=True)
    

    return df


def process_modifier(df):
            
    if temporal==0:
        df=df.groupby(['modifier','context'],observed=True)['count'].sum().to_frame()
        
    else:
        df['time']=year_binner(df['year'].values,temporal)
        df=df.groupby(['modifier','time','context'],observed=True)['count'].sum().to_frame()
        
    df.reset_index(inplace=True)
    
    return df


def process_head(df):
    
    if temporal==0:
        df=df.groupby(['head','context'],observed=True)['count'].sum().to_frame()
    
    else:
        df['time']=year_binner(df['year'].values,temporal)
        df=df.groupby(['head','time','context'],observed=True)['count'].sum().to_frame()
    
    df.reset_index(inplace=True)
    
    return df


def calculate_productivity():
    
    if temporal==0:
        all_comps=compounds[['modifier','head']]
        all_comps.drop_duplicates(inplace=True)
        mod_prod=all_comps.groupby(['modifier']).size().to_frame()
        mod_prod.columns=['mod_prod']
        mod_prod['N']=mod_prod['mod_prod'].sum()
        mod_prod['mod_family_size']=-np.log2((mod_prod.mod_prod+1)/(mod_prod.N-mod_prod.mod_prod+1))

        head_prod=all_comps.groupby(['head']).size().to_frame()
        head_prod.columns=['head_prod']
        head_prod['N']=head_prod['head_prod'].sum()
        head_prod['head_family_size']=-np.log2((head_prod.head_prod+1)/(head_prod.N-head_prod.head_prod+1))

        prod1=pd.merge(all_comps,mod_prod.reset_index(),how='left',on=['modifier'])
        productivity=pd.merge(prod1,head_prod.reset_index(),how='left',on=['head'])
        productivity.set_index(['modifier','head'],inplace=True)
        productivity.drop(['N_x','N_y'],axis=1,inplace=True)
    
    else:
        all_comps=compounds[['modifier','head','time']]
        all_comps.drop_duplicates(inplace=True)
        compound_counts=all_comps.groupby(['time']).size().to_frame()
        compound_counts.columns=['N']    

        mod_prod=all_comps.groupby(['modifier','time'],observed=True).size().to_frame()
        mod_prod.columns=['mod_prod']
        mod_prod=pd.merge(mod_prod.reset_index(),compound_counts.reset_index(),on=['time'],how='left')
        mod_prod['mod_family_size']=-np.log2((mod_prod.mod_prod+1)/(mod_prod.N-mod_prod.mod_prod+1))

        head_prod=all_comps.groupby(['head','time'],observed=True).size().to_frame()
        head_prod.columns=['head_prod']
        head_prod=pd.merge(head_prod.reset_index(),compound_counts.reset_index(),on=['time'],how='left')
        head_prod['head_family_size']=-np.log2((head_prod.head_prod+1)/(head_prod.N-head_prod.head_prod+1))

        prod1=pd.merge(all_comps,mod_prod,how='left',on=['modifier','time'])
        productivity=pd.merge(prod1,head_prod,how='left',on=['head','time'])
        productivity.set_index(['modifier','head','time'],inplace=True)
        productivity.drop(['N_x','N_y'],axis=1,inplace=True)
        
    return productivity
    
def calculate_information_theory():
    
    if temporal!=0:
        compound_time_counts=compounds.groupby('time').sum().sum(axis=1).to_frame()
        compound_time_counts.columns=['N']

        XY=compounds.groupby(['modifier','head','time'],observed=True).sum().sum(axis=1).to_frame()
        X_star=compounds.groupby(['modifier','time'],observed=True).sum().sum(axis=1).to_frame()
        Y_star=compounds.groupby(['head','time'],observed=True).sum().sum(axis=1).to_frame()

    else:
        XY=compounds.groupby(['modifier','head'],observed=True).sum().sum(axis=1).to_frame()
        X_star=compounds.groupby(['modifier']).sum().sum(axis=1).to_frame()
        Y_star=compounds.groupby(['head']).sum().sum(axis=1).to_frame()

    XY.columns=['a']

    X_star.columns=['x_star']
    Y_star.columns=['star_y']

    if temporal!=0:

        merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

        information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])
    else:
        merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier'])

        information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head'])    

    information_feat['b']=information_feat['x_star']-information_feat['a']
    information_feat['c']=information_feat['star_y']-information_feat['a']

    if temporal!=0:
        information_feat=pd.merge(information_feat,compound_time_counts.reset_index(),on=['time'])

    else: 
        information_feat['N']=compounds['count'].sum()


    information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
    information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
    information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']

    if temporal!=0:

        information_feat.set_index(['modifier','head','time'],inplace=True)
    else:
        information_feat.set_index(['modifier','head'],inplace=True)

    information_feat['log_ratio']=2*(information_feat['a']*np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))+\
    information_feat['b']*np.log2((information_feat['b']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y_bar']+1))+\
    information_feat['c']*np.log2((information_feat['c']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y']+1))+\
    information_feat['d']*np.log2((information_feat['d']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y_bar']+1)))
    information_feat['ppmi']=np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']))
    information_feat['local_mi']=information_feat['a']*information_feat['ppmi']
    information_feat.ppmi.loc[information_feat.ppmi<=0]=0
    information_feat.drop(['a','x_star','star_y','b','c','d','N','d','x_bar_star','star_y_bar'],axis=1,inplace=True)
    
    return information_feat,XY


def calculate_denom():

    if temporal==0:

        compound_denom=compounds.copy()
        compound_denom['count']=compound_denom['count']**2
        compound_denom=compound_denom.groupby(['modifier','head'],observed=True)['count'].sum().to_frame()
        compound_denom['count']=np.sqrt(compound_denom['count'])
        compound_denom.columns=['compound_denom']

        modifier_denom=modifiers.copy()
        modifier_denom['count']=modifier_denom['count']**2
        modifier_denom=modifier_denom.groupby(['modifier'],observed=True)['count'].sum().to_frame()
        modifier_denom['count']=np.sqrt(modifier_denom['count'])
        modifier_denom.columns=['modifier_denom']

        head_denom=heads.copy()
        head_denom['count']=head_denom['count']**2
        head_denom=head_denom.groupby(['head'],observed=True)['count'].sum().to_frame()
        head_denom['count']=np.sqrt(head_denom['count'])
        head_denom.columns=['head_denom']

    else:

        compound_denom=compounds.copy()
        compound_denom['count']=compound_denom['count']**2
        compound_denom=compound_denom.groupby(['modifier','head','time'],observed=True)['count'].sum().to_frame()
        compound_denom['count']=np.sqrt(compound_denom['count'])
        compound_denom.columns=['compound_denom']

        modifier_denom=modifiers.copy()
        modifier_denom['count']=modifier_denom['count']**2
        modifier_denom=modifier_denom.groupby(['modifier','time'],observed=True)['count'].sum().to_frame()
        modifier_denom['count']=np.sqrt(modifier_denom['count'])
        modifier_denom.columns=['modifier_denom']

        head_denom=heads.copy()
        head_denom['count']=head_denom['count']**2
        head_denom=head_denom.groupby(['head','time'],observed=True)['count'].sum().to_frame()
        head_denom['count']=np.sqrt(head_denom['count'])
        head_denom.columns=['head_denom']
        
    return compound_denom,modifier_denom,head_denom
    
    
    
def calculate_cos_head():
    
    if temporal==0:

        compound_head_sim=pd.merge(compounds,heads,on=["head","context"])
        compound_head_sim['numerator']=compound_head_sim['comp_count']*compound_head_sim['head_count']
        compound_head_sim=compound_head_sim.groupby(['modifier','head'],observed=True)['numerator'].sum().to_frame()
        compound_head_sim=pd.merge(compound_head_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head"])
        compound_head_sim=pd.merge(compound_head_sim,head_denom.reset_index(),on=['head'])
        compound_head_sim['sim_with_head']=compound_head_sim['numerator']/(compound_head_sim['compound_denom']*compound_head_sim['head_denom'])
        compound_head_sim.set_index(['modifier','head'],inplace=True)
        compound_head_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
    else:
        compound_head_sim=pd.merge(compounds,heads,on=["head","context",'time'])
        compound_head_sim['numerator']=compound_head_sim['comp_count']*compound_head_sim['head_count']
        compound_head_sim=compound_head_sim.groupby(['modifier','head','time'],observed=True)['numerator'].sum().to_frame()
        compound_head_sim=pd.merge(compound_head_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
        compound_head_sim=pd.merge(compound_head_sim,head_denom.reset_index(),on=['head','time'])
        compound_head_sim['sim_with_head']=compound_head_sim['numerator']/(compound_head_sim['compound_denom']*compound_head_sim['head_denom'])
        compound_head_sim.set_index(['modifier','head','time'],inplace=True)
        compound_head_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
        
    return compound_head_sim
    
    
def calculate_cos_mod():
    
    if temporal==0:

        compound_modifier_sim=pd.merge(compounds,modifiers,on=["modifier","context"])
        compound_modifier_sim['numerator']=compound_modifier_sim['comp_count']*compound_modifier_sim['mod_count']
        compound_modifier_sim=compound_modifier_sim.groupby(['modifier','head'],observed=True)['numerator'].sum().to_frame()
        compound_modifier_sim=pd.merge(compound_modifier_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head"])
        compound_modifier_sim=pd.merge(compound_modifier_sim,modifier_denom.reset_index(),on=['modifier'])
        compound_modifier_sim['sim_with_modifier']=compound_modifier_sim['numerator']/(compound_modifier_sim['compound_denom']*compound_modifier_sim['modifier_denom'])
        compound_modifier_sim.set_index(['modifier','head'],inplace=True)
        compound_modifier_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
    else:
        mod_cols=modifiers.columns.tolist()
        mod_cols[-1]="mod_count"
        modifiers.columns=mod_cols
        #compounds.drop(['comp_count'],axis=1,inplace=True)
        comp_cols=compounds.columns.tolist()
        comp_cols[-1]="comp_count"
        compounds.columns=comp_cols
        compound_modifier_sim=pd.merge(compounds,modifiers,on=["modifier","context",'time'])
        compound_modifier_sim['numerator']=compound_modifier_sim['comp_count']*compound_modifier_sim['mod_count']
        compound_modifier_sim=compound_modifier_sim.groupby(['modifier','head','time'],observed=True)['numerator'].sum().to_frame()
        compound_modifier_sim=pd.merge(compound_modifier_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
        compound_modifier_sim=pd.merge(compound_modifier_sim,modifier_denom.reset_index(),on=['modifier','time'])
        compound_modifier_sim['sim_with_modifier']=compound_modifier_sim['numerator']/(compound_modifier_sim['compound_denom']*compound_modifier_sim['modifier_denom'])
        compound_modifier_sim.set_index(['modifier','head','time'],inplace=True)
        compound_modifier_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
        
    return compound_modifier_sim

    
def calculate_cos_const():
    
    if temporal==0:

        constituent_sim=pd.merge(heads,compounds,on=["head","context"])
        #constituent_sim.drop('comp_count',axis=1,inplace=True)
        constituent_sim=pd.merge(constituent_sim,modifiers,on=["modifier","context"])
        constituent_sim['numerator']=constituent_sim['head_count']*constituent_sim['mod_count']
        constituent_sim=constituent_sim.groupby(['modifier','head'],observed=True)['numerator'].sum().to_frame()
        constituent_sim=pd.merge(constituent_sim.reset_index(),head_denom.reset_index(),on=["head"])
        constituent_sim=pd.merge(constituent_sim,modifier_denom.reset_index(),on=["modifier"])
        constituent_sim['sim_bw_constituents']=constituent_sim['numerator']/(constituent_sim['head_denom']*constituent_sim['modifier_denom'])
        constituent_sim.set_index(['modifier','head'],inplace=True)
        constituent_sim.drop(['numerator','modifier_denom','head_denom'],axis=1,inplace=True)
    else:
        
        constituent_sim=pd.merge(heads,compounds,on=["head","context","time"])
        #constituent_sim.drop('comp_count',axis=1,inplace=True)
        constituent_sim=pd.merge(constituent_sim,modifiers,on=["modifier","context","time"])
        constituent_sim['numerator']=constituent_sim['head_count']*constituent_sim['mod_count']
        constituent_sim=constituent_sim.groupby(['modifier','head','time'],observed=True)['numerator'].sum().to_frame()
        constituent_sim=pd.merge(constituent_sim.reset_index(),head_denom.reset_index(),on=["head","time"])
        constituent_sim=pd.merge(constituent_sim,modifier_denom.reset_index(),on=["modifier","time"])
        constituent_sim['sim_bw_constituents']=constituent_sim['numerator']/(constituent_sim['head_denom']*constituent_sim['modifier_denom'])
        constituent_sim.set_index(['modifier','head','time'],inplace=True)
        constituent_sim.drop(['numerator','modifier_denom','head_denom'],axis=1,inplace=True)
        
    return constituent_sim


    
    
cutoff_list=[0,10,20,50,100]
temporal_list=[0,10,20,50,100]

if args.contextual:
    
    
    print("CompoundAware Model")
    print("Loading the constituent and compound datasets")

    complete_compounds=pd.read_pickle(args.inputdir+"/compounds_1.pkl")
    
    complete_modifiers=pd.read_pickle(args.inputdir+"/modifiers_1.pkl")

    complete_heads=pd.read_pickle(args.inputdir+"/heads_1.pkl")
    
    
    if not args.tag:
        print('Removing tags')
        complete_compounds['head']=complete_compounds['head'].str.replace('_NOUN|_PROPN','',regex=True)
        complete_compounds.modifier=complete_compounds.modifier.str.replace('_NOUN|_PROPN','',regex=True)

        complete_modifiers.modifier=complete_modifiers.modifier.str.replace('_NOUN|_PROPN','',regex=True)

        complete_heads['head']=complete_heads['head'].str.replace('_NOUN|_PROPN','',regex=True)
    
    
    for cutoff in cutoff_list:
        for temporal in temporal_list:
            
            print(f'Cutoff: {cutoff}')
            print(f'Time span:  {temporal}')
            temp_cutoff_str=str(temporal)+'_'+str(cutoff)

            compounds=process_compound(complete_compounds)

            print('Done reading compounds')


            modifiers=process_modifier(complete_modifiers)

            print('Done reading modifiers')

            heads=process_head(complete_heads)

            print('Done reading heads')

            print('Calculating productivity features')

            productivity=calculate_productivity()

            print('Calculating information theory features')

            information_feat,XY=calculate_information_theory()
            XY.columns=['freq']

            print('Calculating denominator values')


            compound_denom,modifier_denom,head_denom=calculate_denom()

            mod_cols=modifiers.columns.tolist()
            mod_cols[-1]="mod_count"
            modifiers.columns=mod_cols

            head_cols=heads.columns.tolist()
            head_cols[-1]="head_count"
            heads.columns=head_cols

            comp_cols=compounds.columns.tolist()
            comp_cols[-1]="comp_count"
            compounds.columns=comp_cols

            print('Calculating cosine features')

            compound_modifier_sim=calculate_cos_mod()

            compound_head_sim=calculate_cos_head()

            constituent_sim=calculate_cos_const()


            print('Storing all features together')

            dfs = [XY,constituent_sim, compound_head_sim.sim_with_head, compound_modifier_sim.sim_with_modifier, information_feat,productivity]
            compounds_final = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), dfs)



            if temporal!=0:
                compounds_final=pd.pivot_table(compounds_final.reset_index(), index=['modifier','head'], columns=['time'])
                compounds_final_1=compounds_final.columns.get_level_values(0)
                compounds_final_2=compounds_final.columns.get_level_values(1)

                cur_year=0
                new_columns=[]
                for year in compounds_final_2:
                    new_columns.append(str(year)+"_"+compounds_final_1[cur_year])
                    cur_year+=1
                compounds_final.columns=new_columns


            else:
                compounds_final = reduce(lambda left,right: pd.merge(left,right,on=['modifier','head']), dfs)

            compounds_final.reset_index(inplace=True)

            print('Storing Reddy compounds')

            compounds_final=compounds_final.merge(compound_df,on=['modifier','head'])

            if args.contextual:
                comp_str='CompoundAware'
            else:
                comp_str='CompoundAgnostic'

            if args.tag:
                tag_str='Tagged'
            else:
                tag_str='UnTagged'

            print('Storing compounds')
            compounds_final.to_csv(f'{args.inputdir}/features_{comp_str}_{tag_str}_{str(temporal)}_{str(cutoff)}.csv',sep='\t',index=False)

else:
    pass
    
