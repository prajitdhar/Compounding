import pandas as pd
import numpy as np

import argparse
import pickle

from functools import reduce


def year_binner(year,val=10):
    return year - year%val


parser = argparse.ArgumentParser(description='Compute features from sparse dataset')

parser.add_argument('--temporal',  type=int,default=0,
                    help='Value to bin the temporal information: 0 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')

parser.add_argument('--cutoff', type=int, default=50,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')

parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')
#parser.add_argument('--sparse', action='store_true',
#                    help='Is the model sparse')
parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')



args = parser.parse_args()

print(f'Cutoff: {args.cutoff}')
print(f'Time span:  {args.temporal}')
temp_cutoff_str=str(args.temporal)+'_'+str(args.cutoff)

if args.contextual:
    context='CompoundAware'
else:
    context='CompoundAgnostic'
    
save_path=context+'_Sparse_'+temp_cutoff_str


context_list = pickle.load( open( f'{args.inputdir}context.pkl', "rb" ))



if args.contextual:
    print("CompoundCentric Model")

    print('Reading compounds')
    compounds=pd.read_pickle(args.inputdir+"/compounds.pkl")
    print(compounds.shape[0])
    compounds.context=compounds.context.str.replace(r'.+_NUM','NUM',regex=True)
    compounds=compounds.loc[compounds.context.isin(context_list)]
    print(compounds.shape[0])
    
    compounds.modifier=compounds.modifier.str.replace(r'_.+','',regex=True)
    compounds['head']=compounds['head'].str.replace(r'_.+','',regex=True)

    if args.temporal==0:
        print('No temporal information is stored')

        #compounds=compounds.loc[~compounds.modifier.str.contains('^(?:of|the|-)_.+')]
        #compounds=compounds.loc[~compounds['head'].str.contains('^(?:of|the|-)_.+')]

        compounds=compounds.loc[compounds.groupby(['modifier','head'])['count'].transform('sum').gt(args.cutoff)]
        print(compounds.shape[0])

        compounds=compounds.groupby(['modifier','head','context'])['count'].sum().to_frame().reset_index()

    else:
        print(f'Temporal information is stored with intervals {args.temporal}')
        compounds.year=compounds.year.astype("int32")
        #compounds.query('1800 <= year <= 2010',inplace=True)
        compounds['time']=year_binner(compounds['year'].values,args.temporal)
        compounds=compounds.loc[compounds.groupby(['modifier','head','time'])['count'].transform('sum').gt(args.cutoff)]
        print(compounds.shape[0])

        compounds=compounds.groupby(['modifier','head','time','context'])['count'].sum().to_frame().reset_index()
    
    print(compounds.shape[0])    
    print('Done reading compounds')
    
    modifier_lst=compounds.modifier.unique().tolist()
    print(f'Number of unique modifiers {len(modifier_lst)}')

    head_lst=compounds['head'].unique().tolist()
    len(head_lst)    
    print(f'Number of unique heads {len(head_lst)}')

    print('Reading modifiers')

    modifiers=pd.read_pickle(args.inputdir+"/modifiers.pkl")
    print(modifiers.shape[0])
    modifiers.context=modifiers.context.str.replace(r'.+_NUM','NUM',regex=True)
    modifiers=modifiers.loc[modifiers.context.isin(context_list)]
    print(modifiers.shape[0])
    modifiers.modifier=modifiers.modifier.str.replace(r'_.+','',regex=True)

    if args.temporal==0:
        modifiers=modifiers.groupby(['modifier','context'])['count'].sum().to_frame().reset_index()
    else:
        modifiers.year=modifiers.year.astype("int32")
        #modifiers.query('1800 <= year <= 2010',inplace=True)        
        modifiers['time']=year_binner(modifiers['year'].values,args.temporal)
        modifiers=modifiers.groupby(['modifier','time','context'])['count'].sum().to_frame().reset_index()

    print(modifiers.shape[0])
    
    modifiers=modifiers.loc[modifiers.modifier.isin(modifier_lst)]
    print(modifiers.shape[0])

    print('Done reading modifiers')        

    print('Reading heads')

    heads=pd.read_pickle(args.inputdir+"/heads.pkl")
    print(heads.shape[0])
    heads.context=heads.context.str.replace(r'.+_NUM','NUM',regex=True)
    heads=heads.loc[heads.context.isin(context_list)]
    print(heads.shape[0])
    heads['head']=heads['head'].str.replace(r'_.+','',regex=True)
    
    if args.temporal==0:
        heads=heads.groupby(['head','context'])['count'].sum().to_frame().reset_index()
    else:
        heads.year=heads.year.astype("int32")
        #heads.query('1800 <= year <= 2010',inplace=True)
        heads['time']=year_binner(heads['year'].values,args.temporal)
        heads=heads.groupby(['head','time','context'])['count'].sum().to_frame().reset_index()
        
    print(heads.shape[0])
    
    heads=heads.loc[heads['head'].isin(modifier_lst)]
    print(heads.shape[0])
    print('Done reading heads') 

else:
    print("CompoundAgnostic Model")
    
    print('Reading phrases')
    compounds=pd.read_pickle(args.inputdir+"/phrases.pkl")
    print(compounds.shape[0])
    compounds.context=compounds.context.str.replace(r'.+_NUM','NUM',regex=True)
    compounds=compounds.loc[compounds.context.isin(context_list)]
    print(compounds.shape[0])
    
    compounds.modifier=compounds.modifier.str.replace(r'_.+','',regex=True)
    compounds['head']=compounds['head'].str.replace(r'_.+','',regex=True)

    if args.temporal==0:
        print('No temporal information is stored')

        #compounds=compounds.loc[~compounds.modifier.str.contains('^(?:of|the|-)_.+')]
        #compounds=compounds.loc[~compounds['head'].str.contains('^(?:of|the|-)_.+')]

        compounds=compounds.loc[compounds.groupby(['modifier','head'])['count'].transform('sum').gt(args.cutoff)]
        print(compounds.shape[0])

        compounds=compounds.groupby(['modifier','head','context'])['count'].sum().to_frame().reset_index()

    else:
        print(f'Temporal information is stored with intervals {args.temporal}')
        compounds.year=compounds.year.astype("int32")
        #compounds.query('1800 <= year <= 2010',inplace=True)
        compounds['time']=year_binner(compounds['year'].values,args.temporal)
        compounds=compounds.loc[compounds.groupby(['modifier','head','time'])['count'].transform('sum').gt(args.cutoff)]
        print(compounds.shape[0])

        compounds=compounds.groupby(['modifier','head','time','context'])['count'].sum().to_frame().reset_index()

        
    print(compounds.shape[0])    
    print('Done reading compounds')
    
    constituents_lst=list(set(compounds.modifier.unique().tolist()+compounds['head'].unique().tolist()))
    print(f'Number of unique constituents {len(constituents_lst)}')
    
    print('Reading constituents')
    constituents=pd.read_pickle(args.inputdir+"/words.pkl")
    print(constituents.shape[0])
    constituents.context=constituents.context.str.replace(r'.+_NUM','NUM',regex=True)
    constituents=constituents.loc[constituents.context.isin(context_list)]
    print(constituents.shape[0])
    constituents.word=constituents.word.str.replace(r'_.+','',regex=True)
    constituents=constituents.loc[constituents.word.isin(constituents_lst)]
    print(constituents.shape[0])
    
    if args.temporal==0:
        constituents=constituents.groupby(['word','context'])['count'].sum().to_frame().reset_index()
        
        modifiers=constituents.copy()
        modifiers.columns=['modifier','context','count']
        heads=constituents.copy()
        heads.columns=['head','context','count']
    else:
        constituents.year=constituents.year.astype("int32")
        #constituents.query('1800 <= year <= 2010',inplace=True)
        constituents['time']=year_binner(constituents['year'].values,args.temporal)
        constituents=constituents.groupby(['word','time','context'])['count'].sum().to_frame().reset_index()
        
        modifiers=constituents.copy()
        modifiers.columns=['modifier','time','context','count']
        heads=constituents.copy()
        heads.columns=['head','time','context','count']
    print(constituents.shape[0])
    print('Done reading constituents')
    
    
print("Calculating Productivity features")    
    
if args.temporal==0:
    all_comps=compounds[['modifier','head']].copy()
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

    all_comps=compounds[['modifier','head','time']].copy()
    all_comps.drop_duplicates(inplace=True)
    compound_counts=all_comps.groupby(['time']).size().to_frame()
    compound_counts.columns=['N']    
    
    mod_prod=all_comps.groupby(['modifier','time']).size().to_frame()
    mod_prod.columns=['mod_prod']
    mod_prod=pd.merge(mod_prod.reset_index(),compound_counts.reset_index(),on=['time'],how='left')
    mod_prod['mod_family_size']=-np.log2((mod_prod.mod_prod+1)/(mod_prod.N-mod_prod.mod_prod+1))
    
    
    head_prod=all_comps.groupby(['head','time']).size().to_frame()
    head_prod.columns=['head_prod']
    head_prod=pd.merge(head_prod.reset_index(),compound_counts.reset_index(),on=['time'],how='left')
    head_prod['head_family_size']=-np.log2((head_prod.head_prod+1)/(head_prod.N-head_prod.head_prod+1))
    
    prod1=pd.merge(all_comps,mod_prod,how='left',on=['modifier','time'])
    productivity=pd.merge(prod1,head_prod,how='left',on=['head','time'])
    productivity.set_index(['modifier','head','time'],inplace=True)
    productivity.drop(['N_x','N_y'],axis=1,inplace=True)
    
    
    
    
print("Calculating Information Theory features")    

if args.temporal!=0:
    
    compound_decade_counts=compounds.groupby('time').sum().sum(axis=1).to_frame()
    compound_decade_counts.columns=['N']

    XY=compounds.groupby(['modifier','head','time']).sum().sum(axis=1).to_frame()
    X_star=compounds.groupby(['modifier','time']).sum().sum(axis=1).to_frame()
    Y_star=compounds.groupby(['head','time']).sum().sum(axis=1).to_frame()


else:
    XY=compounds.groupby(['modifier','head']).sum().sum(axis=1).to_frame()
    X_star=compounds.groupby(['modifier']).sum().sum(axis=1).to_frame()
    Y_star=compounds.groupby(['head']).sum().sum(axis=1).to_frame()

XY.columns=['a']

X_star.columns=['x_star']
Y_star.columns=['star_y']

if args.temporal!=0:
 
    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])
else:
    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head'])    

information_feat['b']=information_feat['x_star']-information_feat['a']
information_feat['c']=information_feat['star_y']-information_feat['a']

if args.temporal!=0:
    information_feat=pd.merge(information_feat,compound_decade_counts.reset_index(),on=['time'])

else: 
    information_feat['N']=compounds['count'].sum()
    

information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']

if args.temporal!=0:

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


print("Calculating Frequency features")    


if args.temporal!=0:
 
    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    frequency=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])
    frequency.set_index(['modifier','head','time'],inplace=True)
else:
    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier'])

    frequency=pd.merge(merge1,Y_star.reset_index(),on=['head'])
    frequency.set_index(['modifier','head'],inplace=True)

frequency.columns=['comp_freq','mod_freq','head_freq']


print("Calculating Cosine features")    


if args.temporal==0:

    compound_denom=compounds.groupby(['modifier','head'])['count'].agg(lambda x: np.sqrt(np.sum(np.square(x)))).to_frame()
    compound_denom.columns=['compound_denom']
    
    modifier_denom=modifiers.groupby(['modifier'])['count'].agg(lambda x: np.sqrt(np.sum(np.square(x)))).to_frame()
    modifier_denom.columns=['modifier_denom']

    head_denom=heads.groupby(['head'])['count'].agg(lambda x: np.sqrt(np.sum(np.square(x)))).to_frame()
    head_denom.columns=['head_denom']
else:

    compound_denom=compounds.groupby(['modifier','head','time'])['count'].agg(lambda x: np.sqrt(np.sum(np.square(x)))).to_frame()
    compound_denom.columns=['compound_denom']

    modifier_denom=modifiers.groupby(['modifier','time'])['count'].agg(lambda x: np.sqrt(np.sum(np.square(x)))).to_frame()
    modifier_denom.columns=['modifier_denom']
    
    head_denom=heads.groupby(['head','time'])['count'].agg(lambda x: np.sqrt(np.sum(np.square(x)))).to_frame()
    head_denom.columns=['head_denom']

mod_cols=modifiers.columns.tolist()
mod_cols[-1]="mod_count"
modifiers.columns=mod_cols



head_cols=heads.columns.tolist()
head_cols[-1]="head_count"
heads.columns=head_cols

#compounds.drop(['comp_count'],axis=1,inplace=True)
comp_cols=compounds.columns.tolist()
comp_cols[-1]="comp_count"
compounds.columns=comp_cols



if args.temporal==0:
    
    compound_modifier_sim=pd.merge(compounds,modifiers,on=["modifier","context"])
    compound_modifier_sim['numerator']=compound_modifier_sim['comp_count']*compound_modifier_sim['mod_count']
    compound_modifier_sim=compound_modifier_sim.groupby(['modifier','head'])['numerator'].sum().to_frame()
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
    compound_modifier_sim=compound_modifier_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    compound_modifier_sim=pd.merge(compound_modifier_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_modifier_sim=pd.merge(compound_modifier_sim,modifier_denom.reset_index(),on=['modifier','time'])
    compound_modifier_sim['sim_with_modifier']=compound_modifier_sim['numerator']/(compound_modifier_sim['compound_denom']*compound_modifier_sim['modifier_denom'])
    compound_modifier_sim.set_index(['modifier','head','time'],inplace=True)
    compound_modifier_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
    
    
    
    
if args.temporal==0:
    
    compound_head_sim=pd.merge(compounds,heads,on=["head","context"])
    compound_head_sim['numerator']=compound_head_sim['comp_count']*compound_head_sim['head_count']
    compound_head_sim=compound_head_sim.groupby(['modifier','head'])['numerator'].sum().to_frame()
    compound_head_sim=pd.merge(compound_head_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head"])
    compound_head_sim=pd.merge(compound_head_sim,head_denom.reset_index(),on=['head'])
    compound_head_sim['sim_with_head']=compound_head_sim['numerator']/(compound_head_sim['compound_denom']*compound_head_sim['head_denom'])
    compound_head_sim.set_index(['modifier','head'],inplace=True)
    compound_head_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
else:
    compound_head_sim=pd.merge(compounds,heads,on=["head","context",'time'])
    compound_head_sim['numerator']=compound_head_sim['comp_count']*compound_head_sim['head_count']
    compound_head_sim=compound_head_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    compound_head_sim=pd.merge(compound_head_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_head_sim=pd.merge(compound_head_sim,head_denom.reset_index(),on=['head','time'])
    compound_head_sim['sim_with_head']=compound_head_sim['numerator']/(compound_head_sim['compound_denom']*compound_head_sim['head_denom'])
    compound_head_sim.set_index(['modifier','head','time'],inplace=True)
    compound_head_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)
    
    
    
if args.temporal==0:
    
    constituent_sim=pd.merge(heads,compounds,on=["head","context"])
    #constituent_sim.drop('comp_count',axis=1,inplace=True)
    constituent_sim=pd.merge(constituent_sim,modifiers,on=["modifier","context"])
    constituent_sim['numerator']=constituent_sim['head_count']*constituent_sim['mod_count']
    constituent_sim=constituent_sim.groupby(['modifier','head'])['numerator'].sum().to_frame()
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
    constituent_sim=constituent_sim.groupby(['modifier','head','time'])['numerator'].sum().to_frame()
    constituent_sim=pd.merge(constituent_sim.reset_index(),head_denom.reset_index(),on=["head","time"])
    constituent_sim=pd.merge(constituent_sim,modifier_denom.reset_index(),on=["modifier","time"])
    constituent_sim['sim_bw_constituents']=constituent_sim['numerator']/(constituent_sim['head_denom']*constituent_sim['modifier_denom'])
    constituent_sim.set_index(['modifier','head','time'],inplace=True)
    constituent_sim.drop(['numerator','modifier_denom','head_denom'],axis=1,inplace=True)
    
    
print("Combining all features")

dfs = [constituent_sim, compound_head_sim, compound_modifier_sim, information_feat,frequency,productivity]
compounds_final = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), dfs)



if args.temporal!=0:
    compounds_final=pd.pivot_table(compounds_final.reset_index(), index=['modifier','head'], columns=['time'])

    compounds_final.fillna(0,inplace=True)
    #compounds_final -= compounds_final.min()
    #compounds_final /= compounds_final.max()
    compounds_final_1=compounds_final.columns.get_level_values(0)
    compounds_final_2=compounds_final.columns.get_level_values(1)

    cur_year=0
    new_columns=[]
    for year in compounds_final_2:
        new_columns.append(str(year)+"_"+compounds_final_1[cur_year])
        cur_year+=1
    compounds_final.columns=new_columns


else:
    #compounds_final = reduce(lambda left,right: pd.merge(left,right,on=['modifier','head']), dfs)
    #compounds_final.drop(['head_denom','modifier_denom'],axis=1,inplace=True)
    compounds_final.fillna(0,inplace=True)
    #compounds_final -= compounds_final.min()
    #compounds_final /= compounds_final.max()
    
    
all_compounds=pd.read_csv(f"/home/users0/pageljs/dh/repos/Compounding_github/data/all_compounds.txt",sep="\t")
all_compounds.modifier=all_compounds.modifier.str.split('_',n=1,expand=True)[[0]]
all_compounds['head']=all_compounds['head'].str.split('_',n=1,expand=True)[[0]]
all_compounds=all_compounds.apply(pd.to_numeric, errors='ignore')


merge_210_df=all_compounds.merge(compounds_final.reset_index(),on=['modifier','head'],how='inner')
merge_210_df.set_index(["modifier", "head"], inplace = True)

print(f'Number of compounds found {merge_210_df.shape[0]} out of {all_compounds.shape[0]}')


merge_210_df.to_csv(f'{args.outputdir}/{save_path}.csv')

