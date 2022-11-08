import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import Normalizer
import argparse
import time
import pickle
import re
from functools import reduce
import pickle as pkl

def year_binner(year,val=10):
    if val==0:
        return 0
    else:
        return year - year%val
    
    
def dim_reduction(df):
    
    dtype = pd.SparseDtype(np.float64, fill_value=0)
    df=df.astype(dtype)

    df_sparse, rows, cols = df.sparse.to_coo(row_levels=['common','time'],column_levels=['context'],sort_labels=False)

    print(len(cols))
    rcomp = re.compile(".+\s.+")
    compound_rows=[]
    compound_time=[]
    constituent_rows=[]
    constituent_time=[]

    for r in rows:
        if re.match(rcomp, r[0]):
            compound_rows.append(r[0])
            compound_time.append(r[1])
        else:
            constituent_rows.append(r[0])
            constituent_time.append(r[1])        

    assert (len(compound_rows)+len(constituent_rows))==df_sparse.shape[0]
    train_df=df_sparse.tocsr()[0:len(compound_rows),:]
    test_df=df_sparse.tocsr()[len(compound_rows):,:]
    assert (train_df.shape[0]+test_df.shape[0])==df_sparse.shape[0]

    svd = TruncatedSVD(n_components=300, algorithm='arpack', random_state=args.seed)
    print(f'Explained variance ratio {(svd.fit(train_df).explained_variance_ratio_.sum()):2.3f}')
    
    compound_reduced = svd.fit_transform(train_df)
    compound_reduced = Normalizer(copy=False).fit_transform(compound_reduced)

    compound_reduced=pd.DataFrame(compound_reduced,index=list(zip(compound_rows,compound_time)))
    compound_reduced.index = pd.MultiIndex.from_tuples(compound_reduced.index, names=['compound', 'time'])

    compound_reduced.reset_index(inplace=True)
    compound_reduced[['modifier','head']]=compound_reduced['compound'].str.split(' ',expand=True)
    compound_reduced.drop(['compound'],axis=1,inplace=True)
    compound_reduced.set_index(['modifier','head','time'],inplace=True)
    #compound_reduced.reset_index(inplace=True)
    
    constituents_reduced=svd.transform(test_df)
    constituents_reduced = Normalizer(copy=False).fit_transform(constituents_reduced)
    constituents_reduced=pd.DataFrame(constituents_reduced,index=list(zip(constituent_rows,constituent_time)))
    constituents_reduced.index = pd.MultiIndex.from_tuples(constituents_reduced.index, names=['constituent', 'time'])
    constituents_reduced.reset_index(inplace=True)
    
    return compound_reduced,constituents_reduced



def productivity_features(df):
    
    print("Productivity")
     
    all_comps=df.reset_index()[['modifier','head','time']]
    mod_prod=df.groupby(['modifier','time']).size().to_frame()
    mod_prod.columns=['mod_prod']
    head_prod=df.groupby(['head','time']).size().to_frame()
    head_prod.columns=['head_prod']
    prod1=pd.merge(all_comps,mod_prod.reset_index(),how='left',on=['modifier','time'])
    productivity=pd.merge(prod1,head_prod.reset_index(),how='left',on=['head','time'])
    productivity.set_index(['modifier','head','time'],inplace=True)
    
    return productivity
    

def freq_features(df):
    
    print("Frequency features")
        
    compound_decade_counts=df.groupby('time').sum().sum(axis=1).to_frame()
    compound_decade_counts.columns=['N']

    XY=df.groupby(['modifier','head','time']).sum().sum(axis=1).to_frame()
    X_star=df.groupby(['modifier','time']).sum().sum(axis=1).to_frame()
    Y_star=df.groupby(['head','time']).sum().sum(axis=1).to_frame()

    XY.columns=['a']
    X_star.columns=['x_star']
    Y_star.columns=['star_y']


    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    frequency_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])
    
    frequency_feat=frequency_feat.rename(columns = {'a':'comp_freq','x_star':'mod_freq','star_y':'head_freq'})
    frequency_feat.set_index(['modifier','head','time'],inplace=True)

    return frequency_feat

def it_features(df):
    
    print("Information Theory features")
    
    compound_decade_counts=df.groupby('time').sum().sum(axis=1).to_frame()
    compound_decade_counts.columns=['N']

    XY=df.groupby(['modifier','head','time']).sum().sum(axis=1).to_frame()
    X_star=df.groupby(['modifier','time']).sum().sum(axis=1).to_frame()
    Y_star=df.groupby(['head','time']).sum().sum(axis=1).to_frame()


    XY.columns=['a']
    X_star.columns=['x_star']
    Y_star.columns=['star_y']


    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])
    
   
    information_feat['b']=information_feat['x_star']-information_feat['a']
    information_feat['c']=information_feat['star_y']-information_feat['a']

    information_feat=pd.merge(information_feat,compound_decade_counts.reset_index(),on=['time'])
    information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
    information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
    information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']

    information_feat.set_index(['modifier','head','time'],inplace=True)

    information_feat['ppmi']=np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))
    information_feat['local_mi']=information_feat['a']*information_feat['ppmi']
    information_feat['log_ratio']=2*(information_feat['local_mi']+\
    information_feat['b']*np.log2((information_feat['b']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y_bar']+1))+\
    information_feat['c']*np.log2((information_feat['c']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y']+1))+\
    information_feat['d']*np.log2((information_feat['d']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y_bar']+1)))

    information_feat.ppmi.loc[information_feat.ppmi<=0]=0
    information_feat.drop(['a','x_star','star_y','b','c','d','N','d','x_bar_star','star_y_bar'],axis=1,inplace=True)
    
    return information_feat



def cosine_features(compound_df,modifier_df,head_df):
    
    print("Cosine Similarity features")

    
    compound_modifier_sim=(compound_df*modifier_df).dropna().sum(axis=1).to_frame()
    compound_modifier_sim.columns=['sim_with_modifier']
    compound_modifier_sim=compound_modifier_sim.swaplevel('time','head')


    compound_head_sim=(compound_df*head_df).dropna().sum(axis=1).to_frame()
    compound_head_sim.columns=['sim_with_head']
    compound_head_sim=compound_head_sim.swaplevel('time','modifier')
    compound_head_sim=compound_head_sim.swaplevel('head','modifier')


    constituent_sim=compounds_reduced.reset_index()[['modifier','head','time']].merge(modifiers_reduced.reset_index(),how='left',on=['modifier','time'])
    constituent_sim.set_index(['modifier','head','time'],inplace=True)


    constituent_sim=(constituent_sim*heads_reduced).dropna().sum(axis=1).to_frame()
    constituent_sim.columns=['sim_bw_constituents']
    constituent_sim=constituent_sim.swaplevel('time','modifier')
    constituent_sim=constituent_sim.swaplevel('head','modifier')
    
    return compound_modifier_sim,compound_head_sim,constituent_sim


parser = argparse.ArgumentParser(description='Compute features from sparse dataset via SVD')

parser.add_argument('--temporal',  type=int,default=0,
                    help='Value to bin the temporal information: 0 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')

parser.add_argument('--cutoff', type=int, default=50,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')
parser.add_argument('--seed', type=int, default=1991,
                    help='random seed')
parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')
parser.add_argument('--inputdir',type=str,
                    help='Provide directory where features are located')
parser.add_argument('--outputdir',type=str,
                    help='Where should the output be stored?')



args = parser.parse_args()

print(f'Cutoff: {args.cutoff}')
print(f'Time span:  {args.temporal}')
temp_cutoff_str=str(args.temporal)+'_'+str(args.cutoff)


context_list = pickle.load( open( f'{args.inputdir}context.pkl', "rb" ) )

if args.contextual:
    context='CompoundAware'
else:
    context='CompoundAgnostic'

save_path=context+'_Dense_'+temp_cutoff_str


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
    else:
        print(f'Temporal information is stored with intervals {args.temporal}')

    #compounds=compounds.loc[~compounds.modifier.str.contains('^(?:of|the|-)_.+')]
    #compounds=compounds.loc[~compounds['head'].str.contains('^(?:of|the|-)_.+')]
    
    compounds.year=compounds.year.astype("int32")
    #compounds.query('1800 <= year <= 2010',inplace=True)
    compounds['time']=year_binner(compounds['year'].values,args.temporal)
    compounds=compounds.loc[compounds.groupby(['modifier','head','time'])['count'].transform('sum').gt(args.cutoff)]
    print(compounds.shape[0])

    compounds=compounds.groupby(['modifier','head','time','context'])['count'].sum().to_frame().reset_index()
    print(compounds.shape[0])    
    
    modifier_lst=compounds.modifier.unique().tolist()
    print(f'Number of unique modifiers {len(modifier_lst)}')

    head_lst=compounds['head'].unique().tolist()
    len(head_lst)    
    print(f'Number of unique heads {len(head_lst)}')

    compounds['common']=compounds['modifier']+" "+compounds['head']


    compounds=compounds.groupby(['common','time','context'])['count'].sum()
        
    print('Done reading compounds')

    print('Reading modifiers')

    modifiers=pd.read_pickle(args.inputdir+"/modifiers.pkl")
    print(modifiers.shape[0])
    modifiers.context=modifiers.context.str.replace(r'.+_NUM','NUM',regex=True)
    modifiers=modifiers.loc[modifiers.context.isin(context_list)]
    print(modifiers.shape[0])
    modifiers.modifier=modifiers.modifier.str.replace(r'_.+','',regex=True)


    modifiers.year=modifiers.year.astype("int32")
    #modifiers.query('1800 <= year <= 2010',inplace=True)        
    modifiers['time']=year_binner(modifiers['year'].values,args.temporal)
    modifiers=modifiers.groupby(['modifier','time','context'])['count'].sum().to_frame().reset_index()
    modifiers.columns=['common','time','context','count']
    
    print(modifiers.shape[0])
    
    modifiers=modifiers.loc[modifiers.common.isin(modifier_lst)]
    print(modifiers.shape[0])

    modifiers.common=modifiers.common+"_m"

    modifiers=modifiers.groupby(['common','time','context'])['count'].sum()

    print('Done reading modifiers')        

    print('Reading heads')

    heads=pd.read_pickle(args.inputdir+"/heads.pkl")
    print(heads.shape[0])
    heads.context=heads.context.str.replace(r'.+_NUM','NUM',regex=True)
    heads=heads.loc[heads.context.isin(context_list)]
    print(heads.shape[0])
    heads['head']=heads['head'].str.replace(r'_.+','',regex=True)
    
    
    heads.year=heads.year.astype("int32")
    #heads.query('1800 <= year <= 2010',inplace=True)
    heads['time']=year_binner(heads['year'].values,args.temporal)
    heads=heads.groupby(['head','time','context'])['count'].sum().to_frame().reset_index()
    heads.columns=['common','time','context','count']
    print(heads.shape[0])
    
    heads=heads.loc[heads.common.isin(modifier_lst)]
    print(heads.shape[0])
    
    heads.common=heads.common+"_h"
    
    heads=heads.groupby(['common','time','context'])['count'].sum()


    print('Done reading heads')

    print('Concatenating all the datasets together')
    
    
    df=pd.concat([compounds,heads,modifiers], sort=False)

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
    else:
        print(f'Temporal information is stored with intervals {args.temporal}')

    #compounds=compounds.loc[~compounds.modifier.str.contains('^(?:of|the|-)_.+')]
    #compounds=compounds.loc[~compounds['head'].str.contains('^(?:of|the|-)_.+')]

    compounds.year=compounds.year.astype("int32")
    #compounds.query('1800 <= year <= 2010',inplace=True)
    compounds['time']=year_binner(compounds['year'].values,args.temporal)
    compounds=compounds.loc[compounds.groupby(['modifier','head','time'])['count'].transform('sum').gt(args.cutoff)]
    print(compounds.shape[0])

    compounds=compounds.groupby(['modifier','head','time','context'])['count'].sum().to_frame().reset_index()
    constituents_lst=list(set(compounds.modifier.unique().tolist()+compounds['head'].unique().tolist()))

    compounds['common']=compounds['modifier']+" "+compounds['head']
    compounds=compounds.groupby(['common','time','context'])['count'].sum()

    print('Done reading compounds')
    
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
    
    constituents.year=constituents.year.astype("int32")
    #constituents.query('1800 <= year <= 2010',inplace=True)
    constituents['time']=year_binner(constituents['year'].values,args.temporal)
    constituents=constituents.groupby(['word','time','context'])['count'].sum().to_frame().reset_index()
    constituents.columns=['common','time','context','count']
    constituents=constituents.groupby(['common','time','context'])['count'].sum()

    print(constituents.shape[0])
    print('Done reading constituents')
    
    print('Concatenating all the datasets together')
    
    df=pd.concat([compounds,constituents], sort=False)
    
    
    
    
time_lst=compounds.index.unique(level='time').to_list()


