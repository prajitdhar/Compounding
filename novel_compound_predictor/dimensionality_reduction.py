import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,TruncatedSVD,NMF
from sklearn.preprocessing import Normalizer
import argparse
import time
import numba


@numba.jit(nopython=True)
def year_binner(year,val=10):
    return year - year%val


parser = argparse.ArgumentParser(description='Perform dimentionality reduction of the count-based vectors, using SVD')

parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')

parser.add_argument('--temporal', action='store_true',
                    help='Is the model temporal')

parser.add_argument('--seed', type=int, default=1991,
                    help='random seed')

parser.add_argument('--dims', type=int, default=300,
                    help='Desired number of reduced dimensions')

parser.add_argument('--save_format', type=str,default='pkl',
                    help='In what format should the reduced datasets be saved : csv or pkl')

args = parser.parse_args()
modifier_list=pkl.load( open("modifier_list_reduced.pkl",'rb'))
head_list=pkl.load( open("head_list_reduced.pkl",'rb'))
t1=time.time()
# Dimentionality Reduction using SVD

def dim_reduction(df,rows):
    df_svd = TruncatedSVD(n_components=args.dims, n_iter=10, random_state=args.seed)
    print(f'Explained variance ratio {(df_svd.fit(df).explained_variance_ratio_.sum()):2.3f}')
    #df_list=df_svd.fit(df).explained_variance_ratio_
    df_reduced = df_svd.fit_transform(df)
    df_reduced = Normalizer(copy=False).fit_transform(df_reduced)
    df_reduced=pd.DataFrame(df_reduced,index=rows)
    #df_reduced.reset_index(inplace=True)
    if args.temporal:
        df_reduced.index = pd.MultiIndex.from_tuples(df_reduced.index, names=['common', 'decade'])
    return df_reduced

def common_reduction(df):
    df.reset_index(inplace=True)

    df.year=df.year.astype("int32")
    df=df.query('1800 <= year <= 2010').copy()
    df['time']=year_binner(df['year'].values,10)

    df=df.groupby(['modifier','head','context','time'])['count'].sum().to_frame()
    df.reset_index(inplace=True)
    df=df.loc[df.groupby(['modifier','head','time'])['count'].transform('sum').gt(50)]
    df=df.loc[df['modifier'].isin(modifier_list) & df['head'].isin(head_list)]
    return df

if args.contextual:
    
    print("CompoundCentric Model")
    comp_str='CompoundCentric'
    print("Loading the constituent and compound vector datasets")
    
    heads=pd.read_csv("/data/dharp/compounding/datasets/heads_reduced.csv",sep="\t")
    #heads=heads.query('decade != 2000')
    heads.columns=['common','decade','context','count']
    heads['common']=heads['common'].str.replace(r'_n$', r'_h', regex=True)
    
    modifiers=pd.read_csv("/data/dharp/compounding/datasets/modifiers_reduced.csv",sep="\t")
    #modifiers=modifiers.query('decade != 2000')
    modifiers.columns=['common','decade','context','count']
    modifiers['common']=modifiers['common'].str.replace(r'_n$', r'_m', regex=True)
    
    compounds=pd.read_pickle("/data/dharp/compounding/datasets/compounds.pkl")
    compounds=common_reduction(compounds)
    compounds['common']=compounds['modifier']+" "+compounds['head']



    
    if args.temporal:
        print("DecadeCentric Model")
        compounds=compounds.groupby(['common','decade','context'])['count'].sum()
        modifiers=modifiers.groupby(['common','decade','context'])['count'].sum()
        heads=heads.groupby(['common','decade','context'])['count'].sum()
        
    else:
        print("DecadeAgnostic Model")
        compounds=compounds.groupby(['common','context'])['count'].sum()
        modifiers=modifiers.groupby(['common','context'])['count'].sum()
        heads=heads.groupby(['common','context'])['count'].sum()
    
    print('Concatenating all the datasets together')
    df=pd.concat([heads,modifiers,compounds])
        
else:
    print("CompoundAgnostic Model")
    comp_str='CompoundAgnostic'
    print("Loading the word and phrase vector datasets")
    
    constituents=pd.read_csv("/data/dharp/compounding/datasets/words.csv")
    constituents.columns=['common','context','decade','count']
    #constituents=constituents.query('decade != 2000')
    
    
    compounds=pd.read_csv("/data/dharp/compounding/datasets/phrases.csv")
    compounds.columns=['modifier','head','context','decade','count']
    #compounds=compounds.query('decade != 2000')
    compounds['common']=compounds['modifier']+" "+compounds['head']
    
    if args.temporal:
        print("DecadeCentric Model")
        compounds=compounds.groupby(['common','decade','context'])['count'].sum()
        constituents=constituents.groupby(['common','decade','context'])['count'].sum()
        
    else:
        print("DecadeAgnostic Model")
        compounds=compounds.groupby(['common','context'])['count'].sum()
        constituents=constituents.groupby(['common','context'])['count'].sum()
    
    print('Concatenating all the datasets together')
    df=pd.concat([constituents,compounds])

    
df=df.to_sparse()
    
if args.temporal:    
    df, rows, _ = df.to_coo(row_levels=['common','decade'],column_levels=['context'],sort_labels=False)
    dec_str='DecadeCentric'

else:
    df, rows, _ = df.to_coo(row_levels=['common'],column_levels=['context'],sort_labels=False)
    dec_str='DecadeAgnostic'

print('Running SVD')   
df_reduced=dim_reduction(df,rows)
#df_reduced.reset_index(inplace=True)


print('Splitting back into individual datasets are saving them')

if args.temporal:
    df_reduced.index.names = ['common','decade']
else:
    df_reduced.index.names = ['common']
    
compounds_reduced=df_reduced.loc[df_reduced.index.get_level_values(0).str.contains(r'\w \w')]
compounds_reduced.reset_index(inplace=True)
#print(compounds_reduced.head())
compounds_reduced['modifier'],compounds_reduced['head']=compounds_reduced['common'].str.split(' ', 1).str

dim_str=str(args.dims)
if args.contextual:

    
    heads_reduced=df_reduced.loc[df_reduced.index.get_level_values(0).str.endswith(r'_h')]
    heads_reduced.reset_index(inplace=True)
    
    heads_reduced['head']=heads_reduced['common'].str.replace(r'_h$', r'_n', regex=True)
    heads_reduced.drop(['common'],axis=1,inplace=True)
    
    
    modifiers_reduced=df_reduced.loc[df_reduced.index.get_level_values(0).str.endswith(r'_m')]
    modifiers_reduced.reset_index(inplace=True)
    
    modifiers_reduced['modifier']=modifiers_reduced['common'].str.replace(r'_m$', r'_n', regex=True)
    modifiers_reduced.drop(['common'],axis=1,inplace=True)    
    
    if args.temporal:
        compounds_reduced.set_index(['modifier','head','decade'],inplace=True)
        heads_reduced.set_index(['head','decade'],inplace=True)
        modifiers_reduced.set_index(['modifier','decade'],inplace=True)
    else:
        compounds_reduced.set_index(['modifier','head'],inplace=True)
        heads_reduced.set_index(['head'],inplace=True)
        modifiers_reduced.set_index(['modifier'],inplace=True)
    
    print('Saving the files')
    if args.save_format=='pkl':
        compounds_reduced.to_pickle('/data/dharp/compounding/datasets/compounds_'+comp_str+'_'+dec_str+'_'+dim_str+'.pkl')
        heads_reduced.to_pickle('/data/dharp/compounding/datasets/heads_'+comp_str+'_'+dec_str+'_'+dim_str+'.pkl')
        modifiers_reduced.to_pickle('/data/dharp/compounding/datasets/modifiers_'+comp_str+'_'+dec_str+'_'+dim_str+'.pkl')
    elif args.save_format=='csv':
        compounds_reduced.to_csv("/data/dharp/compounding/datasets/"+'compounds_'+comp_str+'_'+dec_str+'_'+dim_str+'.csv',header=False,sep='\t')
        heads_reduced.to_csv('/data/dharp/compounding/datasets/heads_'+comp_str+'_'+dec_str+'_'+dim_str+'.csv',header=False,sep='\t')
        modifiers_reduced.to_pickle('/data/dharp/compounding/datasets/modifiers_'+comp_str+'_'+dec_str+'_'+dim_str+'.csv',header=False,sep='\t')
        
else:
    
    constituents_reduced=df_reduced.loc[~df_reduced.index.get_level_values(0).str.contains(r'\w \w')]
    
    if args.temporal:
        compounds_reduced.set_index(['modifier','head','decade'],inplace=True)

    else:
        compounds_reduced.set_index(['modifier','head'],inplace=True)

    print('Saving the files')
    
    if args.save_format=='pkl':
        compounds_reduced.to_pickle('/data/dharp/compounding/datasets/compounds_'+comp_str+'_'+dec_str+'_'+dim_str+'.pkl')
        constituents_reduced.to_pickle('/data/dharp/compounding/datasets/constituents_'+comp_str+'_'+dec_str+'_'+dim_str+'.pkl')
        
    elif args.save_format=='csv':
        
        compounds_reduced.to_csv("/data/dharp/compounding/datasets/"+'compounds_'+comp_str+'_'+dec_str+'_'+dim_str+'.csv',header=False,sep='\t')
        constituents_reduced.to_csv('/data/dharp/compounding/datasets/constituents_'+comp_str+'_'+dec_str+'_'+dim_str+'.csv',header=False,sep='\t')

print(f'Time taken {time.time()-t1:10.3f}')