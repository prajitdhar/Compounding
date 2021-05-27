import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,TruncatedSVD,NMF
from sklearn.preprocessing import Normalizer
import argparse
import time
import pickle as pkl


#@numba.jit(nopython=True)
def year_binner(year,val=10):
    return year - year%val

def dim_reduction(df,rows):
    df_svd = TruncatedSVD(n_components=300, n_iter=10, random_state=args.seed)
    print(f'Explained variance ratio {(df_svd.fit(df).explained_variance_ratio_.sum()):2.3f}')
    #df_list=df_svd.fit(df).explained_variance_ratio_
    df_reduced = df_svd.fit_transform(df)
    df_reduced = Normalizer(copy=False).fit_transform(df_reduced)
    df_reduced=pd.DataFrame(df_reduced,index=rows)
    #df_reduced.reset_index(inplace=True)
    if args.temporal!=0:
        df_reduced.index = pd.MultiIndex.from_tuples(df_reduced.index, names=['common', 'time'])
    return df_reduced


parser = argparse.ArgumentParser(description='Performing dimentionality reduction to get dense embeddings')

parser.add_argument('--inputdir',type=str,
                    help='Provide directory that has the files with the fivegram counts')
parser.add_argument('--outputdir',type=str,
                    help='Provide directory in that the output files should be stored')
parser.add_argument('--temporal',  type=int, default=0,
                    help='Value to bin the temporal information: 0 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')

parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')

parser.add_argument('--cutoff', type=int, default=50,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')


parser.add_argument('--seed', type=int, default=1991,
                    help='random seed')

parser.add_argument('--storedf', action='store_true',
                    help='Should the embeddings be saved')

parser.add_argument('--dims', type=int, default=300,
                    help='Desired number of reduced dimensions')

parser.add_argument('--input_format',type=str,default='csv',choices=['csv','pkl'],
                    help='In what format are the input files : csv or pkl')
parser.add_argument('--save_format', type=str,default='pkl',choices=['pkl','csv'],
                    help='In what format should the reduced datasets be saved : csv or pkl')


args = parser.parse_args()

print(f'Cutoff: {args.cutoff}')
print(f'Time span:  {args.temporal}')
print(f'Dimensionality: {args.dims}')

                    
                    
print("Creating dense embeddings")
if args.contextual:
    print("CompoundCentric Model")
    print("Loading the constituent and compound vector datasets")

    if args.input_format=="csv":
        compounds=pd.read_csv(args.inputdir+"/compounds.csv",sep="\t")
    elif args.input=="pkl":
        compounds=pd.read_pickle(args.inputdir+"/compounds.pkl")
        compounds.reset_index(inplace=True)
    compounds.year=compounds.year.astype("int32")
    compounds=compounds.query('1800 <= year <= 2010').copy()
    compounds['common']=compounds['modifier']+" "+compounds['head']

        #head_list_reduced=compounds['head'].unique().tolist()
        #modifier_list_reduced=compounds['modifier'].unique().tolist()

    if args.temporal==0:
        print('No temporal information is stored')
        compounds=compounds.groupby(['common','context'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)
        compounds=compounds.loc[compounds.groupby(['common'])['count'].transform('sum').gt(args.cutoff)]
        compounds=compounds.groupby(['common','context'])['count'].sum()

    else:
        compounds['time']=year_binner(compounds['year'].values,args.temporal)
        compounds=compounds.groupby(['common','context','time'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)
        compounds=compounds.loc[compounds.groupby(['common','time'])['count'].transform('sum').gt(args.cutoff)]
        compounds=compounds.groupby(['common','time','context'])['count'].sum()




    if args.input_format=="csv":
        modifiers=pd.read_csv(args.inputdir+"/modifiers.csv",sep="\t")
    elif args.input=="pkl":
        modifiers=pd.read_pickle(args.inputdir+"/modifiers.pkl")
        modifiers.reset_index(inplace=True)
    modifiers.year=modifiers.year.astype("int32")
    modifiers=modifiers.query('1800 <= year <= 2010').copy()
    modifiers.columns=['common','context','year','count']
    modifiers['common']=modifiers['common'].str.replace(r'_noun$', r'_m', regex=True)
        
    if args.temporal==0:
        print('No temporal information is stored')
        modifiers=modifiers.groupby(['common','context'])['count'].sum().to_frame()
        modifiers.reset_index(inplace=True)
        modifiers=modifiers.loc[modifiers.groupby(['common'])['count'].transform('sum').gt(args.cutoff)]
        modifiers=modifiers.groupby(['common','context'])['count'].sum()
    else:
        modifiers['time']=year_binner(modifiers['year'].values,args.temporal)
        modifiers=modifiers.groupby(['common','context','time'])['count'].sum().to_frame()
        modifiers=modifiers.loc[modifiers.groupby(['common','time'])['count'].transform('sum').gt(args.cutoff)]
        modifiers=modifiers.groupby(['common','time','context'])['count'].sum()

    if args.input_format=="csv":
        heads=pd.read_csv(args.inputdir+"/heads.csv",sep="\t")
    elif args.input_format=="pkl":
        heads=pd.read_pickle(args.inputdir+"/heads.pkl")
        heads.reset_index(inplace=True)
    heads.year=heads.year.astype("int32")
    heads=heads.query('1800 <= year <= 2010').copy()
    heads.columns=['common','context','year','count']
    heads['common']=heads['common'].str.replace(r'_noun$', r'_h', regex=True)
    if args.temporal==0:
        print('No temporal information is stored')
        heads=heads.groupby(['common','context'])['count'].sum().to_frame()
        heads.reset_index(inplace=True)
        heads=heads.loc[heads.groupby(['common'])['count'].transform('sum').gt(args.cutoff)]
        heads=heads.groupby(['common','context'])['count'].sum()
    else:
        heads['time']=year_binner(heads['year'].values,args.temporal)
        heads=heads.groupby(['common','context','time'])['count'].sum().to_frame()
        heads=heads.loc[heads.groupby(['common','time'])['count'].transform('sum').gt(args.cutoff)]
        heads=heads.groupby(['common','time','context'])['count'].sum()

    print('Concatenating all the datasets together')
    df=pd.concat([heads,modifiers,compounds], sort=True)

else:
    print("CompoundAgnostic Model")
    wordlist = pkl.load( open( "../data/coha_wordlist.pkl", "rb" ) )
    
    if args.input_format=="csv":
        compounds=pd.read_csv(args.inputdir+"/phrases.csv",sep="\t")
    elif args.input_format=="pkl":
        compounds=pd.read_pickle(args.inputdir+"/phrases.pkl")
        compounds.reset_index(inplace=True)
    compounds.year=compounds.year.astype("int32")
    compounds=compounds.query('1800 <= year <= 2010').copy()
    compounds['common']=compounds['modifier']+" "+compounds['head']


    if args.temporal==0:
        print('No temporal information is stored')
        compounds=compounds.groupby(['common','context'])['count'].sum().to_frame()
        compounds.reset_index(inplace=True)
        compounds=compounds.loc[compounds.groupby(['common'])['count'].transform('sum').gt(args.cutoff)]
        compounds=compounds.groupby(['common','context'])['count'].sum()
    else:
        compounds['time']=year_binner(compounds['year'].values,args.temporal)
            #compounds = dd.from_pandas(compounds, npartitions=100)
        compounds=compounds.groupby(['common','context','time'])['count'].sum().to_frame()
        compounds=compounds.loc[compounds.groupby(['common','time'])['count'].transform('sum').gt(args.cutoff)]
        compounds=compounds.groupby(['common','time','context'])['count'].sum()
        
    if args.input_format=="csv":
        constituents=pd.read_csv(args.outputdir+"/words.csv",sep="\t")
    elif args.input_format=="pkl":
        constituents=pd.read_pickle(args.outputdir+"/words.pkl")
        constituents.reset_index(inplace=True)
    constituents.year=constituents.year.astype("int32")
    constituents=constituents.query('1800 <= year <= 2010').copy()
    constituents.columns=['common','context','year','count']
    constituents.query('common in @wordlist',inplace=True)
    if args.temporal==0:
        print('No temporal information is stored')
        constituents=constituents.groupby(['common','context'])['count'].sum().to_frame()
        constituents.reset_index(inplace=True)
        constituents=constituents.loc[constituents.groupby(['common'])['count'].transform('sum').gt(args.cutoff)]
        constituents=constituents.groupby(['common','context'])['count'].sum()           
    else:
        constituents['time']=year_binner(constituents['year'].values,args.temporal)
        constituents=constituents.groupby(['common','context','time'])['count'].sum().to_frame()
        constituents.reset_index(inplace=True)
        constituents=constituents.loc[constituents.groupby(['common','time'])['count'].transform('sum').gt(args.cutoff)]
        constituents=constituents.groupby(['common','time','context'])['count'].sum()

    print('Concatenating all the datasets together')
    df=pd.concat([constituents,compounds], sort=True)
        

dtype = pd.SparseDtype(np.float, fill_value=0)
df=df.astype(dtype)
if args.temporal!=0:    
    df, rows, _ = df.sparse.to_coo(row_levels=['common','time'],column_levels=['context'],sort_labels=False)

else:
    df, rows, _ = df.sparse.to_coo(row_levels=['common'],column_levels=['context'],sort_labels=False)

print('Running SVD')   
df_reduced=dim_reduction(df,rows)

print('Splitting back into individual datasets are saving them')
if args.temporal!=0:
    df_reduced.index.names = ['common','time']
else:
    df_reduced.index.names = ['common']

compounds_reduced=df_reduced.loc[df_reduced.index.get_level_values(0).str.contains(r'\w \w')]
compounds_reduced.reset_index(inplace=True)
    #print(compounds_reduced.head())
compounds_reduced[['modifier','head']]=compounds_reduced['common'].str.split(' ', n=1,expand=True).copy()

if args.contextual:
    heads_reduced=df_reduced.loc[df_reduced.index.get_level_values(0).str.endswith(r'_h')]
    heads_reduced.reset_index(inplace=True)
    heads_reduced['head']=heads_reduced['common'].str.replace(r'_h$', r'_noun', regex=True)
    heads_reduced.drop(['common'],axis=1,inplace=True)

    modifiers_reduced=df_reduced.loc[df_reduced.index.get_level_values(0).str.endswith(r'_m')]
    modifiers_reduced.reset_index(inplace=True)   
    modifiers_reduced['modifier']=modifiers_reduced['common'].str.replace(r'_m$', r'_noun', regex=True)
    modifiers_reduced.drop(['common'],axis=1,inplace=True)

    if args.temporal!=0:
        compounds_reduced.set_index(['modifier','head','time'],inplace=True)
        heads_reduced.set_index(['head','time'],inplace=True)
        modifiers_reduced.set_index(['modifier','time'],inplace=True)
    else:
        compounds_reduced.set_index(['modifier','head'],inplace=True)
        heads_reduced.set_index(['head'],inplace=True)
        modifiers_reduced.set_index(['modifier'],inplace=True)

    if args.storedf:
        print('Saving the files')
        comp_str="CompoundAware"
        if args.save_format=='pkl':
            compounds_reduced.to_pickle(args.outputdir+'/compounds_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.pkl')
            heads_reduced.to_pickle(args.outputdir+'/heads_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.pkl')
            modifiers_reduced.to_pickle(args.outputdir+'/modifiers_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.pkl')
        elif args.save_format=='csv':
            compounds_reduced.to_csv(args.outputdir+'/compounds_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.csv',header=False,sep='\t')
            heads_reduced.to_csv(args.outputdir+'/heads_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.csv',header=False,sep='\t')
            modifiers_reduced.to_pickle(args.outputdir+'/modifiers_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.csv',header=False,sep='\t')
    else:
        print('Files are not being saved')

else:
    constituents_reduced=df_reduced.loc[~df_reduced.index.get_level_values(0).str.contains(r'\w \w')]
    constituents_reduced.reset_index(inplace=True)
    constituents_reduced['constituent']=constituents_reduced['common']
    constituents_reduced.drop(['common'],axis=1,inplace=True)
    
    if args.temporal!=0:
        compounds_reduced.set_index(['modifier','head','time'],inplace=True)
        constituents_reduced.set_index(['constituent','time'],inplace=True)
    else:
        compounds_reduced.set_index(['modifier','head'],inplace=True)
        constituents_reduced.set_index(['constituent'],inplace=True)
            
    if args.storedf:
        print('Saving the files')
        comp_str="CompoundAgnostic"
        if args.save_format=='pkl':
            compounds_reduced.to_pickle(args.outputdir+'/compounds_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.pkl')
            constituents_reduced.to_pickle(args.outputdir+'/constituents_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.pkl')

        elif args.save_format=='csv':
            compounds_reduced.to_csv(args.outputdir+'/compounds_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.csv',header=False,sep='\t')
            constituents_reduced.to_csv(args.outputdir+'/constituents_'+comp_str+'_'+str(args.temporal)+'_'+str(args.cutoff)+'_'+'300'+'.csv',header=False,sep='\t')
    else:
        print('Files are not being saved')
