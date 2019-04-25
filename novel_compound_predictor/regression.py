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


parser = argparse.ArgumentParser(description='Gather data necessary for performing Regression')

parser.add_argument('--temporal',  type=int,
                    help='Value to bin the temporal information: 0 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')

parser.add_argument('--contextual', action='store_true',
                    help='Is the model contextual')

parser.add_argument('--cutoff', type=int, default=50,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')

parser.add_argument('--embeddings', type=str, default='dense',
                    help='Type of embeddings : sparse or dense')

parser.add_argument('--seed', type=int, default=1991,
                    help='random seed')

parser.add_argument('--storedf', action='store_true',
                    help='Should the embeddings be saved')

#parser.add_argument('--dims', type=int, default=300,
                    #help='Desired number of reduced dimensions')

parser.add_argument('--save_format', type=str,default='pkl',
                    help='In what format should the reduced datasets be saved : csv or pkl')


args = parser.parse_args()


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
    compounds.reset_index(inplace=True)
    compounds.year=compounds.year.astype("int32")
    compounds=compounds.query('1800 <= year <= 2010').copy()
    if args.temporal==0:
        print('No temporal information is stored')
        compounds=compounds.groupby(['modifier','head','context'])['count'].sum().to_frame()
    else:
        compounds['time']=year_binner(compounds['year'].values,args.temporal)
    #compounds = dd.from_pandas(compounds, npartitions=100)
        compounds=compounds.groupby(['modifier','head','context','time'])['count'].sum().to_frame()
    compounds.reset_index(inplace=True)
    if args.temporal==0:
        compounds=compounds.loc[compounds.groupby(['modifier','head'])['count'].transform('sum').gt(args.cutoff)]
    else:
        compounds=compounds.loc[compounds.groupby(['modifier','head','time'])['count'].transform('sum').gt(args.cutoff)]

    head_list_reduced=compounds['head'].unique().tolist()
    modifier_list_reduced=compounds['modifier'].unique().tolist()
        
    modifiers=pd.read_pickle("/data/dharp/compounding/datasets/modifiers.pkl")
    modifiers.reset_index(inplace=True)
    modifiers.year=modifiers.year.astype("int32")
    modifiers=modifiers.query('1800 <= year <= 2010').copy()
    modifiers['time']=np.vectorize(year_binner)(modifiers['year'],10)
    modifiers=modifiers.groupby(['modifier','context','time'])['count'].sum().to_frame()
    modifiers.reset_index(inplace=True)