import pandas as pd
import numpy as np
from sklearn.decomposition import PCA,TruncatedSVD,NMF
from sklearn.preprocessing import Normalizer
import argparse
import time
#import dask
import pickle as pkl
#import dask.dataframe as dd
import numba
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = None
from functools import reduce
import matplotlib
#matplotlib.use('agg')
#matplotlib.style.use('ggplot')
from matplotlib import pyplot as plt
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

br_to_us=pd.read_excel("Book.xlsx",skiprows=[0])
br_to_us_dict=dict(zip(br_to_us.UK.tolist(),br_to_us.US.tolist()))
spelling_replacement={'modifier':br_to_us_dict,'head':br_to_us_dict}

def lemma_maker(x, y):
    #print(lemmatizer(x,y)[0])
    return lemmatizer(x,y)[0]

parser = argparse.ArgumentParser(description='Compute features from embeddings')

parser.add_argument('--temporal',  type=int,
                    help='Value to bin the temporal information: 0 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')

parser.add_argument('--cutoff', type=int, default=50,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')


args = parser.parse_args()

print(f'Cutoff: {args.cutoff}')
print(f'Time span:  {args.temporal}')

heads=pd.read_pickle("../../datasets/heads_CompoundAware_"+str(args.temporal)+"_"+str(args.cutoff)+"_300.pkl")
#heads.reset_index(inplace=True)
#heads=heads.drop(['decade'],axis=1).groupby(['head']).mean()
#heads=heads+1
if args.temporal!=0:
    heads.index.set_names('time', level=1,inplace=True)


modifiers=pd.read_pickle("../../datasets/modifiers_CompoundAware_"+str(args.temporal)+"_"+str(args.cutoff)+"_300.pkl")
#heads.reset_index(inplace=True)
#heads=heads.drop(['decade'],axis=1).groupby(['head']).mean()
#modifiers=modifiers+1
if args.temporal!=0:
    modifiers.index.set_names('time', level=1,inplace=True)


compounds=pd.read_pickle("../../datasets/compounds_CompoundAware_"+str(args.temporal)+"_"+str(args.cutoff)+"_300.pkl")
#heads.reset_index(inplace=True)
#heads=heads.drop(['decade'],axis=1).groupby(['head']).mean()
if args.temporal!=0:
    compounds.index.set_names('time', level=2,inplace=True)
compounds.drop(['common'],axis=1,inplace=True)
compounds=compounds+1

if args.temporal!=0:
    
    compound_decade_counts=compounds.groupby('time').sum().sum(axis=1).to_frame()
    compound_decade_counts.columns=['N']

#compounds = dd.from_pandas(compounds, npartitions=30)

    XY=compounds.groupby(['modifier','head','time']).sum().sum(axis=1).to_frame()
    X_star=compounds.groupby(['modifier','time']).sum().sum(axis=1).to_frame()
    Y_star=compounds.groupby(['head','time']).sum().sum(axis=1).to_frame()


else:
    XY=compounds.groupby(['modifier','head']).sum().sum(axis=1).to_frame()
    X_star=compounds.groupby(['modifier']).sum().sum(axis=1).to_frame()
    Y_star=compounds.groupby(['head']).sum().sum(axis=1).to_frame()


    
    
#XY=XY.compute()
XY.columns=['a']

#X_star=X_star.compute()
X_star.columns=['x_star']
#Y_star=Y_star.compute()
Y_star.columns=['star_y']


if args.temporal!=0:
 
    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])
else:
    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head'])    

#information_feat=dd.from_pandas(information_feat, npartitions=30)
information_feat['b']=information_feat['x_star']-information_feat['a']
information_feat['c']=information_feat['star_y']-information_feat['a']

if args.temporal!=0:
    information_feat=pd.merge(information_feat,compound_decade_counts.reset_index(),on=['time'])



#information_feat=information_feat.compute()
else:
 
    information_feat['N']=compounds.reset_index().drop(['modifier','head'],axis=1).sum().sum()
    

#information_feat=dd.from_pandas(information_feat, npartitions=30)
information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']
#information_feat['LR']=-2*np.sum(information_feat['a']*np.log2((information_feat['a']*information_feat['N'])/(information_feat['x_star']*information_feat['star_y'])))

#information_feat=information_feat.compute()
if args.temporal!=0:

    information_feat.set_index(['modifier','head','time'],inplace=True)
else:
    information_feat.set_index(['modifier','head'],inplace=True)


information_feat.replace(0,0.0001,inplace=True)
information_feat['log_ratio']=2*(information_feat['a']*np.log((information_feat['a']*information_feat['N'])/(information_feat['x_star']*information_feat['star_y']))+\
information_feat['b']*np.log((information_feat['b']*information_feat['N'])/(information_feat['x_star']*information_feat['star_y_bar']))+\
information_feat['c']*np.log((information_feat['c']*information_feat['N'])/(information_feat['x_bar_star']*information_feat['star_y']))+\
information_feat['d']*np.log((information_feat['d']*information_feat['N'])/(information_feat['x_bar_star']*information_feat['star_y_bar'])))
information_feat['ppmi']=np.log2((information_feat['a']*information_feat['N'])/(information_feat['x_star']*information_feat['star_y']))
information_feat['local_mi']=information_feat['a']*information_feat['ppmi']
information_feat.ppmi.loc[information_feat.ppmi<=0]=0
information_feat.drop(['a','x_star','star_y','b','c','d','N','d','x_bar_star','star_y_bar'],axis=1,inplace=True)
#information_feat.info()
#information_feat.head()


#modifier_denom=np.square(modifiers).sum(axis=1)**0.5
#modifier_denom=modifier_denom.to_frame()
#modifier_denom.columns=['modifier_denom']


#head_denom=np.square(heads).sum(axis=1)**0.5
#head_denom=head_denom.to_frame()
#head_denom.columns=['head_denom']


compounds=compounds-1
#compound_denom=np.square(compounds).sum(axis=1)**0.5
#compound_denom=compound_denom.to_frame()
#compound_denom.columns=['compound_denom']

compound_modifier_sim=compounds.multiply(modifiers.reindex(compounds.index,level=0).ffill()).sum(axis=1).to_frame()
#compound_modifier_sim=compounds.multiply(modifiers.reindex(compounds.index, method='ffill')).sum(axis=1).to_frame()
compound_modifier_sim.columns=['sim_with_modifier']


compound_head_sim=compounds.multiply(heads.reindex(compounds.index,level=1).ffill()).sum(axis=1).to_frame()
compound_head_sim.columns=['sim_with_head']

if args.temporal!=0:
    constituent_sim=compounds.reset_index()[['modifier','head','time']].merge(modifiers.reset_index(),how='left',on=['modifier','time'])
    constituent_sim.set_index(['modifier','head','time'],inplace=True)
else:
    constituent_sim=compounds.reset_index()[['modifier','head']].merge(modifiers.reset_index(),how='left',on=['modifier'])
    constituent_sim.set_index(['modifier','head'],inplace=True)


constituent_sim=constituent_sim.multiply(heads.reindex(constituent_sim.index, level=1).ffill()).sum(axis=1).to_frame()
constituent_sim.columns=['sim_bw_constituents']

dfs = [constituent_sim, compound_head_sim, compound_modifier_sim, information_feat]
compounds_final = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), dfs)
if args.temporal!=0:
    compounds_final=pd.pivot_table(compounds_final.reset_index(), index=['modifier','head'], columns=['time'])

    compounds_final.fillna(0,inplace=True)
    compounds_final -= compounds_final.min()
    compounds_final /= compounds_final.max()
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
    #compounds_final.set_index(['modifier','head'],inplace=True)
    compounds_final.fillna(0,inplace=True)
    compounds_final -= compounds_final.min()
    compounds_final /= compounds_final.max()

reddy11_study=pd.read_csv("../../datasets/ijcnlp_compositionality_data/MeanAndDeviations.clean.txt",sep="\t")
#print(reddy11_study.columns)
reddy11_study.columns=['compound','to_divide']
reddy11_study['modifier_mean'],reddy11_study['modifier_std'],reddy11_study['head_mean'],reddy11_study['head_std'],reddy11_study['compound_mean'],reddy11_study['compound_std'],_=reddy11_study.to_divide.str.split(" ",7).str
reddy11_study['modifier'],reddy11_study['head']=reddy11_study['compound'].str.split(" ",2).str
reddy11_study.modifier=reddy11_study.modifier.str[:-2]
reddy11_study['head']=reddy11_study['head'].str[:-2]
reddy11_study.drop(['compound','to_divide'],axis=1,inplace=True)
reddy11_study['modifier']=np.vectorize(lemma_maker)(reddy11_study['modifier'],'noun')
reddy11_study['head']=np.vectorize(lemma_maker)(reddy11_study['head'],'noun')
reddy11_study.replace(spelling_replacement,inplace=True)
reddy11_study['modifier']=reddy11_study['modifier']+"_noun"
reddy11_study['head']=reddy11_study['head']+"_noun"
reddy11_study=reddy11_study.apply(pd.to_numeric, errors='ignore')



merge_df=reddy11_study.merge(compounds_final.reset_index(),on=['modifier','head'],how='inner')
merge_df.set_index(["modifier", "head"], inplace = True)

merge_df.to_csv("../../datasets/features_CompoundAware_"+str(args.temporal)+"_"+str(args.cutoff)+"_300.csv",sep='\t')
