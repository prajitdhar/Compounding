import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
pd.options.display.float_format = '{:,.3f}'.format
import argparse
pd.options.mode.chained_assignment = None

import pickle as pkl

from scipy.stats.stats import pearsonr

br_to_us=pd.read_excel("../data/Book.xlsx",skiprows=[0])
br_to_us_dict=dict(zip(br_to_us.UK.tolist(),br_to_us.US.tolist()))
spelling_replacement={'modifier':br_to_us_dict,'head':br_to_us_dict}

def lemma_maker(x, y):
    #print(lemmatizer.lemmatize(x,y))
    return lemmatizer.lemmatize(x,y)
from functools import reduce

parser = argparse.ArgumentParser(description='Compute features from embeddings')

parser.add_argument('--temporal',  type=int,
                    help='Value to bin the temporal information: 0 (remove temporal information), 1 (no binning), 10 (binning to decades), 20 (binning each 20 years) or 50 (binning each 50 years)')

parser.add_argument('--cutoff', type=int, default=50,
                    help='Cut-off frequency for each compound per time period : none (0), 20, 50 and 100')

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


if args.contextual:
    comp_df_path=args.inputdir+'/compounds_CompoundAware_'+temp_cutoff_str+'_300.pkl'
    mod_df_path=args.inputdir+'/modifiers_CompoundAware_'+temp_cutoff_str+'_300.pkl'
    head_df_path=args.inputdir+'/heads_CompoundAware_'+temp_cutoff_str+'_300.pkl'
    features_df_path=args.outputdir+'/features_CompoundAware_'+temp_cutoff_str+'_300.pkl'
else:
    comp_df_path=args.inputdir+'/compounds_CompoundAgnostic_'+temp_cutoff_str+'_300.pkl'
    mod_df_path=args.inputdir+'/constituents_CompoundAgnostic_'+temp_cutoff_str+'_300.pkl'
    head_df_path=args.inputdir+'/constituents_CompoundAgnostic_'+temp_cutoff_str+'_300.pkl'
    features_df_path=args.outputdir+'/features_CompoundAgnostic_'+temp_cutoff_str+'_300.pkl'



heads=pd.read_pickle(head_df_path)

if args.temporal!=0:
    heads.index.set_names('time', level=1,inplace=True)
    heads.index.set_names('head',level=0,inplace=True)

else:
    heads.index.set_names('head',inplace=True)

modifiers=pd.read_pickle(mod_df_path)

if args.temporal!=0:
    modifiers.index.set_names('time', level=1,inplace=True)
    modifiers.index.set_names('modifier',level=0,inplace=True)
else:
    modifiers.index.set_names('modifier',inplace=True)


compounds=pd.read_pickle(comp_df_path)

if args.temporal!=0:
    compounds.index.set_names('time', level=2,inplace=True)
compounds.drop(['common'],axis=1,inplace=True)
compounds=compounds+1


####Productivity


if args.temporal!=0:
    all_comps=compounds.reset_index()[['modifier','head','time']]
    mod_prod=compounds.groupby(['modifier','time']).size().to_frame()
    mod_prod.columns=['mod_prod']
    head_prod=compounds.groupby(['head','time']).size().to_frame()
    head_prod.columns=['head_prod']
    prod1=pd.merge(all_comps,mod_prod.reset_index(),how='left',on=['modifier','time'])
    productivity=pd.merge(prod1,head_prod.reset_index(),how='left',on=['head','time'])
    productivity.set_index(['modifier','head','time'],inplace=True)
else:
    all_comps=compounds.reset_index()[['modifier','head']]
    mod_prod=compounds.groupby(['modifier']).size().to_frame()
    mod_prod.columns=['mod_prod']
    head_prod=compounds.groupby(['head']).size().to_frame()
    head_prod.columns=['head_prod']
    prod1=pd.merge(all_comps,mod_prod.reset_index(),how='left',on=['modifier'])
    productivity=pd.merge(prod1,head_prod.reset_index(),how='left',on=['head'])
    productivity.set_index(['modifier','head'],inplace=True)  


####Information Theory

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
 
    information_feat['N']=compounds.reset_index().drop(['modifier','head'],axis=1).sum().sum()
    

information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']

if args.temporal!=0:

    information_feat.set_index(['modifier','head','time'],inplace=True)
else:
    information_feat.set_index(['modifier','head'],inplace=True)


#information_feat.replace(0,0.0001,inplace=True)
information_feat['ppmi']=np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))
information_feat['local_mi']=information_feat['a']*information_feat['ppmi']
information_feat['log_ratio']=2*(information_feat['local_mi']+\
information_feat['b']*np.log2((information_feat['b']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y_bar']+1))+\
information_feat['c']*np.log2((information_feat['c']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y']+1))+\
information_feat['d']*np.log2((information_feat['d']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y_bar']+1)))


information_feat.ppmi.loc[information_feat.ppmi<=0]=0
information_feat.drop(['a','x_star','star_y','b','c','d','N','d','x_bar_star','star_y_bar'],axis=1,inplace=True)


###Cosine Features

new_compounds=compounds-1


compound_modifier_sim=new_compounds.multiply(modifiers.reindex(new_compounds.unstack('head').index)).sum(axis=1).to_frame()
compound_modifier_sim.columns=['sim_with_modifier']


compound_head_sim=new_compounds.multiply(heads.reindex(new_compounds.unstack('modifier').index)).sum(axis=1).to_frame()
compound_head_sim.columns=['sim_with_head']


if args.temporal!=0:
    constituent_sim=new_compounds.reset_index()[['modifier','head','time']].merge(modifiers.reset_index(),how='left',on=['modifier','time'])
    constituent_sim.set_index(['modifier','head','time'],inplace=True)
else:
    constituent_sim=new_compounds.reset_index()[['modifier','head']].merge(modifiers.reset_index(),how='left',on=['modifier'])
    constituent_sim.set_index(['modifier','head'],inplace=True)

constituent_sim=constituent_sim.multiply(heads.reindex(constituent_sim.unstack('modifier').index)).sum(axis=1).to_frame()
constituent_sim.columns=['sim_bw_constituents']



dfs = [constituent_sim, compound_head_sim, compound_modifier_sim, information_feat,productivity]
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
    compounds_final.fillna(0,inplace=True)
    compounds_final -= compounds_final.min()
    compounds_final /= compounds_final.max()





reddy_comp=pd.read_csv("../data/reddy_compounds.txt",sep="\t")
#print(reddy_comp.columns)
reddy_comp.columns=['compound','to_divide']
reddy_comp['modifier_mean'],reddy_comp['modifier_std'],reddy_comp['head_mean'],reddy_comp['head_std'],reddy_comp['compound_mean'],reddy_comp['compound_std'],_=reddy_comp.to_divide.str.split(" ",7).str
reddy_comp['modifier'],reddy_comp['head']=reddy_comp['compound'].str.split(" ",2).str
reddy_comp.modifier=reddy_comp.modifier.str[:-2]
reddy_comp['head']=reddy_comp['head'].str[:-2]
reddy_comp.drop(['compound','to_divide'],axis=1,inplace=True)
reddy_comp['modifier']=np.vectorize(lemma_maker)(reddy_comp['modifier'],'n')
reddy_comp['head']=np.vectorize(lemma_maker)(reddy_comp['head'],'n')
reddy_comp.replace(spelling_replacement,inplace=True)
#reddy_comp['modifier']=reddy_comp['modifier']+"_noun"
#reddy_comp['head']=reddy_comp['head']+"_noun"
reddy_comp=reddy_comp.apply(pd.to_numeric, errors='ignore')
#reddy_comp.set_index(['modifier','head'],inplace=True)



comp_90=pd.read_csv("../data/compounds90.txt",sep="\t")
comp_90['mod_pos'],comp_90['head_pos']=comp_90.compound_lemmapos.str.split('_').str
comp_90['modifier'],comp_90['mod_pos']=comp_90.mod_pos.str.split('/').str
comp_90['head'],comp_90['head_pos']=comp_90.head_pos.str.split('/').str
comp_90=comp_90.loc[~(comp_90.mod_pos=="ADJ")]
comp_90=comp_90.loc[:,['avgModifier','stdevModifier','avgHead','stdevHeadModifier','compositionality','stdevHeadModifier','modifier','head']]
comp_90.columns=reddy_comp.columns


comp_ext=pd.read_csv("../data/compounds_ext.txt",sep="\t")
comp_ext['mod_pos'],comp_ext['head_pos']=comp_ext.compound_lemmapos.str.split('_').str
comp_ext['modifier'],comp_ext['mod_pos']=comp_ext.mod_pos.str.split('/').str
comp_ext['head'],comp_ext['head_pos']=comp_ext.head_pos.str.split('/').str
comp_ext=comp_ext.loc[~(comp_ext.mod_pos=="ADJ")]

comp_ext=comp_ext.loc[:,['avgModifier','stdevModifier','avgHead','stdevHeadModifier','compositionality','stdevHeadModifier','modifier','head']]
comp_ext.columns=reddy_comp.columns


all_compounds=pd.concat([reddy_comp,comp_ext,comp_90],ignore_index=True)
all_compounds['modifier']=all_compounds['modifier']+"_noun"
all_compounds['head']=all_compounds['head']+"_noun"



merge_df=all_compounds.merge(compounds_final.reset_index(),on=['modifier','head'],how='inner')
merge_df.set_index(["modifier", "head"], inplace = True)

merge_df.to_csv(features_df_path,sep='\t')