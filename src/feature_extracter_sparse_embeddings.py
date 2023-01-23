import pandas as pd
import numpy as np
import argparse
import time
import pickle as pkl

from itertools import product
from functools import reduce

import seaborn as sns
sns.set(style="whitegrid", font_scale = 2.5)
sns.set_context(rc={"lines.markersize": 17, "lines.linewidth": 2})

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Compute features from sparse dataset')

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

parser.add_argument('--tag', action='store_true',
                    help='Should the POS tag be kept?')

args = parser.parse_args()



test_df=pd.read_csv('Compounding/data/all_compounds.txt',sep='\t')


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
    

def process_time_compound(df):

    print(f'Temporal information is stored with intervals {temporal}')

    if temporal==0:
        df['time']=0
    else:
        df['time']=df['year'] - df['year']%temporal
    
    df=df.groupby(['modifier','head','time','context'],observed=True)['count'].sum().to_frame()
    df.reset_index(inplace=True)
    return df
        
def process_cutoff_compound(df):

    df=df.loc[df.groupby(['modifier','head','time'],observed=True)['count'].transform('sum').gt(cutoff)]
    df=df.groupby(['modifier','head','time','context'],observed=True)['count'].sum().to_frame()
    
    df.reset_index(inplace=True)
    return df


def process_modifier(df):
            
    if temporal==0:
        df['time']=0
    else:
        df['time']=df['year'] - df['year']%temporal
        
    df=df.groupby(['modifier','time','context'],observed=True)['count'].sum().to_frame()

    df.reset_index(inplace=True)
    
    return df


def process_head(df):
    
    if temporal==0:
        df['time']=0
    else:
        df['time']=df['year'] - df['year']%temporal
        
    df=df.groupby(['head','time','context'],observed=True)['count'].sum().to_frame()
    
    df.reset_index(inplace=True)
    
    return df


def calculate_compound_features():
            
            
    print('Calculating productivity features')

    compound_counts=all_comps.groupby(['time']).size().to_frame()
    compound_counts.columns=['N']


    mod_prod=all_comps.groupby(['modifier','time'],observed=True).size().to_frame()
    mod_prod.columns=['mod_prod']
    mod_prod=pd.merge(mod_prod.reset_index(),compound_counts.reset_index(),on=['time'],how='left')
    mod_prod['mod_family_size']=1+np.log2(mod_prod.N/(mod_prod.mod_prod+1))

    not_found_mod_prod=not_found_modifiers_df.copy()
    not_found_mod_prod['mod_prod']=0
    not_found_mod_prod=pd.merge(not_found_mod_prod,compound_counts.reset_index(),on=['time'],how='left')
    not_found_mod_prod['mod_family_size']=1+np.log2(not_found_mod_prod.N/(not_found_mod_prod.mod_prod+1))

    head_prod=all_comps.groupby(['head','time'],observed=True).size().to_frame()
    head_prod.columns=['head_prod']
    head_prod=pd.merge(head_prod.reset_index(),compound_counts.reset_index(),on=['time'],how='left')
    head_prod['head_family_size']=1+np.log2(head_prod.N/(head_prod.head_prod+1))

    not_found_head_prod=not_found_heads_df.copy()
    not_found_head_prod['head_prod']=0
    not_found_head_prod=pd.merge(not_found_head_prod,compound_counts.reset_index(),on=['time'],how='left')
    not_found_head_prod['head_family_size']=1+np.log2(not_found_head_prod.N/(not_found_head_prod.head_prod+1))

    mod_prod=pd.concat([mod_prod,not_found_mod_prod],ignore_index=True)
    head_prod=pd.concat([head_prod,not_found_head_prod],ignore_index=True)

    prod1=pd.merge(mod_prod.drop('N',axis=1),all_comps,on=['modifier','time'],how='right')
    productivity=pd.merge(head_prod,prod1,on=['head','time'],how='right')
    productivity['comp_prod']=1
    productivity['comp_family_size']=1+np.log2(productivity.N/(productivity.comp_prod+1))

    not_found_prod1=pd.merge(not_found_compounds_df,mod_prod.drop('N',axis=1),how='left',on=['modifier','time'])
    not_found_productivity=pd.merge(not_found_prod1,head_prod,how='left',on=['head','time'])
    not_found_productivity['comp_prod']=0
    not_found_productivity['comp_family_size']=1+np.log2(not_found_productivity.N/(not_found_productivity.comp_prod+1))


    productivity=pd.concat([productivity,not_found_productivity],ignore_index=True)
    productivity.set_index(['modifier','head','time'],inplace=True)

    productivity['const_prod']=productivity.mod_family_size*productivity.head_family_size
    productivity.drop('N',axis=1,inplace=True)

    print('Calculating information theory features')

    compound_time_counts=compounds.groupby('time').sum().sum(axis=1).to_frame()
    compound_time_counts.columns=['N']
    compound_time_counts.N=compound_time_counts.N.astype('float64')

    XY=compounds.groupby(['modifier','head','time'],observed=True).sum().sum(axis=1).to_frame()
    X_star=compounds.groupby(['modifier','time'],observed=True).sum().sum(axis=1).to_frame()
    Y_star=compounds.groupby(['head','time'],observed=True).sum().sum(axis=1).to_frame()

    XY.columns=['a']
    XY.a=XY.a.astype('float64')

    X_star.columns=['x_star']
    X_star.x_star=X_star.x_star.astype('float64')

    Y_star.columns=['star_y']
    Y_star.star_y=Y_star.star_y.astype('float64')

    merge1=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'])

    information_feat=pd.merge(merge1,Y_star.reset_index(),on=['head','time'])    

    information_feat['b']=information_feat['x_star']-information_feat['a']
    information_feat['c']=information_feat['star_y']-information_feat['a']


    information_feat=pd.merge(information_feat,compound_time_counts.reset_index(),on=['time'])

    information_feat['d']=information_feat['N']-(information_feat['a']+information_feat['b']+information_feat['c'])
    information_feat['x_bar_star']=information_feat['N']-information_feat['x_star']
    information_feat['star_y_bar']=information_feat['N']-information_feat['star_y']

    information_feat['log_ratio']=2*(information_feat['a']*np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))+\
            information_feat['b']*np.log2((information_feat['b']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y_bar']+1))+\
            information_feat['c']*np.log2((information_feat['c']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y']+1))+\
            information_feat['d']*np.log2((information_feat['d']*information_feat['N']+1)/(information_feat['x_bar_star']*information_feat['star_y_bar']+1)))
            information_feat['ppmi']=np.log2((information_feat['a']*information_feat['N']+1)/(information_feat['x_star']*information_feat['star_y']+1))
    information_feat['local_mi']=information_feat['a']*information_feat['ppmi']
    information_feat.loc[information_feat.ppmi<=0,'ppmi']=0
    information_feat.drop(['a','x_star','star_y','b','c','d','N','d','x_bar_star','star_y_bar'],axis=1,inplace=True)

    not_found_information_feat=not_found_compounds_df.copy()
    not_found_information_feat['log_ratio']=0
    not_found_information_feat['ppmi']=0
    not_found_information_feat['local_mi']=0


    information_feat=pd.concat([information_feat,not_found_information_feat],ignore_index=True)

    information_feat.set_index(['modifier','head','time'],inplace=True)
            
    print('Frequency features')
            
            
    not_found_X_star=not_found_modifiers_df.groupby(['modifier','time'],observed=True).sum().sum(axis=1).to_frame()
    not_found_X_star.columns=['x_star']

    not_found_Y_star=not_found_heads_df.groupby(['head','time'],observed=True).sum().sum(axis=1).to_frame()
    not_found_Y_star.columns=['star_y']

    X_star=pd.concat([X_star,not_found_X_star])
    Y_star=pd.concat([Y_star,not_found_Y_star])
            
            
    frequency_feat=pd.merge(XY.reset_index(),X_star.reset_index(),on=['modifier','time'],how='left')
    frequency_feat=frequency_feat.merge(Y_star.reset_index(),on=['head','time'],how='left')
    frequency_feat=frequency_feat.merge(compound_time_counts.reset_index(),on='time')
    frequency_feat.set_index(['modifier','head','time'],inplace=True)
    frequency_feat.columns=['comp_freq','mod_freq','head_freq','N']
    frequency_feat['comp_tf']=np.log2(1+frequency_feat.comp_freq)
    frequency_feat['log_comp_freq']=np.log2(frequency_feat.N/(frequency_feat.comp_freq+1))

    frequency_feat['mod_tf']=np.log2(1+frequency_feat.mod_freq)
    frequency_feat['log_mod_freq']=np.log2(frequency_feat.N/(frequency_feat.mod_freq+1))

    frequency_feat['head_tf']=np.log2(1+frequency_feat.head_freq)
    frequency_feat['log_head_freq']=np.log2(frequency_feat.N/(frequency_feat.head_freq+1))

    not_found_frequency_feat=not_found_compounds_df.groupby(['modifier','head','time'],observed=True).sum().sum(axis=1).to_frame()
    not_found_frequency_feat=pd.merge(not_found_frequency_feat.reset_index(),X_star.reset_index(),on=['modifier','time'])
    not_found_frequency_feat=pd.merge(not_found_frequency_feat,Y_star.reset_index(),on=['head','time'])
    not_found_frequency_feat=not_found_frequency_feat.merge(compound_time_counts.reset_index(),on='time')
    not_found_frequency_feat.set_index(['modifier','head','time'],inplace=True)
    not_found_frequency_feat.columns=['comp_freq','mod_freq','head_freq','N']


    not_found_frequency_feat['comp_tf']=np.log2(1+not_found_frequency_feat.comp_freq)
    not_found_frequency_feat['log_comp_freq']=np.log2(not_found_frequency_feat.N/(not_found_frequency_feat.comp_freq+1))

    not_found_frequency_feat['mod_tf']=np.log2(1+not_found_frequency_feat.mod_freq)
    not_found_frequency_feat['log_mod_freq']=np.log2(not_found_frequency_feat.N/(not_found_frequency_feat.mod_freq+1))

    not_found_frequency_feat['head_tf']=np.log2(1+not_found_frequency_feat.head_freq)
    not_found_frequency_feat['log_head_freq']=np.log2(not_found_frequency_feat.N/(not_found_frequency_feat.head_freq+1))


    frequency_feat=pd.concat([frequency_feat,not_found_frequency_feat])
    frequency_feat.drop('N',axis=1,inplace=True)
    
    return productivity,information_feat,frequency_feat

    
    
def calculate_cosine_features():
    
    print('Calculating denominator values')

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

    print('Calculating cosine features')

    compound_modifier_sim=pd.merge(compounds,modifiers,on=["modifier","context",'time'])
    compound_modifier_sim['numerator']=compound_modifier_sim['comp_count']*compound_modifier_sim['mod_count']
    compound_modifier_sim=compound_modifier_sim.groupby(['modifier','head','time'],observed=True)['numerator'].sum().to_frame()
    compound_modifier_sim=pd.merge(compound_modifier_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_modifier_sim=pd.merge(compound_modifier_sim,modifier_denom.reset_index(),on=['modifier','time'])
    compound_modifier_sim['sim_with_modifier']=compound_modifier_sim['numerator']/(compound_modifier_sim['compound_denom']*compound_modifier_sim['modifier_denom'])
    compound_modifier_sim.set_index(['modifier','head','time'],inplace=True)
    compound_modifier_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)

    compound_head_sim=pd.merge(compounds,heads,on=["head","context",'time'])
    compound_head_sim['numerator']=compound_head_sim['comp_count']*compound_head_sim['head_count']
    compound_head_sim=compound_head_sim.groupby(['modifier','head','time'],observed=True)['numerator'].sum().to_frame()
    compound_head_sim=pd.merge(compound_head_sim.reset_index(),compound_denom.reset_index(),on=["modifier","head",'time'])
    compound_head_sim=pd.merge(compound_head_sim,head_denom.reset_index(),on=['head','time'])
    compound_head_sim['sim_with_head']=compound_head_sim['numerator']/(compound_head_sim['compound_denom']*compound_head_sim['head_denom'])
    compound_head_sim.set_index(['modifier','head','time'],inplace=True)
    compound_head_sim.drop(['numerator','compound_denom'],axis=1,inplace=True)

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
    
    return compound_modifier_sim,compound_head_sim,constituent_sim
    
cutoff_list=[0,10,20,50,100,500,1000]
temporal_list=[0]

if args.contextual:
    
    
    print("CompoundAware Model")
    print("Loading the constituent and compound datasets")

    complete_compounds=pd.read_pickle(args.inputdir+"/compounds.pkl")
    
    complete_modifiers=pd.read_pickle(args.inputdir+"/modifiers.pkl")

    complete_heads=pd.read_pickle(args.inputdir+"/heads.pkl")
    
    
    if not args.tag:
        print('Removing tags')
        complete_compounds['head']=complete_compounds['head'].str.replace('_NOUN|_PROPN','',regex=True)
        complete_compounds.modifier=complete_compounds.modifier.str.replace('_NOUN|_PROPN','',regex=True)

        complete_modifiers.modifier=complete_modifiers.modifier.str.replace('_NOUN|_PROPN','',regex=True)

        complete_heads['head']=complete_heads['head'].str.replace('_NOUN|_PROPN','',regex=True)
    
    
    for temporal in temporal_list:
        
        print(f'Time span:  {temporal}')
        
        temporal_compounds=process_time_compound(complete_compounds)

        modifiers=process_modifier(complete_modifiers)
        print('Done reading modifiers')

        heads=process_head(complete_heads)
        print('Done reading heads')
        
   
        for cutoff in cutoff_list:
        
            mod_cols=modifiers.columns.tolist()
            mod_cols[-1]="count"
            modifiers.columns=mod_cols

            head_cols=heads.columns.tolist()
            head_cols[-1]="count"
            heads.columns=head_cols
            
            print(f'Cutoff: {cutoff}')
            print(f'Time span:  {temporal}')
            temp_cutoff_str=str(temporal)+'_'+str(cutoff)
            
            if cutoff==0:
                compounds=temporal_compounds.copy()
                
            else:

                compounds=process_cutoff_compound(temporal_compounds)

            print('Done reading compounds')
            
            
            timespan_list_df=pd.DataFrame(compounds.time.unique())
            timespan_list_df.columns=['time']

            compound_list_df=compound_df[['modifier','head']].copy()
            compound_list_df=compound_list_df.merge(timespan_list_df,how='cross')

            modifier_list_df=compound_df[['modifier']].drop_duplicates()
            modifier_list_df=modifier_list_df.merge(timespan_list_df,how='cross')

            head_list_df=compound_df[['head']].drop_duplicates()
            head_list_df=head_list_df.merge(timespan_list_df,how='cross')
            
            all_comps=compounds[['modifier','head','time']].copy()
            all_comps.drop_duplicates(inplace=True)
           
            all_mods=compounds[['modifier','time']].copy()
            all_mods.drop_duplicates(inplace=True)
            
            all_heads=compounds[['head','time']].copy()
            all_heads.drop_duplicates(inplace=True)
            
            not_found_compounds_df=compound_list_df.merge(all_comps, on=['modifier','head','time'], how='outer', suffixes=['', '_'], indicator=True)
            not_found_compounds_df=not_found_compounds_df.loc[not_found_compounds_df['_merge']=='left_only']
            not_found_compounds_df.drop('_merge',axis=1,inplace=True)
            
            
            not_found_modifiers_df=modifier_list_df.merge(all_mods, on=['modifier','time'], how='outer', suffixes=['', '_'], indicator=True)
            not_found_modifiers_df=not_found_modifiers_df.loc[not_found_modifiers_df['_merge']=='left_only']
            not_found_modifiers_df.drop('_merge',axis=1,inplace=True)
            
            not_found_heads_df=head_list_df.merge(all_heads, on=['head','time'], how='outer', suffixes=['', '_'], indicator=True)
            not_found_heads_df=not_found_heads_df.loc[not_found_heads_df['_merge']=='left_only']
            not_found_heads_df.drop('_merge',axis=1,inplace=True)
            
            
            productivity,information_feat,frequency_feat=calculate_compound_features()
            

            compound_modifier_sim,compound_head_sim,constituent_sim=calculate_cosine_features()

            print('Storing all features together')

            dfs = [frequency_feat,constituent_sim, compound_head_sim.sim_with_head, compound_modifier_sim.sim_with_modifier, information_feat,productivity]
            compounds_final = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True,how='outer'), dfs)

            compounds_final=pd.pivot_table(compounds_final.reset_index(), index=['modifier','head'], columns=['time'])


            compounds_final_1=compounds_final.columns.get_level_values(0)
            compounds_final_2=compounds_final.columns.get_level_values(1)

            cur_year=0
            new_columns=[]
            for year in compounds_final_2:
                new_columns.append(compounds_final_1[cur_year]+":"+str(year))
                cur_year+=1

            compounds_final.columns=new_columns

            plotdir = "/data/dharp/compounds/Compounding/compositionality_over_time/Plots/google/"
            if args.contextual:
                comp_str='CompoundAware'
            else:
                comp_str='CompoundAgnostic'

            if args.tag:
                tag_str='Tagged'
            else:
                tag_str='UnTagged'

            print(f'Cutoff: {cutoff}')
            print(f'Time span:  {temporal}')
            temp_cutoff_str=str(temporal)+'_'+str(cutoff)
            
            if temporal==0:
                
                qcut_labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100']

                compounds_final['comp_freq_bins']=pd.qcut(compounds_final['comp_freq:0'],q=10,labels=qcut_labels)
                compounds_final['comp_freq_bins']=compounds_final['comp_freq_bins'].astype(object)
                compounds_final.loc[pd.MultiIndex.from_frame(compound_df[['modifier','head']]),"comp_freq_bins"]="Compounds"
                compounds_final['comp_freq_bins']=compounds_final['comp_freq_bins'].astype('category')
                qcut_labels.append('Compounds')


                print('Log Frequency')

                to_select_cols=['comp_tf:0','mod_tf:0','head_tf:0']
                new_labels=['Compound Log Freq','Modifier Log Freq','Head Log Freq']
                plot_freq_df=compounds_final.reset_index().melt(id_vars=['comp_freq_bins'],value_vars=to_select_cols)
                plot_freq_df['variable'] = plot_freq_df['variable'].map(dict(zip(to_select_cols,new_labels)))

                plt.figure(figsize=(15,15))
                g=sns.boxplot(data=plot_freq_df, x="value", y="comp_freq_bins",hue='variable',order=qcut_labels,dodge=True,showfliers = False)
                g.set_xlabel("Log Frequency")
                g.set_ylabel("Percentile Bin")
                g.set_yticklabels(qcut_labels)

                med_values = plot_freq_df.groupby(['variable','comp_freq_bins'])['value'].median()
                std_dev_values= plot_freq_df.groupby(['variable','comp_freq_bins'])['value'].std()

                i=0
                for ytick in g.get_yticks():

                    g.text(med_values[new_labels[0],qcut_labels[i]],ytick-0.25,f"{round(med_values[new_labels[0],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[0],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    g.text(med_values[new_labels[1],qcut_labels[i]],ytick,f"{round(med_values[new_labels[1],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[1],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    g.text(med_values[new_labels[2],qcut_labels[i]],ytick+0.25,f"{round(med_values[new_labels[2],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[2],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    i+=1

                g.legend(title='Similarity Features', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15,title_fontsize=20)

                plt.savefig(f'{plotdir}/log_feature_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                plt.show()

                print('Family Size Features')


                to_select_cols=['mod_family_size:0','head_family_size:0']
                new_labels=['Mod Family Size','Head Family Size']
                plot_family_size_df=compounds_final.reset_index().melt(id_vars=['comp_freq_bins'],value_vars=to_select_cols)
                plot_family_size_df['variable'] = plot_family_size_df['variable'].map(dict(zip(to_select_cols,new_labels)))

                plt.figure(figsize=(15,15))
                g=sns.boxplot(data=plot_family_size_df, x="value", y="comp_freq_bins",hue='variable',order=qcut_labels,dodge=True,showfliers = False)
                g.set_xlabel("Family Size")
                g.set_ylabel("Percentile Bin")
                g.set_yticklabels(qcut_labels)

                med_values = plot_family_size_df.groupby(['variable','comp_freq_bins'])['value'].median()
                std_dev_values= plot_family_size_df.groupby(['variable','comp_freq_bins'])['value'].std()

                i=0
                for ytick in g.get_yticks():

                    g.text(med_values[new_labels[0],qcut_labels[i]],ytick-0.2,f"{round(med_values[new_labels[0],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[0],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    g.text(med_values[new_labels[1],qcut_labels[i]],ytick+0.2,f"{round(med_values[new_labels[1],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[1],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    #g.text(med_values[new_labels[2],qcut_labels[i]],ytick+0.25,f"{round(med_values[new_labels[2],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[2],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    i+=1

                g.legend(title='Family Size Features', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15,title_fontsize=20)

                plt.savefig(f'{plotdir}/family_size_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                plt.show()

                print('Productivity Features')

                to_select_cols=['mod_prod:0','head_prod:0']
                new_labels=['Mod Prod','Head Prod']
                plot_prod_df=compounds_final.reset_index().melt(id_vars=['comp_freq_bins'],value_vars=to_select_cols)
                plot_prod_df['variable'] = plot_prod_df['variable'].map(dict(zip(to_select_cols,new_labels)))

                plt.figure(figsize=(15,15))
                g=sns.boxplot(data=plot_prod_df, x="value", y="comp_freq_bins",hue='variable',order=qcut_labels,dodge=True,showfliers = False)
                g.set_xlabel("Family Size")
                g.set_ylabel("Percentile Bin")
                g.set_yticklabels(qcut_labels)

                med_values = plot_prod_df.groupby(['variable','comp_freq_bins'])['value'].median()
                std_dev_values= plot_prod_df.groupby(['variable','comp_freq_bins'])['value'].std()

                i=0
                for ytick in g.get_yticks():

                    g.text(med_values[new_labels[0],qcut_labels[i]],ytick-0.2,f"{round(med_values[new_labels[0],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[0],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    g.text(med_values[new_labels[1],qcut_labels[i]],ytick+0.2,f"{round(med_values[new_labels[1],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[1],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    #g.text(med_values[new_labels[2],qcut_labels[i]],ytick+0.25,f"{round(med_values[new_labels[2],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[2],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    i+=1

                g.legend(title='Productivity Features', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15,title_fontsize=20)

                plt.savefig(f'{plotdir}/prod_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                plt.show()

                print('PPMI')

                to_select_cols=['ppmi:0']
                new_labels=['PPMI']
                plot_ppmi_df=compounds_final.reset_index().melt(id_vars=['comp_freq_bins'],value_vars=to_select_cols)
                plot_ppmi_df['variable'] = plot_ppmi_df['variable'].map(dict(zip(to_select_cols,new_labels)))

                plt.figure(figsize=(15,15))
                g=sns.boxplot(data=plot_ppmi_df, x="value", y="comp_freq_bins",hue='variable',order=qcut_labels,dodge=True,showfliers = False)
                g.set_xlabel("PPMI")
                g.set_ylabel("Percentile Bin")
                g.set_yticklabels(qcut_labels)

                med_values = plot_ppmi_df.groupby(['variable','comp_freq_bins'])['value'].median()
                std_dev_values= plot_ppmi_df.groupby(['variable','comp_freq_bins'])['value'].std()

                i=0
                for ytick in g.get_yticks():

                    g.text(med_values[new_labels[0],qcut_labels[i]],ytick,f"{round(med_values[new_labels[0],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[0],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    #g.text(med_values[new_labels[1],qcut_labels[i]],ytick+0.2,f"{round(med_values[new_labels[1],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[1],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    #g.text(med_values[new_labels[2],qcut_labels[i]],ytick+0.25,f"{round(med_values[new_labels[2],qcut_labels[i]],2)}±{round(std_dev_values[new_labels[2],qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    i+=1

                #g.legend(title='PPMI Features', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=15,title_fontsize=20)
                g.get_legend().remove()
                plt.savefig(f'{plotdir}/ppmi_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                plt.show()

                print('Cosine Features')

                to_select_cols=['sim_bw_constituents:0','sim_with_head:0','sim_with_modifier:0']
                new_labels=['Between Const.', 'With Head','With Modifier']

                plot_cos_df=compounds_final.reset_index().melt(id_vars=['comp_freq_bins'],value_vars=to_select_cols)
                plot_cos_df['variable'] = plot_cos_df['variable'].map(dict(zip(to_select_cols,new_labels)))

                plt.figure(figsize=(15,15))
                g=sns.boxplot(data=plot_cos_df, x="value", y="comp_freq_bins",hue='variable',order=qcut_labels,dodge=True,showfliers = False)
                g.set_xlabel("Cosine Similarity")
                g.set_ylabel("Percentile Bin")
                g.set_yticklabels(qcut_labels)

                med_values = plot_cos_df.groupby(['variable','comp_freq_bins'])['value'].median()
                std_dev_values= plot_cos_df.groupby(['variable','comp_freq_bins'])['value'].std()

                i=0
                for ytick in g.get_yticks():

                    g.text(med_values['Between Const.',qcut_labels[i]],ytick-0.25,f"{round(med_values['Between Const.',qcut_labels[i]],2)}±{round(std_dev_values['Between Const.',qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    g.text(med_values['With Head',qcut_labels[i]],ytick,f"{round(med_values['With Head',qcut_labels[i]],2)}±{round(std_dev_values['With Head',qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    g.text(med_values['With Modifier',qcut_labels[i]],ytick+0.25,f"{round(med_values['With Modifier',qcut_labels[i]],2)}±{round(std_dev_values['With Modifier',qcut_labels[i]],2)}", va='center',size=11,color='black',weight='semibold')
                    i+=1

                plt.legend(title='Similarity Features', loc='upper right')

                plt.savefig(f'{plotdir}/cosine_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                plt.show()
                
                print('Saving the dataset')
                cur_ratings_df=compounds_final.reset_index().merge(compound_df,on=['modifier','head'])

                cur_ratings_df.to_csv(f'{args.inputdir}/features_{comp_str}_{tag_str}_{temp_cutoff_str}.csv',sep='\t',index=False)


            else:
                compounds_complete_index=compounds_final.index
                print(len(compounds_complete_index))

                compound_pivot=pd.pivot_table(constituent_sim,columns='time',index=['modifier','head'],values='sim_bw_constituents')
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

                compound_index_lst=[compounds_complete_index,compounds_decades_all_index,compounds_1900_end_index,compounds_1950_end_index,compounds_2000_end_index]
                tags_lst=['complete','all','1900','1950','2000']


                for cur_index_lst,cur_tag in zip(compound_index_lst,tags_lst):

                    print(cur_tag)
                    cur_df=compounds_final.loc[cur_index_lst]

                    cur_ratings_df=cur_df.reset_index().merge(compound_df,on=['modifier','head'])
                    print(cur_ratings_df.shape)

                    print('Raw frequency features')
                    to_select_cols_1=[col for col in cur_df if col.startswith('comp_freq')]
                    to_select_cols_2=[col for col in cur_df if col.startswith('mod_freq')]
                    to_select_cols_3=[col for col in cur_df if col.startswith('head_freq')]
                    to_select_cols=to_select_cols_1+to_select_cols_2+to_select_cols_3

                    plot_freq_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_freq_df[['variable','time']]=plot_freq_df['variable'].str.split(':',expand=True)


                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_freq_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Frequency")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/freq_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    plot_freq_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_freq_ratings_df[['variable','time']]=plot_freq_ratings_df['variable'].str.split(':',expand=True)
                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_freq_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Frequency")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/freq_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    print('Log frequency features')

                    to_select_cols_1=[col for col in cur_df if col.startswith('comp_tf')]
                    to_select_cols_2=[col for col in cur_df if col.startswith('mod_tf')]
                    to_select_cols_3=[col for col in cur_df if col.startswith('head_tf')]
                    to_select_cols=to_select_cols_1+to_select_cols_2+to_select_cols_3

                    plot_tf_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_tf_df[['variable','time']]=plot_tf_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_tf_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Log Frequency")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/log_freq_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    plot_tf_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_tf_ratings_df[['variable','time']]=plot_tf_ratings_df['variable'].str.split(':',expand=True)
                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_tf_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Log Frequency")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/log_freq_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    print('Family size')

                    to_select_cols_1=[col for col in cur_df if col.startswith('mod_family_size')]
                    to_select_cols_2=[col for col in cur_df if col.startswith('head_family_size')]
                    to_select_cols=to_select_cols_1+to_select_cols_2

                    plot_family_size_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_family_size_df[['variable','time']]=plot_family_size_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_family_size_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Family Size")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/family_size_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    plot_family_size_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_family_size_ratings_df[['variable','time']]=plot_family_size_ratings_df['variable'].str.split(':',expand=True)
                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_family_size_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Family Size")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/family_size_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    print('Productivity')

                    to_select_cols_1=[col for col in cur_df if col.startswith('mod_prod')]
                    to_select_cols_2=[col for col in cur_df if col.startswith('head_prod')]
                    to_select_cols=to_select_cols_1+to_select_cols_2


                    plot_prod_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_prod_df[['variable','time']]=plot_prod_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_prod_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Productivity")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/prod_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    plot_prod_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_prod_ratings_df[['variable','time']]=plot_prod_ratings_df['variable'].str.split(':',expand=True)
                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_prod_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Productivity")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/prod_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    print('Information Theory')

                    to_select_cols_1=[col for col in compounds_decades_all_df if col.startswith('log_ratio')]
                    to_select_cols_2=[col for col in compounds_decades_all_df if col.startswith('local_mi')]

                    plot_log_ratio_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols_1)
                    plot_log_ratio_df[['variable','time']]=plot_log_ratio_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_log_ratio_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Log Ratio")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/log_ratio_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    plot_lmi_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols_2)
                    plot_lmi_df[['variable','time']]=plot_lmi_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_lmi_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Local MI")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/lmi_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)


                    plot_log_ratio_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols_1)
                    plot_log_ratio_ratings_df[['variable','time']]=plot_log_ratio_ratings_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_log_ratio_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Log Ratio")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/log_ratio_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    plot_lmi_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols_2)
                    plot_lmi_ratings_df[['variable','time']]=plot_lmi_ratings_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_lmi_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Local MI")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/lmi_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    print('Cosine')
                    to_select_cols_1=[col for col in compounds_decades_all_df if col.startswith('sim_with_modifier')]
                    to_select_cols_2=[col for col in compounds_decades_all_df if col.startswith('sim_with_head')]
                    to_select_cols_3=[col for col in compounds_decades_all_df if col.startswith('sim_bw_constituents')]
                    to_select_cols=to_select_cols_1+to_select_cols_2+to_select_cols_3

                    plot_cosine_df=cur_df.reset_index().melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_cosine_df[['variable','time']]=plot_cosine_df['variable'].str.split(':',expand=True)

                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_cosine_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper right')
                    g.set_xlabel("Time")
                    g.set_ylabel("Cosine Similarity")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/cosine_{cur_tag}_wo_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    plot_cosine_ratings_df=cur_ratings_df.melt(id_vars=['modifier', 'head'],value_vars=to_select_cols)
                    plot_cosine_ratings_df[['variable','time']]=plot_cosine_ratings_df['variable'].str.split(':',expand=True)
                    plt.figure(figsize=(15,15))
                    g=sns.lineplot(x="time", y="value", hue="variable",data=plot_cosine_ratings_df,palette="Dark2", marker='o',linewidth=1,dashes=False,markers=True)#,err_style="bars", ci=68)
                    g.legend(loc='upper left')
                    g.set_xlabel("Time")
                    g.set_ylabel("Cosine Similarity")
                    plt.setp(g.get_xticklabels(), rotation=60)
                    plt.savefig(f'{plotdir}/cosine_{cur_tag}_with_ratings_{comp_str}_{tag_str}_{temp_cutoff_str}.png',dpi=300)

                    print('Saving the dataset')

                    cur_ratings_df.to_csv(f'{args.inputdir}/features_{cur_tag}_{comp_str}_{tag_str}_{temp_cutoff_str}.csv',sep='\t',index=False)

else:
    pass
    
