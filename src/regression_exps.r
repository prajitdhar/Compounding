library(tidyverse)
library(caret)
library(glmnet)
library(dplyr)
#library(psych) 
library(lme4)
library(e1071)
library(GGally)
library(optparse)
options(warn=-1)
options(scipen=999)
#library(crosstable)
options(error = quote({
  dump.frames(to.file=T, dumpto='last.dump')
  load('last.dump.rda')
  print(last.dump)
  q()
}))

option_list = list(
  make_option(c("-p", "--ppmi"), type="character", default="PPMI"),
  make_option(c("-s", "--setting"), type="character", default="Agnostic"),
  make_option(c("-m", "--median"), type="character", default="med"),
  make_option(c("-c", "--cpus"), type="integer", default=30)
  
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

seed_list<-c(1001,1111,1221,1331,1441,1551,1661,1771,1881,1991)
cut_off<-c(0,10,50,100,500,1000)
time_off<-c(10000,10,20,50,100)


lambda <- 10^seq(-3, 3, length = 100)
alpha<-seq(0,1,length=10)

library(doParallel)
cl <- makePSOCKcluster(opt$cpus)
registerDoParallel(cl)

m<-1
v<-1
#list_of_vi <- vector(mode="list", length=5)
#names(list_of_vi) <- c('10000', '10', '20', '50', '100')

list_of_vi_10000<-list()
list_of_vi_10<-list()
list_of_vi_20<-list()
list_of_vi_50<-list()
list_of_vi_100<-list()
list_of_rsqr=list()

corpus_list<-c("google")

ratings_list<-c("reddy","cordeiro90","cordeiro100")#,"reddy_true","cordeiro90_true","cordeiro100_true")
tagged_list<-c("Tagged")

ppmi_setting_list<-c(opt$ppmi)
comp_setting_list<-c(opt$setting)
impute_list<-c(opt$median)
to_predict_list<-c("compound","modifier","head")

feature_setting_list<-c("all")#,"cordeiro","cosine_sim","with_setting","info_theory","freq","prod")
caret_spearman <- function(data, lev = NULL, model = NULL) {
  spearman_val <- cor(x = data$pred, y = data$obs, method = "spearman")
  c(Spearman = spearman_val)
}

needed_cols<-c('modifier','head','avgModifier','stdevModifier','avgHead','stdevHead','compositionality','stdevHeadModifier','is_adj','compound','source')

cordeiro_cols<-c('arith_mean_sim.','beta.','geom_mean_sim.','sim_cpf_0.','sim_cpf_100.','sim_cpf_25.','sim_cpf_50.','sim_cpf_75.','sim_cpf_beta.')
cosine_sim_cols<-c("sim_bw_constituents.",'sim_with_head.','sim_with_modifier.')

with_setting_cols<-c("sim_bw_settings_comp.","sim_bw_settings_head.","sim_bw_settings_modifier.")
info_theory_cols<-c("local_mi.","log_ratio.","ppmi.")
freq_cols<-c("comp_freq.","comp_tf.","head_freq.","head_tf.","log_comp_freq.","log_head_freq.","log_head_freq_new.","log_mod_freq.","log_mod_freq_new.","mod_freq.","mod_tf.")
prod_cols<-c("head_family_size.","head_family_size_new.","head_prod.","mod_family_size.","mod_family_size_new.","mod_prod.")



for (c in corpus_list){
  for (t in tagged_list) {
    
    for (p in ppmi_setting_list){
      for (a in comp_setting_list){
        for (i in time_off) {
          
          for (j in cut_off) {
            
            for (im in impute_list) {
              
              print(paste0("/data/dharp/compounds/datasets/",c,"/features_Compound",a,"_withSetting_",p,"_",t,"_",i,"_",j,"_",im,".csv")) 
              input_df<-read.csv(paste0("/data/dharp/compounds/datasets/",c,"/features_Compound",a,"_withSetting_",p,"_",t,"_",i,"_",j,"_",im,".csv"),sep = '\t')
              
              for (r in ratings_list){
                
                reduced_df<-input_df %>% filter(source==r)
                reduced_df <- reduced_df %>% filter(!compound %in% c('call_centre','benign_tumour', 'computer_programme', 'grey_matter', 'labour_union','baby_blues', 'contact_lenses'))
                
                
                
                
                if (t=="Tagged") {
                  compounds_list<-c("all","an","nn")
                }
                else {
                  compounds_list<-c("all")
                  
                }
                for (k in compounds_list) {
                  df_features<-reduced_df
                  
                  if (k=="an") { 
                    group1_df<-df_features %>% filter(is_adj=="True" & grepl("_ADJ",modifier) & grepl("_NOUN",head))
                    group2_df<-df_features %>% filter(is_adj=="False" & grepl("_NOUN",modifier) & grepl("_NOUN",head))
                    df_features<-rbind(group1_df,group2_df)                   
                  }
                  else if (k=="nn") {
                    group1_df<-df_features %>% filter(is_adj=="True" & grepl("_NOUN",modifier) & grepl("_NOUN",head))
                    group2_df<-df_features %>% filter(is_adj=="False" & grepl("_NOUN",modifier) & grepl("_NOUN",head))
                    df_features<-rbind(group1_df,group2_df)                    
                  }
                  
                  else if (k=="ap") {
                    group1_df<-df_features %>% filter(is_adj=="True" & grepl("_ADJ",modifier) & grepl("_PROPN",head))
                    group2_df<-df_features %>% filter(is_adj=="False" & grepl("_PROPN",modifier) & grepl("_PROPN",head))
                    df_features<-rbind(group1_df,group2_df)                    
                  }
                  
                  else if (k=="pp") {
                    group1_df<-df_features %>% filter(is_adj=="True" & grepl("_PROPN",modifier) & grepl("_PROPN",head))
                    group2_df<-df_features %>% filter(is_adj=="False" & grepl("_PROPN",modifier) & grepl("_PROPN",head))
                    df_features<-rbind(group1_df,group2_df)                    
                  }
                  
                  df_features_X<-df_features %>% select(-c(compositionality,modifier,head,avgModifier,stdevModifier,avgHead,stdevHead,stdevHeadModifier,compound,is_adj,source)) %>% select(-one_of("comp_freq_bins"))
                  
                  for (f in feature_setting_list) {
                    
                    if(f=="cordeiro"){
                      trainX<-df_features_X %>% select(starts_with(cordeiro_cols))
                    }
                    else if(f=="cosine_sim"){
                      trainX<-df_features_X %>% select(starts_with(cosine_sim_cols))
                      
                    }
                    
                    else if(f=="with_setting"){
                      trainX<-df_features_X %>% select(starts_with(with_setting_cols))
                    }
                    
                    else if(f=="info_theory"){
                      trainX<-df_features_X %>% select(starts_with(info_theory_cols))
                    }
                    
                    else if(f=="freq"){
                      trainX<-df_features_X %>% select(starts_with(freq_cols))
                    }
                    else if(f=="prod"){
                      trainX<-df_features_X %>% select(starts_with(prod_cols))
                    }
                    else {
                      trainX<-df_features_X
                    }
                    
                    if (dim(trainX)[1]<10) {
                      break
                    }
                    for (pr in to_predict_list){
                      
                      if (pr=="compound") {
                        trainY<-df_features %>% select(compositionality)
                        trainY<-trainY$compositionality      
                      }
                      
                      else if (pr=="modifier") {
                        trainY<-df_features %>% select(avgModifier)
                        trainY<-trainY$avgModifier      
                      }
                      
                      else if (pr=="head") {
                        trainY<-df_features %>% select(avgHead)
                        trainY<-trainY$avgHead      
                      }                               
                      
                      print(paste0(c," ",t," ",p," ",a," ",i," ",j," ",im," ",r," ",k," ",f," ",pr))
                      
                      for (s in seed_list)  {
                        set.seed(s)
                        seeds <- vector(mode = "list", length = 14)
                        for(z in 1:13) seeds[[z]] <- sample.int(n=1000, 10)
                        #for the last model
                        seeds[[14]]<-sample.int(1000, 1)
                        
                        if (im=="na") {
                          preprocess_list<-c("nzv","medianImpute", "center", "scale")
                        }
                        else {
                          preprocess_list<-c("nzv", "center", "scale")
                        }
                        
                        
                        elastic_model <- train(trainX,trainY,method = "glmnet",metric = "Rsquared",
                                               trControl = trainControl("cv", number = 10,search="grid",seeds=seeds),tuneGrid = expand.grid(alpha = alpha, lambda = lambda),
                                               preProcess = preprocess_list)
                        
                        elastic_spearman_model <- train(trainX,trainY,method = "glmnet",metric = "Spearman",
                                                        trControl = trainControl("cv", number = 10,search="grid",seeds=seeds,summaryFunction = caret_spearman),tuneGrid = expand.grid(alpha = alpha, lambda = lambda),
                                                        preProcess = preprocess_list)
                        
                        
                        perf_elastic<-data.frame(corpus=c,tag=t,ppmi=p,setting=a,timespan=i,cutoff=j,impute=im,dataset=r,pattern=k,features=f,y=pr,n=nrow(trainX),seed=s,ml_algo="elastic",method=getTrainPerf(elastic_model)[,"method"],TrainRsquared=getTrainPerf(elastic_model)[,"TrainRsquared"],TrainSpearman=getTrainPerf(elastic_spearman_model)[,"TrainSpearman"])
                        
                        varimp_elastic<-data.frame(corpus=c,tag=t,ppmi=p,setting=a,timespan=i,cutoff=j,impute=im,dataset=r,pattern=k,features=f,y=pr,n=nrow(trainX),seed=s,ml_algo="elastic",t(varImp(elastic_model)$importance))
                        
                        
                        list_of_rsqr[[m]]<-perf_elastic
                        m<-m+1 
                        
                        if (i==10000) {
                          list_of_vi_10000[[v]]<-varimp_elastic
                          v<-v+1
                        }
                        else if (i==10) {
                          list_of_vi_10[[v]]<-varimp_elastic
                          v<-v+1
                        }
                        
                        else if (i==20) {
                          list_of_vi_20[[v]]<-varimp_elastic
                          v<-v+1
                        }
                        else if (i==50) {
                          list_of_vi_50[[v]]<-varimp_elastic
                          v<-v+1
                        }
                        else if (i==100) {
                          list_of_vi_100[[v]]<-varimp_elastic
                          v<-v+1
                        }                                                                                                   
                        
                      }
                      rsquared_df<-bind_rows(list_of_rsqr)
                      rsquared_df$cutoff<-as.factor(rsquared_df$cutoff)
                      write.csv(rsquared_df,paste0("~/rsquared_",c,"_",t,"_",p,"_",a,"_",i,"_",j,"_",im,"_",r,"_",k,"_",f,"_",pr,".csv"),row.names = FALSE)
                      list_of_rsqr<-list()
                      if (i==10000) {
                        varimp_10000_df<-bind_rows(list_of_vi_10000)
                        varimp_10000_df$cutoff<-as.factor(varimp_10000_df$cutoff)
                        varimp_10000_df[is.na(varimp_10000_df)] <- 0
                        write.csv(varimp_10000_df,paste0("~/varimp_10000_",c,"_",t,"_",p,"_",a,"_",i,"_",j,"_",im,"_",r,"_",k,"_",f,"_",pr,".csv"),row.names = FALSE)
                        list_of_vi_10000<-list()
                        
                      }
                      else if (i==10) {
                        varimp_10_df<-bind_rows(list_of_vi_10)
                        varimp_10_df$cutoff<-as.factor(varimp_10_df$cutoff)
                        varimp_10_df[is.na(varimp_10_df)] <- 0
                        write.csv(varimp_10_df,paste0("~/varimp_10_",c,"_",t,"_",p,"_",a,"_",i,"_",j,"_",im,"_",r,"_",k,"_",f,"_",pr,".csv"),row.names = FALSE)
                        list_of_vi_10<-list()
                      }
                      
                      else if (i==20) {
                        varimp_20_df<-bind_rows(list_of_vi_20)
                        varimp_20_df$cutoff<-as.factor(varimp_20_df$cutoff)
                        varimp_20_df[is.na(varimp_20_df)] <- 0
                        write.csv(varimp_20_df,paste0("~/varimp_20_",c,"_",t,"_",p,"_",a,"_",i,"_",j,"_",im,"_",r,"_",k,"_",f,"_",pr,".csv"),row.names = FALSE)
                        list_of_vi_20<-list()
                        
                      }
                      else if (i==50) {
                        varimp_50_df<-bind_rows(list_of_vi_50)
                        varimp_50_df$cutoff<-as.factor(varimp_50_df$cutoff)
                        varimp_50_df[is.na(varimp_50_df)] <- 0
                        write.csv(varimp_50_df,paste0("~/varimp_50_",c,"_",t,"_",p,"_",a,"_",i,"_",j,"_",im,"_",r,"_",k,"_",f,"_",pr,".csv"),row.names = FALSE)
                        list_of_vi_50<-list()
                        
                      }
                      else if (i==100) {
                        varimp_100_df<-bind_rows(list_of_vi_100)
                        varimp_100_df$cutoff<-as.factor(list_of_vi_100$cutoff)
                        varimp_100_df[is.na(list_of_vi_100)] <- 0
                        write.csv(varimp_100_df,paste0("~/varimp_50_",c,"_",t,"_",p,"_",a,"_",i,"_",j,"_",im,"_",r,"_",k,"_",f,"_",pr,".csv"),row.names = FALSE)
                        list_of_vi_100<-list()
                        
                      } 
                      
                      
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}