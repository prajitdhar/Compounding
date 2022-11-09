import pandas as pd
import csv
import os
import io
import zipfile
import spacy
import re
import time
import fasttext
import gc
import argparse
import pickle

parser = argparse.ArgumentParser(description='Program to process the coha files and store the sentences for each decade')

parser.add_argument('--input', type=str,
                    help='location of the directory with the coha zip files')
parser.add_argument('--fasttext', type=str,
                    help='location of fasttext model')

parser.add_argument('--output', type=str,
                    help='directory to save dataset in')

parser.add_argument('--start_from', type=str,
                    help='which decade to resume from')



args = parser.parse_args()



fmodel = fasttext.load_model(args.fasttext+'lid.176.bin')



_dir = args.input

coha_files = sorted(os.listdir(_dir))[1:]
print(len(coha_files))


def split_in_sentences(text,sent_segmenter):
    doc = sent_segmenter(text)
    return [str(sent).strip() for sent in doc.sents]

def lang_detect(sents):
    new_sents=[]
    for sent in sents:
        labels,_=fmodel.predict(sent,k=-1)
        if labels[0]=='__label__en':
            new_sents.append(sent)
    return new_sents


for zfile in coha_files:
    dec_time=time.time()
    cur_decade=zfile.split('_')[1]
    print(cur_decade)
    if int(cur_decade.rstrip('s'))<int(args.start_from.rstrip('s')):
        continue

    
    df_list=[]
    zip_file_orig    = zipfile.ZipFile(os.path.join(_dir, zfile))
    zinfos_orig = zip_file_orig.infolist()
    
    names=[]
    sizes=[]
    ids=[]
    for i,zfile in enumerate(zinfos_orig):
        names.append(zfile.filename)
        ids.append(i)
        sizes.append(zfile.file_size)
    zfile_df=pd.DataFrame({'fid':ids,'fname':names,'fsize':sizes})
    zfile_df['fsize_perc']=zfile_df.fsize/zfile_df.fsize.sum()*100
    zfile_df.sort_values(by=['fsize'],ascending=False,inplace=True,ignore_index=True)
    zfile_df.fsize/=1024*1024
    
    file_list=zfile_df.fname.to_list()
    
    
    sent_segmenter=spacy.load('en_core_web_sm')
    sent_segmenter.disable_pipe("parser")
    sent_segmenter.enable_pipe("senter")
    sent_segmenter.add_pipe("doc_cleaner")
    sent_segmenter.max_length=10_000_000
    
    sent_dict={}
    for i,file_id in enumerate(file_list):
        print(f'File {i+1} out of {len(file_list)}')
        print(file_id)
        items_file_orig  = zip_file_orig.open(file_id, 'r')
        inp_text=io.TextIOWrapper(items_file_orig).read()

        cur_year=int(file_id.split('_')[1])
        inp_text=re.sub('\\|p[\d]+', '', inp_text)
        inp_text=re.sub('\\|', '', inp_text)
        inp_text=re.sub('txt','',inp_text)
        inp_text=inp_text.split('\n\n')[-1]
        print(f'Number of characters {len(inp_text)}')
        print(f"Running sentence segmenter")

        sents=split_in_sentences(inp_text,sent_segmenter)

        print(f'Number of sentences {len(sents)}')
        print(f"Running language identifier")


        sents=lang_detect(sents)
        print(f'Number of sentences {len(sents)}')
        sent_dict[file_id]=sents
    

    print(f"Total time taken for decade {cur_decade} : {round(time.time()-dec_time)} secs")
    with open(args.output+"/"+cur_decade+".pkl", 'wb') as f:
        pickle.dump(sent_dict, f)