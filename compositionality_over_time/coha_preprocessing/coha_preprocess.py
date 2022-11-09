import pandas as pd
import csv
import os
import io
from zipfile import ZipFile,ZipInfo
import editdistance
from collections import Counter
import numpy as np
from multiprocessing import Pool
import multiprocessing as mp
import unicodedata
import re
import spacy
import time
#spacy.prefer_gpu()
#import contextualSpellCheck

nlp = spacy.load('en_core_web_sm')
nlp.max_length=10_000_000


def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name).decode('utf-8').strip() for name in input_zip.namelist()}



def docenizer(dict_content):

    str1=re.sub("@@.*|<P>|/\S*/","",dict_content)
    str2 = re.sub(' +',' ',str1)
    str2=str2.replace("@ @ @ @ @ @ @ @ @ @","@@@@@@@@@@")
    str2=str2.replace("\n\n","")
    doc=nlp(str2)
    to_hold=[]
    n_words=0
    n_sents=0
    for sent in doc.sents:
        if len(sent)==1:
            continue
        cur_sent=[]
        for word in sent:
            cur_sent.append(word.text)
            n_words+=1
        to_hold.append(' '.join(cur_sent))
        n_sents+=1
    content='\n'.join(to_hold)
    content=content.lower()
    #content=content.replace("@@@@@@@@@@", "<mask>")
    return content,n_words,n_sents



def setmaker(fname,curr_year):
    cur_set=zipfiles[fname].split('\n')
    testset=cur_set[:1000]
    validset=cur_set[1001:2001]
    
    test_file="./"+str(curr_year)+"/test.txt"
    with open(test_file,'w') as f:
        for sent in testset:
            f.write(sent+"\n\n")

    valid_file="./"+str(curr_year)+"/valid.txt"           
    with open(valid_file,'w') as f:
        for sent in validset:
            f.write(sent+"\n\n") 
    return '\n'.join(cur_set[2002:])


def write_to_file(fnames,dec,set_type):
    save_file="./"+str(dec)+"/"+set_type+".txt"
    print(save_file)
    with open(save_file,'w') as f:
        for doc in fnames:
            f.write(zipfiles[doc]+"\n\n")


            
_dir = "/resources/corpora/COHA/text/"

files = sorted(os.listdir(_dir))

to_keep=[]
for f in files:
    if 'zip' in f:
        to_keep.append(f)


for decade in to_keep:
    zipfiles=extract_zip(os.path.join(_dir, decade))
    zfnames=list(zipfiles.keys())
    docs=list(zipfiles.values())
    
    decade=decade.split('_')[1][:-1]
    print(f"Current decade "+str(decade))
    cur_time=time.time()
    n_proc = mp.cpu_count()-1

    pool = Pool(n_proc)
    results=pool.map_async(docenizer,docs)
    pool.close()
    pool.join()

    print("Done parallelizing")
    print("Total time taken",round(time.time()-cur_time),"secs")
    results=results.get()
    contents=[val[0] for val in results]
    n_words=[val[1] for val in results]
    n_sents=[val[2] for val in results]

    zipfiles=dict(zip(zfnames,contents))
    
    genres=pd.DataFrame({'fname':zfnames,'content':contents,'n_words':n_words,'n_sents':n_sents})
    genres['gtype']=genres['fname'].str.split('_').str[0]

    min_docs=genres.sort_values('n_sents', ascending=False).head(1)['fname'].to_list()


    zipfiles[min_docs[0]]=setmaker(min_docs[0],decade)
    
    trainset_list=zipfiles.keys()
    
    write_to_file(trainset_list,decade,'train')