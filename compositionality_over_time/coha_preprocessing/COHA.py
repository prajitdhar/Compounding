import pandas as pd
import csv
import os
import io
import zipfile
import random

outputfile = 'coha_compounds/coha_fivegrams.tsv'

#_dir = "/resources/corpora/COHA/text/"
_dir = "/resources/corpora/COHA/CCOHA/tagged/"
#_dir = "/resources/corpora/COHA/ALL/"
files = sorted(os.listdir(_dir))

print("Read files from " + _dir)
df_list = []
for f in files:
    year = f.split("_")[1].split(".")[0][:-1]
    if os.path.basename(os.path.normpath(_dir)) == "ALL":
        df_decade = pd.read_csv(os.path.join(_dir, f), sep='\t', quotechar='"', quoting=csv.QUOTE_NONE, header=None, usecols=[2,3,4], encoding = "latin")
    elif os.path.basename(os.path.normpath(_dir)) == "tagged":
        z = zipfile.ZipFile(os.path.join(_dir, f))
        zinfos = z.infolist()
        for zinfo in zinfos:
            df_decade = pd.read_csv(z.open(zinfo.filename), sep='\t', quotechar='"', quoting=csv.QUOTE_NONE, header=None, usecols=[0,1,2], encoding = "utf-8")
            df_decade.columns = ['token', 'lemma', 'pos']
            df_decade["decade"] = year
            df_list.append(df_decade)
df = pd.concat(df_list)


print("Cleanup...")
df = df[~df['pos'].str.contains('<sub>', na=False)].reset_index(drop = True)

print("Get adjacent N-N...")
index = pd.Index(df[df["pos"].str.match("^nn1$") & 
                    df["pos"].shift(-1).str.match("^nn1$") & 
                    ~(df["pos"].shift(-2).str.match("^nn1$").astype("bool") |
                      df["pos"].shift(-2).str.contains("vhd").astype("bool"))].index)
index_next = index + 1
index_full = index.union(index_next)
nn = df.loc[index_full]

print("Create windows...")
index_window=index
window = 2
window_end = index+(window+1)
window_begin = index-window
for i in range(-window,window+2,1):
    index_window = index_window.union(index+i)
df_windowed = df.loc[index_window]
df_windowed["span"] = "i"
df_windowed["span"][window_begin] = "b"
windows = {}
for i, row in df_windowed.iterrows():
    if row["span"] == "b":
        window_content = []
        for j in range(i,i+((window*2)+2)):
            window_content.append(str(df_windowed.loc[j].lemma) + "_" + str(df_windowed.loc[j].pos))
        if df_windowed.loc[j].decade not in windows:
            windows[df_windowed.loc[j].decade] = [window_content]
        else:
            windows[df_windowed.loc[j].decade].append(window_content)
            
windows_fivegrams = {}
for d in windows:
    for w in windows[d]:
        random.seed(1991)
        n = random.randint(0,1)
        if n == 0:
            if d in windows_fivegrams:
                windows_fivegrams[d].append(w[1:])
            else:
                windows_fivegrams[d] = [w[1:]]
        if n == 1:
            if d in windows_fivegrams:
                windows_fivegrams[d].append(w[:-1])
            else:
                windows_fivegrams[d] = [w[:-1]]

print("Create data frame")
tmp_list = []
for d in windows_fivegrams:
    for w in windows_fivegrams[d]:
        tmp_list.append({'ngram': " ".join(w), 'year': d, 'match_count': 1, 'volume_count': 1})
df_fivegrams = pd.DataFrame(tmp_list)
df_fivegrams.groupby(['ngram', 'year'])['match_count'].sum().to_frame()

print("Write to " + outputfile)
df_fivegrams.to_csv (outputfile, index = False, header=False, sep="\t")