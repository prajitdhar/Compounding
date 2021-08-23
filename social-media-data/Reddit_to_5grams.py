import pandas as pd
import fasttext
import time
import numpy as np
import spacy
import sys

fmodel = fasttext.load_model('/mnt/dhr/CreateChallenge_ICC_0821/lid.176.bin')

def delist_lang(lst):
    lang_lst=[]
    for i,lang in enumerate(lst):
        if not lang:
            lang_lst.append(None)
        else:
            lang_lst.append(lang[0])
    return lang_lst

def lang_tagger(parsed_sent):
    labels,confs=fmodel.predict(parsed_sent,k=-1,threshold=0.1)
    lang_list=delist_lang(labels)
    significance_list=significance(confs)
    assert len(lang_list)==len(significance_list)
    return lang_list,significance_list

def significance(lst):
    significance_list=[]
    for l in lst:
        if len(l)>1:
            significance_list.append(abs(l[0]-l[1])/np.mean(l[0]+l[1])>0.1)
            #print(f'{conf[0]} {conf[1]} {abs(conf[0]-conf[1])/np.mean(conf[0]+conf[1])>0.1}')
        else:
            significance_list.append(True)
    return significance_list

def post_to_five_grams(post):
    parsed_sent = nlp(post, disable=["parser", "ner"])
    token_pos = []
    for token in parsed_sent:
        token_pos.append(token.text+ "_" + token.pos_)
    if len(token_pos) < 4:
        return None
    # generate consecutive lists of 5 tokens
    five_grams = []
    for i in range(0, len(token_pos)-4, 1):
        five_grams.append(" ".join(token_pos[i:i+5]))
    return five_grams

def ner_lemma_reducer(sent):
    """
    parses sentence with spacy
    :param sent:
    :return: list of NER entities, lemmas, POS tags, whether each token is part of a compound
    """
    lemma = []
    pos = []
    is_comp = False
    ner_token = []

    # could limit here which components of spacy are run?
    parsed_sent = nlp(sent)
    for token in parsed_sent:
        lemma.append(token.lemma_)
        pos.append(token.pos_)
        if token.ent_type_ == "":
            to_add = "NONNER"
        else:
            to_add = token.ent_type_
        ner_token.append(to_add)
        if token.dep_ == "compound":
            is_comp = True

    lemma_sent = ' '.join(lemma)
    pos_sent = ' '.join(pos)
    ner_token_sent = ' '.join(ner_token)

    ner_length = 0
    if parsed_sent.ents:
        for ent in parsed_sent.ents:

            ner_length += ent.end_char - ent.start_char

    return ner_token_sent, ner_length, lemma_sent, pos_sent, is_comp, len(lemma)

fname = sys.argv[1]
type = sys.argv[2]
CHUNKSIZE = 500_000
dfs = pd.read_json(fname, lines=True, chunksize=CHUNKSIZE)

for i,df in enumerate(dfs):
    print(f'Split num {i+1}')
# df = pd.read_json(fname, lines=True)
    cur_time=time.time()

    # keep only non-empty, non-deleted texts
    if type == "submission":
        df = df[(df.selftext != "") & (df.selftext.notna()) & (df.selftext != "[deleted]")]
        df["text"] = df[['title', 'selftext']].agg(' '.join, axis=1)
    elif type == "comment":
        df = df[(df.body != "") & (df.body.notna()) & (df.body != "[deleted]")]
        df.rename(columns={"body": "text"}, inplace=True)

    df.loc[:, "text"] = df.loc[:, "text"].str.replace("\n", " ")
    # replace multiple spaces by just one
    df.loc[:, "text"] = df.loc[:, "text"].str.replace("\s+", " ")
    # print(df[["id", "text"]])

    # keep only English posts
    lang_list, significance_list = lang_tagger(df.text.values.tolist())
    df['lang'] = lang_list
    df['lang_conf'] = significance_list
    df.lang = df.lang.str.split('_', n=4).str[-1]
    df = df.loc[(df.lang == 'en') & (df.lang_conf == True)]

    # extract the time
    df["created_at"] = pd.to_datetime(df.created_utc, unit="s")
    df["year"] = df.created_at.dt.year
    # df["month_year"] = df.created_at.dt.to_period("M")

    # tokenize + POS tag
    nlp = spacy.load("en_core_web_lg")

    # parse with spacy to create fivegrams
    fivegrams = []
    timestamps = []
    for text, timestamp in zip(df.text.values, df.year.values):
        cur_fivegrams = post_to_five_grams(text)
        if cur_fivegrams:
            fivegrams.extend(cur_fivegrams)
            timestamps.extend(len(cur_fivegrams) * [timestamp])
    print("Created list of %d fivegrams" %len(fivegrams))
    fivegrams = pd.DataFrame(data={"fivegram_pos": fivegrams, "year": timestamps})
    fivegrams["count"] = 1
    print("Created dataframe of fivegrams")
    fivegrams = fivegrams.groupby(["fivegram_pos", "year"])["count"].sum().reset_index()

    fivegrams.info()

    fivegrams.to_csv(fname.split(".")[0] + "_fivegrams_" + str(i) + ".csv")

    print("Converting %d posts to 5-grams took %s min" %(CHUNKSIZE, (time.time() - cur_time)/60))

    # print(fivegrams)

