#-*- encoding:utf-8 -*-
from __future__ import print_function

import sys
from imp import reload
import pandas as pd


try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs
import TextRank4Keyword
from TextRank4Keyword import *
import re

data=pd.read_excel("../data/附件3.xlsx",usecols=[4])
data=data.values.tolist()
dataSet=[]

for sent in data:
    sent=sent[0]
    sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
    sent = re.sub(u",{2,}", ',', sent)[:-1]
    dataSet.append(sent[1:])

summary=[]
for text in dataSet:
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text,lower=True, window=3, pagerank_config={'alpha':0.85})

    keytext=""
    for ind,item in enumerate(tr4w.get_keywords(30, word_min_len=2)):
        keytext+=item.word
        if ind==10:  break

    #print('--phrase--')

    for ind,phrase in enumerate(tr4w.get_keyphrases(keywords_num=10, min_occur_num = 0)):
        keytext+=phrase
        if ind==2:  break

    summary.append(keytext)
df = pd.read_excel("../data/附件3.xlsx")
df["摘要"]=summary
df.to_excel("../data/附件3.xlsx",index=None)





