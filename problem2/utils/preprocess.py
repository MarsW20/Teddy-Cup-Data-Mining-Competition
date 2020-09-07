import pandas as pd
import numpy as np
import gensim
import nltk
import json
import re
import jieba
'''
1、抽取附件中的所有文本用于word-embedding预训练
'''

class TextReader(object):
    def __init__(self):
        self.theme=self._read_theme()
        self.content=self._read_content()
    def _read_theme(self):
        f1=pd.read_excel("../data/附件2.xlsx",header=0,usecols=[2])
        sent1=f1.values.tolist()
        f2=pd.read_excel("../data/附件3.xlsx",header=0,usecols=[2])
        sent2=f2.values.tolist()
        f3 = pd.read_excel("../data/附件4.xlsx", header=0, usecols=[2])
        sent3 = f3.values.tolist()
        sentences=sent1+sent2+sent3
        return sentences
    def _read_content(self):
        f1 = pd.read_excel("../data/附件2.xlsx", header=0, usecols=[4])
        sent1 = f1.values.tolist()
        f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[4])
        sent2 = f2.values.tolist()
        f3 = pd.read_excel("../data/附件4.xlsx", header=0, usecols=[4])
        sent3 = f3.values.tolist()
        f4 = pd.read_excel("../data/附件4.xlsx", header=0, usecols=[5])
        sent4 = f4.values.tolist()
        sum = sent1 + sent2+sent3+sent4
        #数据清洗
        sentences=[]
        for sent in sum:
            sent[0]=re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]",',',sent[0])
            sent[0]=re.sub(u",{2,}",',',sent[0])[1:-1]
            sentences.append(sent)
        return sentences

def wrod_embedding(sents):
    vocab={}
    model = gensim.models.Word2Vec(sents, min_count=1)
    for word in model.wv.index2word:
        vocab[word] = model[word].tolist()
    f = open("../data/vocab.json", "w", encoding="utf-8")
    json.dump(vocab, f)

def tokenize(sents):
    res=[]
    for sent in sents:
        res.append(jieba.lcut(sent[0]))
    return res


if __name__=="__main__":
    dataSet=TextReader()
    sentences=tokenize(dataSet.theme+dataSet.content)
    wrod_embedding(sentences)
