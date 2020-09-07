#!/usr/bin/python
# -*- coding: utf-8 -*-

from segmenter import segment
import pandas as pd
import re
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #解决中文乱码

class IDFLoader(object):
    def __init__(self, idf_path):
        self.idf_path = idf_path
        self.idf_freq = {}     # idf
        self.mean_idf = 0.0    # 均值
        self.load_idf()

    def load_idf(self):       # 从文件中载入idf
        cnt = 0
        with open(self.idf_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    word, freq = line.strip().split(' ')
                    cnt += 1
                except Exception as e:
                    pass
                self.idf_freq[word] = float(freq)

        print('Vocabularies loaded: %d' % cnt)
        self.mean_idf = sum(self.idf_freq.values()) / cnt


class TFIDF(object):
    def __init__(self, idf_path):
        self.idf_loader = IDFLoader(idf_path)
        self.idf_freq = self.idf_loader.idf_freq
        self.mean_idf = self.idf_loader.mean_idf

    def extract_keywords(self, sentence, topK=20):    # 提取关键词
        # 过滤
        seg_list = segment(sentence)

        freq = {}
        for w in seg_list:
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())

        for k in freq:   # 计算 TF-IDF
            freq[k] *= self.idf_freq.get(k, self.mean_idf) / total

        tags = sorted(freq, key=freq.__getitem__, reverse=True)  # 排序

        if topK:
            return tags[:topK]
        else:
            return tags

def load_sents():
    sents = []
    f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[4])
    for sent in f2.values.tolist():
        sent=str(sent)
        sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
        sent = re.sub(u",{2,}", ',', sent)[:-1]
        sents.append(sent)
    return sents

def gen_summary():
    summary = []
    tdidf = TFIDF("idf.txt")
    sentence =load_sents()
    for sent in sentence:
        tags = tdidf.extract_keywords(sent, 10)
        keytext=""
        for i in tags:  keytext+=i
        summary.append(keytext)

    df = pd.read_excel("../data/附件3.xlsx")
    df["摘要"] = summary
    df.to_excel("../data/附件3.xlsx", index=None)

if __name__=='__main__':
    gen_summary()

