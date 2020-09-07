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
    f2 = pd.read_excel("附件4.xlsx", header=0, usecols=[2, 4])
    f2["merge"] = f2["留言主题"] + f2["留言详情"]
    for sent in f2["merge"].values.tolist():
        sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
        sent = re.sub(u",{2,}", ',', sent)[:-1]
        sents.append(sent)
    return sents
def load_corpus():
    corpus = []
    f2 = pd.read_excel("附件4.xlsx", header=0, usecols=[5])
    for sent in f2.values.tolist():
        corpus.append(sent[0])
    return corpus

def calcu_voacb():
    tdidf = TFIDF("idf.txt")
    sentence =load_sents()
    vocab=[]
    for sent in sentence:
        tags = tdidf.extract_keywords(sent, 10)
        vocab.extend(tags)
    vocab=set(vocab)
    res={}
    for ind,i in enumerate(vocab):
        res[i]=ind
    res["UNK"]=len(vocab)
    return res

import numpy as np


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = cos
    return sim

def main():
    vocab=calcu_voacb()
    #print(vocab)

    sents1=load_sents()
    sents2=load_corpus()
    out=[]
    for i in range(len(sents1)):
        token1=segment(sents1[i])
        token2=segment(sents2[i])
        vec1=[0 for i in range(len(vocab))]
        vec2 = [0 for i in range(len(vocab))]
        for word in token1:
            if word in list(vocab.keys()):
                vec1[vocab[word]]+=1
            else:
                vec1[-1]+=1
        for word in token2:
            if word in list(vocab.keys()):
                vec2[vocab[word]] += 1
            else:
                vec2[-1] += 1
        out.append(cos_sim(vec1[:-1],vec2[:-1]))
    with open("res.txt","w") as f:
        for i in out:
            f.write(str(i)+'\n')
        f.close()

def paint():
    f=open("res.txt","r")
    y=[]
    for line in f.readlines():
        y.append(float(line))
    x=[i for i in range(len(y))]
    plt.plot(x,y)
    plt.ylabel("cos sim")
    plt.title("Correlation")
    plt.show()

def paint_():
    cnt_1=0
    cnt_2=0
    cnt_3=0
    f = open("res.txt", "r")
    y = []
    for line in f.readlines():
        y.append(float(line))
    for i in y:
        if i<0.2:   cnt_1+=1
        elif i>0.35 and i<0.65:    cnt_2+=1
        else: cnt_3+=1

    plt.figure(figsize=(6, 9))  # 调节图形大小
    labels = [u'很相关', u'比较相关', u'不相关']  # 定义标签
    sizes = [cnt_3, cnt_2, cnt_1]  # 每块值
    print(cnt_1,cnt_2,cnt_3)

    colors = ['red', 'yellowgreen', 'lightskyblue']  # 每块颜色定义
    explode = (0, 0, 0.02)  # 将某一块分割出来，值越大分割出的间隙越大
    patches, text1, text2 = plt.pie(sizes,
                                    explode=explode,
                                    labels=labels,
                                    colors=colors,
                                    labeldistance=1.05,  # 图例距圆心半径倍距离
                                    autopct='%3.2f%%',  # 数值保留固定小数位
                                    shadow=False,  # 无阴影设置
                                    startangle=90,  # 逆时针起始角度设置
                                    pctdistance=0.6)  # 数值距圆心半径倍数距离
    # patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
    # x，y轴刻度设置一致，保证饼图为圆形
    plt.axis('equal')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    #main()
    paint()
    paint_()
