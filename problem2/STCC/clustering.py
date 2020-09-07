import numpy as np
import math
import jieba
import random
import time
import torch

import sys
sys.path.append('/Users/mac/Documents/programming/泰迪杯/Text_cluster')

import utils.util as data_util
import simhash
import laplacian_eigenmap
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import jieba
import time
import argparse

from cnn_model import *


def embedding(dic, corpus):
    ret = []
    for sent in corpus:
        ls = []
        for word in sent:
            if dic.get(word, -1) != -1:
                added = dic[word]
            else:
                added = [0] * 300
            ls.append(added)
        ret.append(ls)
    return ret


def getFeature(dic, Corpus):
    net=torch.load("20_cnn_model.pkl",map_location=torch.device("cpu"))
    represented = []
    length = len(Corpus)
    for i in range(0, length, 500):
        input = embedding(dic, Corpus[i:min(length, i + 500)])
        input = np.array(input)
        input = input.reshape(-1, 1, 20, 300)
        outputs, h = net(torch.Tensor(input))
        represented += h.data.numpy().tolist()
        print('(%d %% %d) have been Embedding.' % (i, length))

    print('all sentences finished Embedding')
    return represented


if __name__ == '__main__':
    start_time = time.process_time()

    #路径需要根据自己的修改
    dic = data_util.load_dense_drop_repeat("/Users/mac/Downloads/sgns.wiki.word")
    corpus = data_util.load_corpus()


    # cut and supply the corpus
    Corpus = corpus.copy()
    CorpusLength = len(Corpus)
    for i, line in enumerate(Corpus):
        Corpus[i] = jieba.lcut(Corpus[i])
        Corpus[i]=data_util.del_stopwords([Corpus[i]])[0]
        length = len(Corpus[i])
        if length > 20:
            Corpus[i] = Corpus[i][:20]
        if length < 20:
            Corpus[i] += (20 - length) * ['']
        if i % 10000 == 0:
            print('(%d %% %d) sentences has been cutted' % (i, CorpusLength))

    random.seed()
    random.shuffle(Corpus)

    featureRepresent = getFeature(dic, Corpus)

    X=np.array(featureRepresent)


    """这部分用于降维，可以不用
    svd = TruncatedSVD(30)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("SVD解释方差的step: {}%".format(
        int(explained_variance * 100)))

    print('特征抽取完成！')
    """

    km = KMeans(n_clusters=200, init='k-means++', max_iter=300, n_init=5, n_jobs=-1)
    # 用训练好的聚类模型反推文档的所属的主题类别
    km.fit(X)
    label_prediction = km.predict(X)
    label_prediction = list(label_prediction)
    res=data_util.save_cluster(label_prediction,corpus,200)

    data_util.gen_ans(res, label_prediction)

    news = []
    label_new = []
    for ind, i in enumerate(label_prediction):
        if i in res:
            news.append(X[ind, :].tolist())
            label_new.append(i)
    X = np.array(news)
    labels = label_new

    # 去除掉少数的
    svd = PCA(n_components=2).fit(X)
    datapoint = svd.transform(X)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(32, 28))
    label1 = list(data_util.colors.values())
    color = [label1[i // 2] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    plt.show()

    """"  用于PCA显示全部类的聚类效果
    #改变，用于PCA显示
    news = []
    label_new = []
    for ind, i in enumerate(label_prediction):
        if i in res:
            news.append(X[ind, :].tolist())
            label_new.append(i)
    X = np.array(news)
    labels = label_new
    # svd = TruncatedSVD(n_components=2).fit(X)
    svd = PCA(n_components=2).fit(X)
    datapoint = svd.transform(X)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(32, 28))
    label1 = list(data_util.colors.keys())
    color = [label1[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
    plt.show()
    """




