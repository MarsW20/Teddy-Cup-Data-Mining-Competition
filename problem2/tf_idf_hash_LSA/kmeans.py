# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import jieba
import time
import argparse
import sys
sys.path.append('/Users/mac/Documents/programming/泰迪杯/Text_cluster')

from utils.util import *
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='train.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    k_means_opts(parser)
    opts = parser.parse_args()

    Labels=load_corpus()
    data=load_corpus()
    res=[]
    for line in data:
        res.append(" ".join(list(jieba.lcut(line))))
    #print(res[:2])

    if opts.use_hashing:
        if opts.use_idf:
            # 对HashingVectorizer的输出实施IDF规范化
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words=stop_words(), alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words=stop_words(),
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = NumberNormalizingVectorizer(max_df=0.5, max_features=opts.n_features,
                                                 min_df=2, stop_words=stop_words(), ngram_range=(1, 2),
                                                 use_idf=opts.use_idf)

    X = vectorizer.fit_transform(res)

    if opts.n_components:
        print("用LSA进行维度规约（降维）")

        # Vectorizer的结果被归一化，这使得KMeans表现为球形k均值（Spherical K-means）以获得更好的结果。
        # 由于LSA / SVD结果并未标准化，我们必须重做标准化。

        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)


        explained_variance = svd.explained_variance_ratio_.sum()
        print("SVD解释方差的step: {}%".format(
            int(explained_variance * 100)))

        print('特征抽取完成！')


    """ 这部分用于确定最佳聚类数量
    import matplotlib.pyplot as plt
    n_clusters = 400
    wcss = []
    for i in range(1, n_clusters):
        if opts.minibatch:
            km = MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=2, n_jobs=-1,
                                 init_size=1000, batch_size=1500, verbose=opts.verbose)
        else:
            km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=2, n_jobs=-1,
                        verbose=opts.verbose)
        km.fit(X)
        wcss.append(km.inertia_)
    plt.plot(range(1, n_clusters), wcss)
    plt.title('肘 部 方 法')
    plt.xlabel('聚类的数量')
    plt.ylabel('wcss')
    plt.show()
    """
    km = KMeans(n_clusters=opts.n_clusters, init='k-means++', max_iter=300, n_init=5,n_jobs=-1)
    #用训练好的聚类模型反推文档的所属的主题类别
    km.fit(X)
    label_prediction = km.predict(X)
    label_prediction = list(label_prediction)
    res=save_cluster(label_prediction,Labels,opts.n_clusters)
    gen_ans(res,label_prediction)


    news=[]
    label_new=[]
    for ind,i in enumerate(label_prediction):
        if i in res:
            news.append(X[ind,:].tolist())
            label_new.append(i)
    X=np.array(news)
    labels=label_new

    #去除掉少数的

    svd = TruncatedSVD(n_components=2).fit(X)
    datapoint = svd.transform(X)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(32, 28))
    label1=list(colors.values())

    color = [label1[i//2] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    plt.show()
    #
    # #
    #


