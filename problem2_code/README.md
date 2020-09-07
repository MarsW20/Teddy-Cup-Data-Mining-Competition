"""
请在运行代码之前阅读该说明书：

    1、本项目用于泰迪杯第二问文本聚类，我们利用了tf-idf和CNN两种方式去进行聚类
    2、关于tf-idf原理和CNN原理可以参考我们的论文
    3、运行代码说明

"""
    ##数据准备
    请先将四个附件数据移至"data"文件夹下
"""

1、长文本转化为短文本

## 我们提供了两种关键词提取算法,二者可选其一

## summary1---PageRank算法

(1）cd summary1
(2)python main.py

## summary2---TF-IDF算法

(1)cd summary2
(2)python gen_idf.py
(3)python tfidf.py

2、基于td-idf+hash+LSA

(1)cd tf_idf_hash_LSA
(2)python kmeans.py

## 由于kmeans算法具有一定的随机性，为了得到更好的结果，我们采用多次运行kmeans.py文件
## 每次kmeans.py的运行,会在data目录下生成一个"answer+随机数.json"
## 通过手动选取其中的较好的聚类结果，将其复制至answer.json文件
## 最后answer.json文件内容为

{0:[],1:[],2:[],3:[],4:[]}

## 生成answer.json文件之后在utils文件夹下运行util.py文件，即可生成问题要求的表2及其时间和热度


3、基于CNN

"""
    我们采用了两种方式去预训练word_embedding
    （1）https://github.com/Embedding/Chinese-Word-Vectors
        """sgns.wiki.word"""
    （2）采用utils目录下的preprocess.py预训练word_embedding
     (3) 我们提供了训练好的20_cnn_model.pkl，可以直接执行（3）命令
"""

(1)cd STCC
(2)python cnn_model.py/python model.py
(3)python clustering.py

