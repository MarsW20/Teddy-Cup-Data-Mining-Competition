# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import svm
import random
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import random
# -*- coding: utf-8 -*-
import csv
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #解决中文乱码



#调整了格式，一行是一条数据
def load_data(filename,splitRatio=0.70):
    f = open(filename,'r',encoding='utf-8-sig')
    dataset = [line  for line in f.readlines() if len(line.strip())>3]
    cnt = int(len(dataset) * splitRatio)
    random.shuffle(dataset)
    content=[]
    opinion=[]
    for line in dataset:
        content.append(list(line.strip().split())[1])
        if int(list(line.strip().split())[0])>0:
            opinion.append(1)
        else:
            opinion.append(0)
    train = [content, opinion]
    return train



# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c


# corpus = ["我 来到 北京 清华大学", "他 来到 了 网易 杭研 大厦", "小明 硕士 毕业 与 中国 科学院"]
train = load_data('标注数据.txt')
content = segmentWord(train[0])
opinion = train[1]


# 划分
cnt=int(len(content)*0.65)
train_content = content[:cnt]
test_content = content[cnt:]
train_opinion = opinion[:cnt]
test_opinion = opinion[cnt:]


# 计算权重
vectorizer = CountVectorizer("")
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  # 先转换成词频矩阵，再计算TFIDF值



# 训练和预测一体
x=SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)


text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=10.9, kernel = 'linear',probability=True))])
text_clf = text_clf.fit(train_content, train_opinion)
predicted = text_clf.predict(test_content)
print('SVC',np.mean(predicted == test_opinion))


import pandas as pd
data=pd.read_excel("附件4.xlsx",usecols=[5]).values.tolist()
data=[x[0] for x in data]
predicted = text_clf.predict_proba(segmentWord(data))[:,1].tolist()


import matplotlib.pyplot as plt
y=predicted
x=[i for i in range(len(y))]
plt.plot(x,y)
plt.ylabel("因果关系强弱")
plt.title("因果关系图")
plt.show()

with open("因果关系.txt","w") as f:
    for i in predicted:
        f.write(str(i)+'\n')
    f.close()

