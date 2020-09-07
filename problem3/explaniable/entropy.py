import numpy as np
import pandas as pd
import jieba
import re
import matplotlib.pyplot as plt

def load_corpus():
    corpus = []
    f2 = pd.read_excel("附件4.xlsx", header=0, usecols=[5])
    for sent in f2.values.tolist():
        sent=str(sent)
        sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
        sent = re.sub(u",{2,}", ',', sent)[:-1]
        corpus.append(jieba.lcut(sent))
    return corpus

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def paint():
    f=open("entropy.txt","r")
    y=[]
    for line in f.readlines():
        y.append(float(line))
    x=[i for i in range(len(y))]
    plt.plot(x,y)
    plt.ylabel("entropy")
    plt.title("Entropy")
    plt.show()

def gen_entropy():
    sents = load_corpus()
    entropy = []
    for sent in sents:
        entropy.append(calc_ent(np.array(sent)))
    with open("entropy.txt", "w") as f:
        for line in entropy:
            f.write(str(line) + '\n')
        f.close()

if __name__=='__main__':
    gen_entropy()
    paint()

