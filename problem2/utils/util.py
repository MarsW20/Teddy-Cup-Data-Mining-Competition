import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
import re

def stop_words():
    stopwords = []
    f1 = open("../data/baidu_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("../data/cn_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("../data/hit_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("../data/scu_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    return stopwords

def load_sents():
    sents=[]
    f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[2,7])
    f2["merge"]=f2["留言主题"]+f2["摘要"]
    for sent in f2["merge"].values.tolist():
        sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
        sent = re.sub(u",{2,}", ',', sent)[:-1]
        sents.append(sent)
    return sents

def load_init():
    sents=[]
    f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[2,4])
    f2["merge"]=f2["留言主题"]+f2["留言详情"]
    for sent in f2["merge"].values.tolist():
        sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
        sent = re.sub(u",{2,}", ',', sent)[:-1]
        sents.append(sent)
    return sents

def load():
    corpus = []
    f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[5,6]).values
    return np.sum(f2,1)

def load_corpus():
    corpus = []
    f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[2])

    for sent in f2.values.tolist():
        corpus.append(sent[0])
    return corpus

def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


def k_means_opts(parser):
    group=parser.add_argument_group('Model- Attention')
    group.add_argument("--n_components", type=int,default = 400, help = "利用潜在语义分析对文本数据进行预处理")

    group.add_argument("--n_clusters", type=int, default=200, help="利用潜在语义分析对文本数据进行预处理")

    group.add_argument("--no-minibatch",
                  action="store_false", dest="minibatch", default=False,
                  help="使用一般的K-means算法 (使用batch模式).")
    group.add_argument("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="禁用逆向文档频率特征加权。")
    group.add_argument("--use-hashing",
                  action="store_false", default=True,
                  help="使用Hashing特征向量")
    group.add_argument("--n-features", type=int, default=40000,
                  help="从文本中提取的最大特征（维度）数量。")
    group.add_argument("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="在K-means算法中打印进度报告。")


def save_cluster(label_prediction,data,n_clusters):
    init=load()
    rank=[0 for i in range(n_clusters)]
    file=pd.read_excel("../data/附件3.xlsx",usecols=[3]).values
    tim,tim_format=calcu_tim(file,label_prediction,n_clusters)
    sents = [[] for i in range(n_clusters)]
    for ind, i in enumerate(label_prediction):
        sents[i].append(data[ind])
        rank[i]+=init[ind]
    gen_answer={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}

    res = np.array([(0.5*len(x)+0.5*rank[ind])*50/tim[ind] for ind,x in enumerate(sents)])
    pnt=res.copy()
    res = np.argsort(-res)[:8]
    print("排名前八的热度分别为:")
    for ind,i in enumerate(res):
        print(pnt[i],end=',')
        if ind<8:   gen_answer[ind].append(pnt[i])
    print()

    for lab,i in enumerate(res):
        for ind,j in enumerate(label_prediction):
            if j==i:
                gen_answer[lab].append(ind)
    """第一个数字是热度，剩下的数字是属于该类别的索引号"""
    import json
    f = open("../data/answer{}.json".format(np.random.choice([i for i in range(10000)])), "w", encoding="utf-8")
    json.dump(gen_answer, f)

    new=[sents[i] for i in res]
    sents=new

    f=open("cluster.txt","w")
    for i in range(8):
        f.write("类别号:"+str(i)+'\n')
        f.write("类别大小:"+str(len(sents[i]))+'\n')
        f.write("热度:"+str(pnt[res[i]])+'\n')
        f.write("时间范围:")
        f.write(str(tim_format[res[i]][0][0])+"/"+str(tim_format[res[i]][0][1])+"/"+str(tim_format[res[i]][0][2]))
        f.write(
            "---"+str(tim_format[res[i]][1][0]) + "/" + str(tim_format[res[i]][1][1]) + "/" + str(tim_format[res[i]][1][2]))
        f.write('\n')
        for line in sents[i]:
            f.write(line+'\n')
        f.write('\n\n\n\n')
    return res

def gen_ans(res,labels):
    res=res[:5]
    f=pd.read_excel("../data/附件3.xlsx").values
    ans=[]
    for i in range(len(res)):
        for ind,lab in enumerate(labels):
            if lab==res[i]:
                ans.append([i+1,f[ind,0],f[ind,1],f[ind,2],
                            f[ind,3],f[ind,4],f[ind,6],f[ind,5]])
    ans = pd.DataFrame(np.array(ans),
                       columns=["问题ID", "留言编号", "留言用户", "留言主题",
                                "留言时间", "留言详情", "点赞数", "反对数"])
    ans.to_excel("../data/ans.xlsx",index=None)



def load_dense_drop_repeat(path="/Users/mac/Downloads/sgns.sogounews.bigram-char"):
    import codecs
    vocab = {}
    with codecs.open(path, "r", "utf-8") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                continue
            vec = line.strip().split()
            for i in range(len(vec) - 300):
                if not vocab.__contains__(vec[i]):
                    vocab[vec[i]] = [float(x) for x in vec[-300:]]

    return vocab


def del_stopwords(sents):
    stopwords = []
    f1 = open("../data/baidu_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("../data/cn_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("../data/hit_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())
    f1 = open("../data/scu_stopwords.txt")
    for word in f1.readlines():
        stopwords.append(word.strip())

    sentences = []
    for sent in sents:
        line = []
        for word in sent:
            if word not in stopwords:
                line.append(word)
        sentences.append(line)
    return sentences


colors={'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

tmp_colors={
'mediumblue':           '#0000CD',
'red':                  '#FF0000',
#'lightcoral':           '#F08080',
'purple':               '#800080',
'palegreen':            '#98FB98',
'slategray':            '#708090',
'black':                '#000000',
'crimson':              '#DC143C',
'dodgerblue':           '#1E90FF'
}

def gen_time():
    import re
    f=pd.read_excel("../data/ans.xlsx",usecols=[0,4]).values
    m,n=f.shape
    ans=[]
    for i in range(1,6):
        tim=[]
        for j in range(m):
            if f[j,0]==i:
                if '/' in str(f[j,1]):
                    t=list(map(int,re.split(r'/| ',str(f[j,1]))[:3]))
                    tim.append(t)
                else:
                    t = list(map(int, re.split(r'-| ',str(f[j,1]))[-4:-1]))
                    tim.append(t)
        an1=sorted(tim,key=lambda x: (x[0], x[1], x[2]))[0]
        an2=sorted(tim,key=lambda x: (x[0], x[1], x[2]),reverse=True)[0]

        ans.append([an1,an2])
    print(ans)

def calcu_tim(file,labels,n_clusters):
    import re
    m, n = file.shape
    ans = []
    tim_format=[]
    for i in range(n_clusters):
        tim = []
        for j in range(m):
            if labels[j]==i:
                if '/' in str(file[j,0]):
                    t = list(map(int, re.split(r'/| ', str(file[j, 0]))[:3]))
                    if len(t)==0:
                        print(file[j,0])
                    tim.append(t)
                else:
                    t = list(map(int, re.split(r'-| ', str(file[j, 0]))[-4:-1]))
                    if len(t)==0:
                        print(file[j,0])
                    tim.append(t)
        try:
            an1 = sorted(tim, key=lambda x: (x[0], x[1], x[2]))[0]
            an2 = sorted(tim, key=lambda x: (x[0], x[1], x[2]), reverse=True)[0]
        except:
            print("######处理时间格式报错######")
            print("由于数据集的时间格式并没有规范的定义，因此如果报错，则需要修改utils/util.py文件中calcu_tim函数")

        tim_format.append([an1,an2])
        ans.append(calcu_days(an1,an2))
    return ans,tim_format

def calcu_days(tim1,tim2):
    import datetime
    d1 = datetime.datetime(tim1[0], tim1[1], tim1[2])  # 第一个日期
    d2 = datetime.datetime(tim2[0], tim2[1], tim2[2])  # 第二个日期
    interval = d2 - d1  # 两日期差距
    return interval.days  # 具体的天数


def gen_finall_answer():
    import json
    length=pd.read_excel("../data/附件3.xlsx",usecols=[0]).values.shape[0]
    ans=json.load(open("../data/answer.json","r"))

    keys=[key for key,val in sorted(ans.items(),key=lambda x:x[1][0],reverse=True)]
    #首先去重
    now=[]
    for key in keys:
        for val in ans[key][1:]:
            if val in now:
                ans[key].remove(val)
            else:
                now.append(val)

    #生成表格2
    label_pred=[0 for i in range(length)]
    print("热度为:")
    for key in keys:
        print(ans[key][0],end=',')
        for val in ans[key][1:]:
            label_pred[val]=key
    gen_ans(keys,label_pred)
    print('\n')
    print("时间为:")
    gen_time()


if __name__=='__main__':
    """
        要想运行这个代码，必须手动生成answer.json文件
        该文件是多次运行kmeans得出的
    """
    gen_finall_answer()


