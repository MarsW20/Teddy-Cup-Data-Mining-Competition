#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import re
import datetime
import pandas as pd
from segmenter import segment

def load_sents():
    sents = []
    f2 = pd.read_excel("../data/附件3.xlsx", header=0, usecols=[4])
    for sent in f2.values.tolist():
        sent=str(sent)
        sent = re.sub(u"[^\u4e00-\u9fa5a-zA-Z0-9]", ',', sent)
        sent = re.sub(u",{2,}", ',', sent)[:-1]
        sents.append(sent)
    return sents


def main():   # idf generator
    sents=load_sents()
    sents=[segment(x) for x in sents]
    ignored = {'', ' ', '', '。', '：', '，', '）', '（', '！', '?', '”', '“'}
    id_freq = {}
    i = 0
    for doc in sents:
        doc = set(x for x in doc if x not in ignored)
        for x in doc:
            id_freq[x] = id_freq.get(x, 0) + 1
        if i % 1000 == 0:
            print('Documents processed: ', i, ', time: ',
                  datetime.datetime.now())
        i += 1


    with open("idf.txt", 'w', encoding='utf-8') as f:
        for key, value in id_freq.items():
            f.write(key + ' ' + str(math.log(i / value, 2)) + '\n')


if __name__ == "__main__":
   main()
