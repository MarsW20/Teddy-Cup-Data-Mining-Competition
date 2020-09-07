import math
import random
import numpy as np

def ownhash(word, bits):
    if word == '':
        ret = [0] * bits
        for i in range(1, bits):
            ret[i] = random.randint(-1, 1)
        return ret
    else:
        x = ord(word[0]) << 7
        m = 1000000007
        mask = (1 << bits) - 1
        for ch in word:
            x = ((x * m) ^ ord(ch)) & mask
        ret = [0] * bits
        for i in range(bits):
            ret[i] = 1 if x & (1 << i) else -1
        return ret

def simhash(dataSet, bits=32):
    hashDict = {}
    dfDict = {}
    idfDict = {}
    hashedData = []

    random.seed()
    n = len(dataSet)
    for sentence in dataSet:
        sentSet = set(sentence)
        for word in sentSet:
            if hashDict.get(word, -1) == -1:
                hashDict[word] = ownhash(word, bits)
                dfDict[word] = 1
            else:
                dfDict[word] += 1

    for key in dfDict.keys():
        idfDict[key] = math.log(n / dfDict[key])

    for i, sentence in enumerate(dataSet):
        hashedSentence = [0] * bits
        for word in sentence:
            hashedSentence = [(hashedSentence[j] + hashDict[word][j] * idfDict[word]) for j in range(bits)]
        hashedSentence = [1 if j > 0 else -1 for j in hashedSentence]
        hashedData.append(hashedSentence)
    return hashedData


if __name__ == '__main__':
    data = [['a','b','c'], ['a','b','c'], ['a','d'], ['a','d'], ['c','d']]
    for i, sentence in enumerate(data):
        data[i] += (20 - len(sentence)) * ['']
    hashedData = simhash(data, 32)
    print(len(hashedData))

