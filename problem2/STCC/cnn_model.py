import numpy as np
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  clustering import *
import simhash
import sys
sys.path.append('/Users/mac/Documents/programming/泰迪杯/Text_cluster')

import utils.util as data_util
import jieba
from tensorboardX import SummaryWriter
#from test import *


#logger=Logger("logs/")
writer=SummaryWriter("STCC")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 8, (3, 300))
        self.pool = nn.MaxPool2d(1, 2)
        self.fc1 = nn.Linear(8 * 9 * 1, 128)
        self.fc2 = nn.Linear(128, 32)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 8 * 9 * 1)
        y = F.relu(self.fc1(self.dropout(x)))
        x = torch.sigmoid(self.fc2(y))
        return x, y

def batch_generator(mat, batch_size):
    mat = copy.copy(mat)
    n_batches = mat.shape[0] // batch_size
    mat = mat[:batch_size * n_batches,:]

    random.shuffle(mat)
    for n in range(0, mat.shape[0], batch_size):
        x = mat[n:n + batch_size,:6000]
        y = mat[n:n + batch_size,6000:]
        yield x, y

def train(dic,Corpus):
    corpus = Corpus  # remember to increse highly.
    sentences = corpus
    hashedSentences = simhash.simhash(sentences, 32)
    B = hashedSentences

    # you can try simhash directly, maybe it performs better.
    # hashedSentences = simhash.simhash(sentences, 128)
    # dataMat = np.array(hashedSentences)
    # lambda_, eigenVec_ = laplacian_eigenmap.laplacian_eigenmap(dataMat, 15, 32)
    # B=eigenVec_

    input = embedding(dic, corpus)
    input = np.array(input)
    input = input.reshape(-1, 20 * 300)  # [batch_size,2000]
    output = np.array(B)  # [batch_size,32]
    data = np.concatenate((input, output), axis=1)

    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Start Traning Cnn')

    inputs=0
    for epoch in range(800):  # loop over the dataset multiple times
        generator = batch_generator(data, 30)
        cnt=0
        loss_=0
        for inputs, labels in generator:
            cnt+=1
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs=torch.from_numpy(inputs).reshape(-1,1,20,300).float()
            labels=torch.from_numpy(labels).reshape(-1,32).float()
            # forward + backward + optimize
            outputs, h = net(inputs)
            loss = criterion(outputs, labels)
            loss_+=loss

            loss.backward()
            optimizer.step()

            # print statistics
            if epoch % 100 == 0:
                print('loss:{}'.format(loss))

        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag,value.data.cpu().numpy(),epoch+1)
            #logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
            #logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)


        writer.add_scalar("STCC/loss", loss_/cnt,epoch)
        cnt=0
        loss_=0

    print('Finished Training')
    torch.save(net,"cnn_model.pkl")
    writer.add_graph(net,input_to_model=(inputs,))
    writer.close()

if __name__=='__main__':
    dic = data_util.load_dense_drop_repeat("/Users/mac/Downloads/sgns.wiki.word")
    #corpus = data_util.load_corpus()
    corpus=data_util.load_corpus()
    #corpus=data_util.load_sents()
    # cut and supply the corpus
    Corpus = corpus
    CorpusLength = len(Corpus)
    for i, line in enumerate(Corpus):
        Corpus[i] = jieba.lcut(Corpus[i])
        Corpus[i] = data_util.del_stopwords([Corpus[i]])[0]
        length = len(Corpus[i])
        if length > 20:
            Corpus[i] = Corpus[i][:20]
        if length < 20:
            Corpus[i] += (20 - length) * ['']
        if i % 10000 == 0:
            print('(%d %% %d) sentences has been cutted' % (i, CorpusLength))

    # print(Corpus)

    random.seed()
    random.shuffle(Corpus)
    train(dic,Corpus)





        
