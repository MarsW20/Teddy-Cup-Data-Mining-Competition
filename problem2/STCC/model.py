import torch
import math
import torch.nn as nn
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import simhash
from  clustering import *
import simhash
import sys
sys.path.append('/Users/mac/Documents/programming/泰迪杯/Text_cluster')

import utils.util as data_util
import jieba

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.k_max_number = 5

        #CNN1
        self.conv_kernel_size1 =[5,5]
        self.pad_0_direction1 = math.ceil((self.conv_kernel_size1[0] - 1) / 2)
        self.pad_1_direction1= math.ceil((self.conv_kernel_size1[1] - 1) / 2)
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=self.conv_kernel_size1,
                                    padding=(self.pad_0_direction1, self.pad_1_direction1))
        self.fold1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        #CNN2
        self.conv_kernel_size2 =[5,5]
        self.pad_0_direction2 = math.ceil((self.conv_kernel_size2[0] - 1) / 2)
        self.pad_1_direction2= math.ceil((self.conv_kernel_size2[1] - 1) / 2)
        self.conv_layer2 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=self.conv_kernel_size2,
                                    padding=(self.pad_0_direction2, self.pad_1_direction2))
        self.fold2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        #output
        self.dropout = nn.Dropout(0.5)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64,32)

    def forward(self,inp):

        #CNN1
        conved = self.conv_layer1(inp)  #[30, 12, 20, 100]
        conved = self.fold1(conved)  # [30, 12, 20, 50]
        k_maxed = torch.tanh(torch.topk(conved, self.k_max_number, dim=2, largest=True)[0]) #[30, 12, 5, 50]


        #CNN2
        conved = self.conv_layer2(k_maxed)#[30, 8, 5, 50]
        conved = self.fold2(conved) #[30, 8, 5, 25]
        out = torch.tanh(torch.topk(conved, k=2, dim=2, largest=True)[0])


        out = self.dropout(self.flatten(out)) #[30,400]

        # print(out.size())

        y = F.relu(self.fc1(out))
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


def train(dic, Corpus):
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
    input = input.reshape(-1, 200 * 300)  # [batch_size,2000]
    output = np.array(B)  # [batch_size,32]
    data = np.concatenate((input, output), axis=1)

    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print('Start Traning Cnn')

    for epoch in range(8000):  # loop over the dataset multiple times
        generator = batch_generator(data, 30)
        for inputs, labels in generator:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, h = net(torch.Tensor(inputs.reshape(-1, 1, 200, 300)))
            loss = criterion(outputs, torch.Tensor(labels.reshape(-1, 32)))
            loss.backward()
            optimizer.step()

            # print statistics
            if epoch % 100 == 0:
                print('loss:{}'.format(loss))

    print('Finished Training')
    torch.save(net, "cnn_model.pkl")

import data_util
import jieba
if __name__ == '__main__':
    dic = data_util.load_dense_drop_repeat("/Users/mac/Downloads/sgns.wiki.word")
    #corpus = data_util.load_corpus()
    corpus=data_util.load_sents()
    # cut and supply the corpus
    Corpus = corpus
    CorpusLength = len(Corpus)
    for i, line in enumerate(Corpus):
        Corpus[i] = jieba.lcut(Corpus[i])
        Corpus[i] = data_util.del_stopwords([Corpus[i]])[0]
        length = len(Corpus[i])
        if length > 200:
            Corpus[i] = Corpus[i][:200]
        if length < 200:
            Corpus[i] += (200 - length) * ['']
        if i % 10000 == 0:
            print('(%d %% %d) sentences has been cutted' % (i, CorpusLength))

    # print(Corpus)

    random.seed()
    random.shuffle(Corpus)
    train(dic, Corpus)

