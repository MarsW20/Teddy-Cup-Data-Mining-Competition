##
## blog list:
## https://blog.csdn.net/yujianmin1990/article/details/48420483
## https://blog.csdn.net/qrlhl/article/details/78066994
##
## code list:
## https://blog.csdn.net/u012369559/article/details/79968633
##
import numpy as np
import math
import copy

# ways to get the nearest k neighbor.
def knn(x, dataMat, kNum):
    n = dataMat.shape[0]
    diffMat = dataMat.copy()
    for i in range(n):
        diffMat[i,:] -= x
    sumVec = (diffMat ** 2).sum(axis=1)
    disVec = sumVec ** 0.5
    sortedVec = disVec.argsort()
    return sortedVec[:kNum]

def laplacian_eigenmap(dataMat, kNum, aimDem, tNum=2.0):
    n, m = np.shape(dataMat)
    W = np.zeros((n, n))
    D = np.zeros((n, n))
    # get the mat W ,D and L
    for i in range(n):
        kIndex = knn(dataMat[i,:], dataMat, kNum)
        for j in kIndex:
            diffVec = dataMat[i,:] - dataMat[j,:]
            W[i,j] = math.exp(-(diffVec ** 2).sum() / tNum)
            D[i,i] += W[i,j]
    L = D - W
    invD = np.linalg.inv(D)
    A = np.dot(invD, L)
    lambda_, eigenVec = np.linalg.eig(A) # eigenVec is row vector
    
    # get the p(aimDem) min lambda which is large than 0.
    lambdaIndex = lambda_.argsort()
    countNum = 0
    lambda__ = []
    for i in lambdaIndex:
        if lambda_[i] > 1e-5:
            countNum += 1
            lambda__.append(i)
        if countNum == aimDem:
            break
    eigenVec__ = np.zeros((n, aimDem))

    for i in range(aimDem):
        eigenVec__[:,i] = eigenVec[:,lambda__[i]]
    return lambda__, eigenVec__


if __name__ == '__main__':
    dataMat = np.array([[1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1], [1, -1, -1]])
    lambda_, eigenVec_ = laplacian_eigenmap(dataMat, 1, 2)
    print(lambda_, eigenVec_)
