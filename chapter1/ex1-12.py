#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def generateIndependentData(limits,N):

    uniform = np.sort(np.random.rand(N))
    x = (limits[1] - limits[0]) * uniform + limits[0]
    return x

def generateDependentData(weights, x):

    t = np.zeros(x.size, dtype = float, order = 'C')

    for i in range(0, len(weights)):
        X = np.power(x,i+1)
        t = t + weights[i]*X

    tmp = t + 150*np.random.normal(size=t.size)
    return tmp.reshape((tmp.size,1))

def createFolds(datasetSize, folds):
    # we wish to partition the training data into K approximately equally sized sets
    foldSize = int(datasetSize/folds) # rounding down to nearest int
    sizes = np.tile(foldSize, folds)
    sizes[-1]= datasetSize + sizes[-1] - np.sum(sizes);

    return np.append([0], np.cumsum(sizes))

def leastSquares(X, t):
    # weights computed as
    # inv(X'X) X't
    XT = np.transpose(X)
    print(XT.shape)
    invX = np.linalg.inv(np.matmul(XT,X))
    print(invX.shape)

    return np.matmul(np.matmul(invX, XT), t)
def regularisedLeastSquares(X, t, a):
    # weights computed as
    # inv(X'X + NaI)X't
    XT = np.transpose(X)
    tmp = np.matmul(XT,X)
    reg = X.size * a * np.identity(tmp.shape[0])
    print(reg.shape)
    invX = np.linalg.inv(tmp + reg)
    tmp = np.matmul(invX, XT)

    return np.matmul(tmp, t)

def crossValidation(trainx, traint, testx, testt, numFolds, maxOrder):
    foldSizes = createFolds(trainx.size, numFolds)
    X = np.zeros((trainx.size, maxOrder + 1), dtype = float, order = 'C')
    testX = np.zeros((testx.size, maxOrder + 1), dtype = float, order = 'C')

    loss_cv = np.zeros((maxOrder,numFolds), dtype = float, order = 'C')
    loss_ind = np.zeros((maxOrder,numFolds), dtype = float, order = 'C')
    loss_train = np.zeros((maxOrder,numFolds), dtype = float, order = 'C')
    ave_loss_cv = np.zeros((maxOrder,1), dtype = float, order = 'C')
    ave_loss_ind = np.zeros((maxOrder,1), dtype = float, order = 'C')
    ave_loss_train = np.zeros((maxOrder,1), dtype = float, order = 'C')

    plt.plot(trainx, traint, 'bo')
    plt.plot(testx, testt, 'rx')

    for i in range(0, maxOrder):
        X[:,i] = np.power(trainx, i)
        testX[:,i] = np.power(testx, i)

    for i in range(0, maxOrder):
        for j in range(0, numFolds):
            foldX = X[foldSizes[j]:foldSizes[j+1],:i+1]
            trainX = X[:,:i+1]
            looX = np.delete(trainX, np.s_[foldSizes[j]:foldSizes[j+1]], axis = 0)
            foldt = traint[foldSizes[j]:foldSizes[j+1]]
            loot = np.delete(traint, np.s_[foldSizes[j]:foldSizes[j+1]], axis = 0)
            loot = loot.reshape(loot.size,1)

            w = regularisedLeastSquares(looX,loot, 0.0)
            #w = leastSquares(looX,loot)

            pred_fold = np.dot(foldX, w)
            diff_fold = pred_fold - foldt
            loss_cv[i,j] = np.mean(np.power(diff_fold, 2))

            pred_ind = np.dot(testX[:,:i+1], w)
            diff_ind = pred_ind - testt
            loss_ind[i,j] = np.mean(np.power(diff_ind, 2))

            pred_train = np.dot(looX, w)
            diff_train = pred_train - loot
            loss_train[i,j] = np.mean(np.power(diff_train, 2))

        ave_loss_cv[i] = np.mean(loss_cv[i,:])
        ave_loss_ind[i] = np.mean(loss_ind[i,:])
        ave_loss_train[i] = np.mean(loss_train[i,:])

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.plot(np.linspace(1,maxOrder,maxOrder),ave_loss_cv,'b-')

    ax2 = fig.add_subplot(132)
    ax2.plot(np.linspace(1,maxOrder,maxOrder),ave_loss_train,'b-')

    ax3 = fig.add_subplot(133)
    ax3.plot(np.linspace(1,maxOrder,maxOrder),ave_loss_ind,'b-')





# generating training data
limits = [0, 1]
weights = [1, 1, 1]
N = 6

x = generateIndependentData(limits, N)
t = generateDependentData(weights, x)
plt.plot(x,t,'rx')
t.reshape(t.size,1)

testx = np.linspace(0, 1, 100)
testt = generateDependentData(weights,testx)

a = [0, 1e-6, 0.01, 0.1]



# settings for cross validation
K = 10
maxOrder = 4

X = np.zeros((x.size, maxOrder + 1), dtype = float, order = 'C')
testX = np.zeros((testx.size, maxOrder + 1), dtype = float, order = 'C')
for i in range(0, maxOrder + 1):
    X[:,i] = np.power(x, i)
    testX[:,i] = np.power(testx, i)


for i in a:
    w = regularisedLeastSquares(X, t, i)
    plt.plot(testx, np.matmul(testX,w),'-')



#print(x)

#crossValidation(x, t, testx, testt, K, maxOrder)

#plt.plot(x,t,'bo')
#plt.plot(testx,testt,'rx')
plt.show()
