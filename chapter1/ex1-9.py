#!/usr/bin/env python
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def buildDataMatrix(x, order):
    # size of matrix is size(x)*(order+1)
    X = np.zeros( (x.size, order+1), dtype=float, order='C')
    for i in range(0, order + 1):
        X[:,i] = np.power(x,i)

    return X


def generateSyntheticDataset(limits, N, order):
    uniform = np.array( np.sort(np.random.rand(N)))
    x = (limits[1] - limits[0]) * uniform + limits[0]
    X = buildDataMatrix(x, order)

    return X

def computeWeights(X, t, order):
    # weights are computed as
    # inverse(X'X) X't
    xsplice = np.array(X[:,0:order+1])
    xspliceT = np.transpose(xsplice)
    xspliceInv = np.linalg.inv(np.matmul(xspliceT,xsplice))
    w = np.dot(np.matmul(xspliceInv, xspliceT), t)

    return w

limits = [-5, 5]
weights = [1, -2, 0.5]
max_order = 4
size = 200
X = generateSyntheticDataset(limits, size, max_order)

t = np.dot(X[:,0:3],weights) + np.random.rand(size)

linear_weights = computeWeights(X, t, 1)
quadratic_weights = computeWeights(X, t, 2)
fourth_order_weights = computeWeights(X, t, 4)

plotx = np.linspace(-5,5,200)

plotX = np.zeros((size,max_order+1), dtype = float, order = 'C')

for i in range(0,len(weights)):
    plotX[:,i] = np.power(plotx,i)

plt.plot(X[:,1],t,'bo')
plt.plot(plotx,np.dot(plotX[:,0:2],linear_weights),'r-')
plt.plot(plotx,np.dot(plotX[:,0:3],quadratic_weights),'g-')
plt.plot(plotx,np.dot(plotX[:,0:5],fourth_order_weights),'k-')
plt.show()

