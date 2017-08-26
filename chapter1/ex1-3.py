#!/usr/bin/env python
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def leastSquaresFit(a,b):
    # let a be the independent and b the dependent var
    # a and be must be of the same length
    N = a.size
    m_ab = np.dot(a,b)/N
    m_aa = np.dot(a,a)/N
    m_a = np.sum(a)/N
    m_b = np.sum(b)/N

    w_1 = (m_ab - m_a*m_b)/(m_aa - m_a*m_a)
    w_0 = m_b - w_1*m_a

    return np.array([w_0,w_1])

def prediction(weights,independent):
    return weights[0]+weights[1]*independent

def findCrossover(weights_a, weights_b):
    year = 2012
    time_a = prediction(weights_a,year)
    time_b = prediction(weights_b,year)

    while ( time_a >= time_b ):
        year = year + 4
        time_a = prediction(weights_a,year)
        time_b = prediction(weights_b,year)

    print "At the %i Olympics the winning female 100m time will be %d s, %d s faster than the male time" % (
            year,time_b,time_b - time_a)
    print "female 100m time = % d" % time_a
    print "male 100m time = % d" % time_b



#   using scipy MATLAB data struct compatible load function
data = sio.loadmat('../data/olympics')

female100 = np.array(data['female100'])
male100 = np.array(data['male100'])

# years
x_f = np.array(female100[:,0])
x_m = np.array(male100[:,0])
# winning times
t_f = np.array(female100[:,1])
t_m = np.array(male100[:,1])

weights_f = leastSquaresFit(x_f,t_f)
weights_m = leastSquaresFit(x_m,t_m)

leastx_f = np.array([np.nanmin(x_f)-10, np.nanmax(x_f)+10])
leasty_f = np.array([weights_f[0] + weights_f[1]*np.nanmin(leastx_f), weights_f[0] + weights_f[1]*np.nanmax(leastx_f)])

leastx_m = np.array([np.nanmin(x_m)-10, np.nanmax(x_m)+10])
leasty_m = np.array([weights_m[0] + weights_m[1]*np.nanmin(leastx_m), weights_m[0] + weights_m[1]*np.nanmax(leastx_f)])

plt.plot(x_f,t_f,'bo')
plt.plot(x_m,t_m,'go')
plt.plot(leastx_f,leasty_f,'r-')
plt.plot(leastx_m,leasty_m,'k-')
plt.plot(2012,weights_f[0]+ weights_f[1]*2012,'gs')
plt.plot(2016,weights_f[0]+ weights_f[1]*2016,'gs')
plt.axis('auto')
plt.show()

findCrossover(weights_m,weights_f)
