# coding: utf-8

from scipy.stats import vonmises
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.special import iv
from math import pi


"""
Code adapted from
http://nocotan.github.io/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92/2017/01/24/vmf-copy.html
"""

def pdf(x, k, myu):
    """
    @param x: input vector 
    @param k: concentration parameter vector
    @param myu: mean direction (center of the distribution)
    """

    #print(norm(myu))
    #print(myu / norm(myu) )
    #print(norm((myu / norm(myu) )))

    assert(np.linalg.norm(myu) == 1)
    d = x.ndim
    return _C(d, k) * np.exp(k*np.dot(myu,x))

def _C(d, k):
    """ 
    normalization constant 
    """

    return (k ** (d/2.-1)) / ((2*pi) ** (d/2.) * iv(d/2.-1, k))

kappa = np.array([3.99390425811, 3.99390425811, 3.99390425811])
data_vector = np.array([1, 1, 1]) / norm(np.array([1, 1, 1]))
mean_vector = np.array([1, 1, 1]) / norm(np.array([1, 1, 1]))
print(pdf(data_vector, kappa, mean_vector))
print(pdf(np.array([0.1, 1, 1]), kappa, np.array([1, 1, 1])))
"""
Is the likelihood, the returned value?
We need to estimate the mean vector, and the concentration parameter vector where ||myu||_2 = 1 (Frobenius norm)
"""
 
#fig, ax = plt.subplots(1, 1)
#kappa = 3.99390425811
#mean, var, skew, kurt = vonmises.stats(kappa, moments='mvsk')
#x = np.linspace(vonmises.ppf(0.01, kappa),
#              vonmises.ppf(0.99, kappa), 100)
#ax.plot(x, vonmises.pdf(x, kappa),
#         'r-', lw=5, alpha=0.6, label='vonmises pdf')
#plt.show()
