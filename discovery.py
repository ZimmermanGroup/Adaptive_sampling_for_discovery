"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Mar 9, 2022
Record  :    Base class for dicovery problem
"""

import numpy as np
import scipy.linalg as la
import numpy as np
import scipy as sc
import scipy.stats

import pdb



# data generator
def gen_low_rank(m1, m2, r, shape = 1.0, scale = 1.0):
    gamma = np.random.gamma(shape = shape, scale = scale, size = 1)
    U = np.random.randn(m1 * r).reshape((m1, r)) * gamma
    V = np.random.randn(m2 * r).reshape((m2, r)) * gamma
    return U @ V.transpose()

M = gen_low_rank(10, 10, 3)
