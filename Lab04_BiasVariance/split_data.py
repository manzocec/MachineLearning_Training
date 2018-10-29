# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    np.random.seed(seed)
    N = np.shape(x)[0]
    idx = np.random.permutation(N)
    tr_size = int(np.floor(N*ratio))
    idx_tr = idx[:tr_size]
    idx_te = idx[tr_size:]
    
    x_tr = x[idx_tr]
    x_te = x[idx_te]
    y_tr = y[idx_tr]
    y_te = y[idx_te]
    
    return x_tr, x_te, y_tr, y_te