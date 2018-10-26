# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    A = tx.T.dot(tx) + 2*lamb*np.shape(tx)[0]*np.eye(np.shape(tx)[1])
    b = tx.T.dot(y)
    
    return np.linalg.solve(A, b)

