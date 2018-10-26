# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones(len(x))
    phi = np.vstack((phi, [x**(j+1) for j in range(degree)]))
    
    return phi.T
