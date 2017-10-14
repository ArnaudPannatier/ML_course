# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import numpy.matlib

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    
    degree_range = np.arange(0,degree+1)
    degree_array = np.matlib.repmat(degree_range, len(x), 1)
    x_array = np.matlib.repmat(x, degree+1,1).T
    
    
    return(np.power(x_array,degree_array)) 
