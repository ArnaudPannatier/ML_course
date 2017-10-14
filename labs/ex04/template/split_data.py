# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    random_range = np.random.permutation(np.arange(0,len(x)))
    mid_point = np.int_(len(x)*ratio)
    
    
    train_data = (x[range(mid_point)],y[range(mid_point)])
    test_data = (x[range(mid_point, len(x))],y[range(mid_point, len(x))])
    
    return (train_data, test_data)
    
