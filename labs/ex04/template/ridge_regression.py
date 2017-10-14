# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N = len(y)
    lenX = len(tx.T)
    
    wStar = np.linalg.inv(tx.T.dot(tx)+2*N*lambda_*np.eye(lenX)).dot(tx.T).dot(y);
    
    #Calcul of the error by MSE
    MSE = compute_mse(y, tx, wStar)
    
    return (MSE, wStar)
