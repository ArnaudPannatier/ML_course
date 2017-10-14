# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    wStar = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y));
    
    #Calcul of the error by MSE
    MSE = compute_mse(y, tx, wStar)
    
    return (MSE, wStar)
