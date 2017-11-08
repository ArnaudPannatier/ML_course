# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    to_invert = tx.T.dot(tx)
    mul_y = tx.T.dot(y)
    w_star = np.linalg.solve(to_invert, mul_y)
    # MSE
    const_part = 1/(2*y.size)
    e = (y - (tx.dot(w_star)))
    e_squared = e.T.dot(e)
    return const_part * e_squared, w_star
