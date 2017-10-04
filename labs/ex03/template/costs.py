# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    # ***************************************************
    # compute loss by MSE
    # ***************************************************
    e = y-tx.dot(w)
    N = len(y)
    return 1/(2*N)*e.T.dot(e)