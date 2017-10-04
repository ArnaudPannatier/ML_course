# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from costs import *
import datetime

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]

def grid_search(y, tx, w0, w1):
    losses = np.zeros((len(w0), len(w1)))
    
    # ***************************************************
    # Compute loss for each combination of w0 and w1.
    # ***************************************************
    for i,v0 in enumerate(w0, start=0):
        for j,v1 in enumerate(w1, start=0):
            w = np.array([v0,v1])
            losses[i][j] = compute_loss(y,tx,w)
    return losses


