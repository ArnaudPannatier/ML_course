# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs
from costs import *
from plots import *
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

def grid_search_and_compare(y,tx, mean_x, std_x,height, weight, LSw0, LSw1):
    # Generate the grid of parameters to be swept
    grid_w0, grid_w1 = generate_w(num_intervals=100)

    # Start the grid search
    start_time = datetime.datetime.now()
    grid_losses = grid_search(y, tx, grid_w0, grid_w1)

    # Select the best combinaison
    loss_star, w0_star, w1_star = get_best_parameters(grid_w0, grid_w1, grid_losses)
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    # Print the results
    print("Grid Search: loss*={l}, w0*={w0}, w1*={w1}, execution time={t:.3f} seconds".format(
          l=loss_star, w0=w0_star, w1=w1_star, t=execution_time))

    # Plot the results
    fig = compare_visualization(grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight, LSw0, LSw1)
    fig.set_size_inches(10.0,6.0)
    fig.savefig("grid_plot")  # Optional saving
