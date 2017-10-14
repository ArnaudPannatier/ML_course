# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    N = len(y)
    e = y-tx.dot(w)
    return -1/N*tx.T.dot(e)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    
    # Define parameters to store w and the losses
    losses = []
    ws = [initial_w]
    w = initial_w

    for n_iter in range(max_iters):
        #Stochastic choice
        random_index = np.random.choice(range(len(y)), batch_size)
            
        yp = y[random_index]
        txp = tx[random_index]
        
        #Gradient Descent
        grad = compute_gradient(yp,txp,w)
        loss = compute_loss(yp,txp,w)
        
        w = w-gamma*grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws