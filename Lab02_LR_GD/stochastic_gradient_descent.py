# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(y)
    
    return grad, err


def stochastic_gradient_descent(y, tx, initial_w, batch_size, num_batches, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    conv_epsilon = 0.00001
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            grad, err = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
        
        ws.append(np.copy(w))
        losses.append(loss)
        
        if np.linalg.norm(ws[-1]-ws[-2])/len(w)<conv_epsilon: break
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
	

	
def minibatch_stochastic_gradient_descent(y, tx, initial_w, batch_size, num_batches, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    conv_epsilon = 0.00001
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, num_batches=num_batches):
            grad, err = compute_gradient(minibatch_y, minibatch_tx, w)
        
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)
        ws.append(np.copy(w))
        losses.append(loss)
        
        if np.linalg.norm(ws[-1]-ws[-2])/len(w)<conv_epsilon: break
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
