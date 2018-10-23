# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(y)
    
    return grad, err


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    conv_epsilon = 0.00001
    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)
        loss = compute_mse(err)
        w = w - gamma * grad
        
        ws.append(np.copy(w))
        losses.append(loss)
        if np.linalg.norm(ws[-1]-ws[-2])/len(w)<conv_epsilon: break
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
