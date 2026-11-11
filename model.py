"""
Bayesian and Robust Logistic Regression from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - sigmoid
import numpy as np


def sigmoid(a):
    # TODO: Return 1 / (1 + exp(-a)) elementwise, using exp(a) / (1 + exp(a)) where a < 0
    #       so that no exponent you evaluate is positive.
    a = np.asarray(a, dtype=float)
    
    # Create an empty output array of the same shape
    out = np.empty_like(a)
    
    # Create boolean masks
    pos_mask = a >= 0
    neg_mask = ~pos_mask
    
    # Evaluate each branch only on its respective elements
    out[pos_mask] = 1.0 / (1.0 + np.exp(-a[pos_mask]))
    out[neg_mask] = np.exp(a[neg_mask]) / (1.0 + np.exp(a[neg_mask]))
    
    return out

