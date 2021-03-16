import numpy as np
# regularization functions used to deal with numerical demons of seepage
def _regularize_G(u, reg_factor):
    """Smooths transition of step function with an exponential.

    0<=u<=1.
    """
    return np.exp(-(1 - u) / reg_factor)


def _regularize_R(u):
    """ramp function on u."""
    return u * np.greater_equal(u, 0)
