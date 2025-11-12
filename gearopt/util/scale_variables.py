import numpy as np

def scale_variables(x, lb, ub):
    """
    Scale variables from physical range [lb, ub] to normalized [0, 1].

    Parameters
    ----------
    x : array_like
        Original variable(s).
    lb : array_like
        Lower bounds.
    ub : array_like
        Upper bounds.

    Returns
    -------
    x_scaled : ndarray
        Scaled variable(s) in [0, 1].
    lb_s : ndarray
        Lower bounds (all zeros).
    ub_s : ndarray
        Upper bounds (ones or zero for fixed variables).
    scale : ndarray
        Scaling factors (ub - lb), with zeros replaced by 1.
    """
    x = np.asarray(x).reshape(-1)
    lb = np.asarray(lb).reshape(-1)
    ub = np.asarray(ub).reshape(-1)
    scale = ub - lb

    # Prevent divide-by-zero (if upper = lower)
    idx = (scale == 0)
    scale[idx] = 1.0

    x_scaled = (x - lb) / scale

    lb_s = np.zeros_like(lb)
    ub_s = np.ones_like(ub)
    ub_s[idx] = 0.0

    return x_scaled, lb_s, ub_s, scale
