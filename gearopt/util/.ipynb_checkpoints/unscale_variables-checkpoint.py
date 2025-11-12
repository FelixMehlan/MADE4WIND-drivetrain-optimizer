import numpy as np

def unscale_variables(x_scaled, lb, ub):
    """
    Convert scaled variables (0–1 space) back to physical values.

    Parameters
    ----------
    x_scaled : array_like
        Scaled variable(s) in [0, 1].
    lb : array_like
        Lower bounds.
    ub : array_like
        Upper bounds.

    Returns
    -------
    x : ndarray
        Unscaled variable(s) in the original range.
    """
    x_scaled = np.asarray(x_scaled).reshape(-1)
    lb = np.asarray(lb).reshape(-1)
    ub = np.asarray(ub).reshape(-1)
    scale = ub - lb
    return x_scaled * scale + lb
