import numpy as np

def smoothmax(a, b, k):
    """
    Smooth approximation of max(a, b) using a differentiable function.

    Parameters
    ----------
    a, b : float or array_like
        Input values.
    k : float
        Smoothness parameter (larger k → sharper transition).

    Returns
    -------
    y : ndarray
        Smooth maximum of a and b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return 0.5 * (a + b) + 0.5 * np.sqrt((a - b) ** 2 + (1.0 / k) ** 2)
