import numpy as np
def safe_div(a, b, eps_val=1e-12):
    """
    Safe division: smooth, sign-preserving division avoiding zero denominators.

    Parameters
    ----------
    a : array_like
        Numerator.
    b : array_like
        Denominator.
    eps_val : float
        Smoothing tolerance to avoid division by zero.

    Returns
    -------
    y : ndarray
        Smooth approximation of a / b.
    """
    a, b = np.asarray(a), np.asarray(b)
    absb = np.sqrt(b**2 + eps_val**2)
    return a / absb * np.sign(b)
