import numpy as np

def softclip(x, a, b, c):
    """
    Smooth clipping using a softplus/softminus formulation.
    Provides a differentiable alternative to np.clip(x, a, b).

    Parameters
    ----------
    x : float or array_like
        Input value(s).
    a, b : float
        Lower and upper clipping limits.
    c : float
        Sharpness parameter. Larger values -> sharper transition.

    Returns
    -------
    y : ndarray
        Softly clipped version of x.
    """
    x = np.asarray(x, dtype=float)
    a = float(a)
    b = float(b)
    c = float(c) / ((b - a) / 2)

    def softplus(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)

    def softminus(z):
        return -softplus(-z)

    v = x.copy()
    v = v - softminus(c * (x - a)) / c
    v = v - softplus(c * (x - b)) / c
    return v
