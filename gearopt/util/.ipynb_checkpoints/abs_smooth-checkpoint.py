"""
abs_smooth.py
-------------
Smooth absolute value function with a nonzero slope everywhere except at zero.
Useful for numerical optimization to avoid nondifferentiability of |x|.
"""

import numpy as np

def abs_smooth(x, eps=1e-12):
    """
    Smooth approximation of abs(x) using sqrt(x^2 + eps).
    
    Parameters
    ----------
    x : float or array-like
        Input value(s)
    eps : float, optional
        Small positive number to smooth around zero (default 1e-12)
        
    Returns
    -------
    y : float or ndarray
        Smoothed absolute value of x
    """
    x = np.asarray(x)
    return np.sqrt(x**2 + eps)
