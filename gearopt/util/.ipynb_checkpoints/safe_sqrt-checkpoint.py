import numpy as np

def safe_sqrt(x, epsv=1e-9):
    """
    Safe square root: prevents negative or zero inputs.
    """
    x = np.asarray(x)
    return np.sqrt(np.maximum(x + epsv, epsv))
