import numpy as np
def safe_asin(x, eps_val=1e-9):
    """
    Safe asin: clamps input to [-1, 1] smoothly.
    """
    x = np.asarray(x)
    x_clamped = np.clip(x, -1 + eps_val, 1 - eps_val)
    return np.arcsin(x_clamped)
