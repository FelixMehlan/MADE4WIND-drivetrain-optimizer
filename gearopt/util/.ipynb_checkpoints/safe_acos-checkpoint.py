import numpy as np

def safe_acos(x, eps_val=1e-9):
    """
    Safe acos: clamps input to [-1, 1] smoothly.
    """
    x = np.asarray(x)
    x_clamped = np.clip(x, -1 + eps_val, 1 - eps_val)
    return np.arccos(x_clamped)
