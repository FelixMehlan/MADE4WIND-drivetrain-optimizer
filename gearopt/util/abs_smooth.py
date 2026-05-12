"""
abs_smooth_jax.py
-----------------
Smooth absolute value function compatible with JAX autodiff and JIT.
"""

import jax.numpy as jnp

def abs_smooth(x, eps=1e-12):
    """
    Smooth approximation of abs(x) using sqrt(x^2 + eps).

    Parameters
    ----------
    x : float, array, or JAX tracer
        Input value(s)
    eps : float
        Small smoothing term to avoid nondifferentiability at zero.

    Returns
    -------
    y : jnp.ndarray
        Smooth absolute value of x
    """
    return jnp.sqrt(x * x + eps)
