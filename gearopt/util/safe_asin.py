import jax.numpy as jnp

def safe_asin(x, eps_val=1e-9):
    """
    JAX-safe asin: clamps input to [-1 + eps, 1 - eps] to avoid NaNs
    and keep the function differentiable under JIT and autodiff.

    Parameters
    ----------
    x : array-like, scalar, or JAX tracer
    eps_val : float
        Safety margin from the domain boundaries.

    Returns
    -------
    y : jnp.ndarray
        arcsin(x_clamped)
    """
    x_clamped = jnp.clip(x, -1.0 + eps_val, 1.0 - eps_val)
    return jnp.arcsin(x_clamped)
