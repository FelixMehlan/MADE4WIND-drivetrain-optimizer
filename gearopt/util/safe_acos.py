import jax.numpy as jnp

def safe_acos(x, eps_val=1e-9):
    """
    JAX-safe acos: clamps input to [-1 + eps, 1 - eps] using jnp.clip
    to avoid NaNs and maintain differentiability.

    Parameters
    ----------
    x : array-like or scalar (JAX tracer ok)
    eps_val : float
        Safety margin to avoid evaluating acos() at exactly ±1.

    Returns
    -------
    y : jnp.ndarray
        arccos(x_clamped)
    """
    x_clamped = jnp.clip(x, -1.0 + eps_val, 1.0 - eps_val)
    return jnp.arccos(x_clamped)
