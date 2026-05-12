import jax.numpy as jnp

def safe_sqrt(x, epsv=1e-9):
    """
    JAX-safe square root: clamps input to a minimum value to avoid
    sqrt of negative numbers or zero. Fully differentiable.

    Parameters
    ----------
    x : array-like, scalar, or JAX tracer
        Input value(s).
    epsv : float
        Minimum positive value for numerical stability.

    Returns
    -------
    y : jnp.ndarray
        sqrt(max(x + epsv, epsv))
    """
    return jnp.sqrt(jnp.maximum(x + epsv, epsv))
