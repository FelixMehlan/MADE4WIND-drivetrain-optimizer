import jax.numpy as jnp

def smoothmin2(a, b, k):
    """
    JAX-differentiable smooth approximation of min(a, b).

    Parameters
    ----------
    a, b : float or array-like (JAX tracers allowed)
        Input values.
    k : float
        Smoothness parameter. Larger k → sharper transition.

    Returns
    -------
    y : jnp.ndarray
        Smooth minimum of a and b.
    """
    return 0.5 * (a + b) - 0.5 * jnp.sqrt((a - b)**2 + (1.0 / k)**2)
