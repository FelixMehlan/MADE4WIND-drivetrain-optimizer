import jax.numpy as jnp

def smoothmax(a, b, k):
    """
    JAX-differentiable smooth approximation of max(a, b).

    Parameters
    ----------
    a, b : scalar or array-like
        Input values (JAX tracers allowed).
    k : float
        Smoothness parameter. Larger k → sharper approximation of max(a,b).

    Returns
    -------
    y : jnp.ndarray
        Smooth maximum of a and b.
    """

    return 0.5 * (a + b) + 0.5 * jnp.sqrt((a - b)**2 + (1.0 / k)**2)
