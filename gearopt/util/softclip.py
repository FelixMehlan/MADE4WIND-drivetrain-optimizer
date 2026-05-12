import jax.numpy as jnp

def softclip(x, a, b, c):
    """
    JAX-differentiable smooth clipping (soft alternative to jnp.clip).

    Parameters
    ----------
    x : array-like or scalar (JAX tracers allowed)
        Input values.
    a, b : float
        Lower and upper soft clipping limits.
    c : float
        Sharpness parameter. Larger -> sharper transitions.

    Returns
    -------
    y : jnp.ndarray
        Smoothly clipped x.
    """

    # Normalize sharpness relative to interval width (matches original)
    c = c / ((b - a) / 2)

    # JAX-stable softplus implementation
    def softplus(z):
        return jnp.log1p(jnp.exp(-jnp.abs(z))) + jnp.maximum(z, 0)

    def softminus(z):
        return -softplus(-z)

    # Use pure JAX expressions, no copies
    v = x
    v = v - softminus(c * (x - a)) / c
    v = v - softplus(c * (x - b)) / c

    return v
