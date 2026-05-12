import jax.numpy as jnp

def safe_div(a, b, eps_val=1e-12):
    """
    JAX-safe smooth division: a / b with sign-preserving smoothing
    to avoid division by zero.

    Parameters
    ----------
    a : array-like or scalar
        Numerator.
    b : array-like or scalar
        Denominator.
    eps_val : float
        Smoothing constant to avoid non-differentiability.

    Returns
    -------
    y : jnp.ndarray
        Smooth approximation of a / b.
    """
    # smooth absolute value of denominator
    abs_b = jnp.sqrt(b * b + eps_val * eps_val)

    # preserve the sign like original implementation
    sign_b = jnp.sign(b)

    return a / abs_b * sign_b
