import jax.numpy as jnp

def unscale_variables(x_scaled, lb, ub):
    """
    Convert scaled variables (0–1 space) back to physical values.

    Parameters
    ----------
    x_scaled : array-like (JAX tracers allowed)
        Scaled variable(s) in [0, 1].
    lb : array-like
        Lower bounds.
    ub : array-like
        Upper bounds.

    Returns
    -------
    x : jnp.ndarray
        Unscaled variable(s) in the original physical range.
    """

    x_scaled = jnp.asarray(x_scaled).reshape(-1)
    lb       = jnp.asarray(lb).reshape(-1)
    ub       = jnp.asarray(ub).reshape(-1)

    scale = ub - lb

    return x_scaled * scale + lb
