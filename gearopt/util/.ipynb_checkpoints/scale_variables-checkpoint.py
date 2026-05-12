import jax.numpy as jnp

def scale_variables(x, lb, ub):
    """
    JAX-differentiable variable scaling from [lb, ub] → [0, 1].

    Parameters
    ----------
    x : array-like
        Original variables.
    lb : array-like
        Lower bounds.
    ub : array-like
        Upper bounds.

    Returns
    -------
    x_scaled : jnp.ndarray
        Scaled variables in [0, 1].
    lb_s : jnp.ndarray
        Lower bounds in scaled space (zeros).
    ub_s : jnp.ndarray
        Upper bounds in scaled space (ones or zero for fixed vars).
    scale : jnp.ndarray
        Scaling factors (ub - lb), with zero replaced by 1.
    """

    x  = jnp.asarray(x).reshape(-1)
    lb = jnp.asarray(lb).reshape(-1)
    ub = jnp.asarray(ub).reshape(-1)

    scale = ub - lb

    # Replace zero scale with 1 (keeps them fixed but avoids division by zero)
    scale_safe = jnp.where(scale == 0, 1.0, scale)

    x_scaled = (x - lb) / scale_safe

    lb_s = jnp.zeros_like(lb)
    ub_s = jnp.where(scale == 0, 0.0, 1.0)

    return x_scaled, lb_s, ub_s, scale_safe
