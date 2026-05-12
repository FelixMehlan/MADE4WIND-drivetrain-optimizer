import jax.numpy as jnp
from jax import lax

def interp_linear(x, xp, fp):
    """
    JAX-friendly linear interpolation, equivalent to numpy.interp.
    Assumes xp is strictly increasing.

    Parameters
    ----------
    x : float or array
        Query point(s)
    xp : array
        Grid points (monotonically increasing)
    fp : array
        Values at grid points

    Returns
    -------
    y : float or array
        Interpolated values
    """

    # Ensure arrays
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)

    # Clip input to avoid out-of-range conditions
    x_clamped = jnp.clip(x, xp[0], xp[-1])

    # Searchsorted — returns insertion index
    idx = jnp.searchsorted(xp, x_clamped, side="right") - 1

    # Clip interval index to valid range
    max_idx = xp.shape[0] - 2
    idx = jnp.clip(idx, 0, max_idx)

    # Gather segment endpoints
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    # Linear interpolation factor
    t = (x_clamped - x0) / (x1 - x0)

    return y0 + t * (y1 - y0)
