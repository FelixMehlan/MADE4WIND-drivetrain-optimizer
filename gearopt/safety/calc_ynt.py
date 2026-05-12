import jax.numpy as jnp

def calc_ynt(rpm):
    """
    JAX-differentiable lifetime factor Y_NT(rpm).
    Exact mathematical match to the original NumPy version.
    """

    # Stress cycles over 25-year design life
    n = rpm * 25.0 * 8766.0 * 60.0

    # Slope m from empirical log-log relationship
    m = (
        (jnp.log10(1.0) - jnp.log10(0.93))
        / (jnp.log10(3e6) - jnp.log10(110e6))
    )

    # Constant C to satisfy Y_NT(3e6 cycles) = 1
    C = (3e6) ** (-m)

    # Lifetime factor
    return C * n**m
