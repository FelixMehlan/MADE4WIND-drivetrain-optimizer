import jax.numpy as jnp

def calc_zbd():
    """
    JAX version of ISO 6336 Z_B and Z_D.
    Identical to the NumPy version but JAX-safe.
    """

    f_ZCa = 1.2
    Z_B = jnp.sqrt(f_ZCa)
    Z_D = Z_B
    return Z_B, Z_D
