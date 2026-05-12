import jax.numpy as jnp

def calc_znt(rpm):
    """
    JAX-differentiable ISO 6336-2 contact fatigue life factor Z_NT.

    Parameters
    ----------
    rpm : float or array (JAX-traceable)
        Rotational speed [revolutions per minute].

    Returns
    -------
    Z_NT : float or array
        Contact fatigue life factor (dimensionless).
    """

    # Total cycles over 25-year life
    n = rpm * 25.0 * 8766.0 * 60.0

    # Slope of S–N curve (stay JAX-safe)
    m = (jnp.log10(0.948) - jnp.log10(0.913)) / (
            jnp.log10(281.6e6) - jnp.log10(964.6e6)
        )

    # Constant from reference point
    C = 0.948 * (281.6e6) ** (-m)

    # Fatigue life factor
    Z_NT = C * n**m

    return Z_NT
