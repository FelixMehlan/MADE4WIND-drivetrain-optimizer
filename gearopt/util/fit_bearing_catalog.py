import jax.numpy as jnp


def fit_bearing_catalog():
    """
    JAX-compatible hard-coded polynomial fits equivalent to MATLAB fitBearingCatalog.m.

    Returns
    -------
    dict of callables:
        {
            "fit_C1": f(D, B),
            "fit_d": f(D, B),
            "fit_X1": f(D, B),
            "fit_Y1": f(D, B),
            "fit_W": f(D, B)
        }

    All inputs D, B in mm.
    """

    # === Poly22 coefficients (C1) ===
    p00_C1, p10_C1, p01_C1, p20_C1, p11_C1, p02_C1 = (
        131.8,
        -2.066,
        -0.5334,
        -0.004363,
        0.0992,
        -0.04296,
    )

    def fit_C1(D, B):
        D = jnp.asarray(D)
        B = jnp.asarray(B)
        return (
            p00_C1
            + p10_C1 * D
            + p01_C1 * B
            + p20_C1 * D**2
            + p11_C1 * D * B
            + p02_C1 * B**2
        )

    # === Poly22 (d) ===
    p00_d, p10_d, p01_d, p20_d, p11_d, p02_d = (
        -7.198,
        0.772,
        -0.3018,
        0.0001057,
        -4.265e-05,
        -0.0003866,
    )

    def fit_d(D, B):
        D = jnp.asarray(D)
        B = jnp.asarray(B)
        return (
            p00_d
            + p10_d * D
            + p01_d * B
            + p20_d * D**2
            + p11_d * D * B
            + p02_d * B**2
        )

    # === Poly11 (X1) ===
    p00_X1, p10_X1, p01_X1 = (1.0, -2.335e-19, 1.787e-19)

    def fit_X1(D, B):
        D = jnp.asarray(D)
        B = jnp.asarray(B)
        return p00_X1 + p10_X1 * D + p01_X1 * B

    # === Poly11 (Y1) ===
    p00_Y1, p10_Y1, p01_Y1 = (1.634, 0.0001997, 0.0001115)

    def fit_Y1(D, B):
        D = jnp.asarray(D)
        B = jnp.asarray(B)
        return p00_Y1 + p10_Y1 * D + p01_Y1 * B

    # === Poly22 (W) ===
    p00_W, p10_W, p01_W, p20_W, p11_W, p02_W = (
        42.95,
        -0.1774,
        -0.6657,
        -1.364e-05,
        0.003067,
        0.0004722,
    )

    def fit_W(D, B):
        D = jnp.asarray(D)
        B = jnp.asarray(B)
        return (
            p00_W
            + p10_W * D
            + p01_W * B
            + p20_W * D**2
            + p11_W * D * B
            + p02_W * B**2
        )

    return {
        "fit_C1": fit_C1,
        "fit_d": fit_d,
        "fit_X1": fit_X1,
        "fit_Y1": fit_Y1,
        "fit_W": fit_W,
    }
