import numpy as np


def fit_bearing_catalog():
    """
    Hard-coded polynomial fits equivalent to MATLAB fitBearingCatalog.m.

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
    All inputs in mm.
    """

    # === Poly22: fit_C1 ===
    # ans(x,y) = p00 + p10*x + p01*y + p20*x^2 + p11*x*y + p02*y^2
    def fit_C1(D, B):
        p00, p10, p01, p20, p11, p02 = (
            131.8,
            -2.066,
            -0.5334,
            -0.004363,
            0.0992,
            -0.04296,
        )
        return (
            p00
            + p10 * D
            + p01 * B
            + p20 * D**2
            + p11 * D * B
            + p02 * B**2
        )

    # === Poly22: fit_d ===
    def fit_d(D, B):
        p00, p10, p01, p20, p11, p02 = (
            -7.198,
            0.772,
            -0.3018,
            0.0001057,
            -4.265e-05,
            -0.0003866,
        )
        return (
            p00
            + p10 * D
            + p01 * B
            + p20 * D**2
            + p11 * D * B
            + p02 * B**2
        )

    # === Poly11: fit_X1 ===
    def fit_X1(D, B):
        p00, p10, p01 = (1.0, -2.335e-19, 1.787e-19)
        return p00 + p10 * D + p01 * B

    # === Poly11: fit_Y1 ===
    def fit_Y1(D, B):
        p00, p10, p01 = (1.634, 0.0001997, 0.0001115)
        return p00 + p10 * D + p01 * B

    # === Poly22: fit_W ===
    def fit_W(D, B):
        p00, p10, p01, p20, p11, p02 = (
            42.95,
            -0.1774,
            -0.6657,
            -1.364e-05,
            0.003067,
            0.0004722,
        )
        return (
            p00
            + p10 * D
            + p01 * B
            + p20 * D**2
            + p11 * D * B
            + p02 * B**2
        )

    return {
        "fit_C1": fit_C1,
        "fit_d": fit_d,
        "fit_X1": fit_X1,
        "fit_Y1": fit_Y1,
        "fit_W": fit_W,
    }
