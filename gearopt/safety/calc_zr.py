import jax.numpy as jnp
from gearopt.util.safe_div import safe_div
from gearopt.util.abs_smooth import abs_smooth


def calc_zr(Rz, d_b1, d_b2, alpha_wt):
    """
    ISO 6336-2 surface roughness factor Z_R (JAX version, fully differentiable).

    Parameters
    ----------
    Rz : float or array
        Arithmetic mean surface roughness [µm].
    d_b1 : float
        Base diameter of pinion [mm].
    d_b2 : float
        Base diameter of gear [mm].
    alpha_wt : float
        Operating transverse pressure angle [rad].

    Returns
    -------
    Z_R : float or array
        Surface roughness factor (dimensionless).
    """

    # Equivalent curvature radii
    rho_1 = 0.5 * d_b1 * jnp.tan(alpha_wt)
    rho_2 = 0.5 * d_b2 * jnp.tan(alpha_wt)

    # Reduced radius (Hertzian)
    rho_red = safe_div(rho_1 * rho_2, rho_1 + rho_2)

    # Adjusted roughness term
    Rz_10 = Rz * abs_smooth(10.0 / rho_red) ** (1.0 / 3.0)

    # ISO roughness factor
    Z_R = abs_smooth(3.0 / Rz_10) ** 0.08

    return Z_R
