import jax
import jax.numpy as jnp

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.interp_linear import interp_linear
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog


def calc_l10h(x, i_sts, geo, loads, par, return_all=False):
    """
    JAX-compatible smooth bearing life model (L10h).
    No NumPy. Fully differentiable. Safe for jit+grad.
    """

    # === Parameters ===
    L10h_min = par["L10h_min"]

    # === Design variables (same order as rest of your code) ===
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # === Module interpolation ===
    m_n_table = jnp.array([8., 10., 12., 16., 20., 25., 32., 40., 50.])
    idxs = jnp.arange(1, m_n_table.size + 1)

    dm_n_sc = softclip(dm_n, 1.0, float(len(m_n_table)), 20.0)
    m_n = interp_linear(dm_n_sc, idxs, m_n_table)

    # === Basics ===
    b = db * m_n
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    arg_raw = safe_div(eps_beta * jnp.pi * m_n, b)
    beta = safe_asin(arg_raw)

    # === Geometry ===
    d_i_p = geo["d_i_p"]

    # === Loads ===
    F_rad = loads["F_rad"]
    F_ax = loads["F_ax"]
    rpm_b = loads["rpm_b"]

    # === Bearing fits (already JAX-safe) ===
    fits = fit_bearing_catalog()
    fit_C1 = fits["fit_C1"]
    fit_d = fits["fit_d"]
    fit_X1 = fits["fit_X1"]
    fit_Y1 = fits["fit_Y1"]
    fit_W  = fits["fit_W"]

    B = 0.5 * b
    D = d_i_p

    C_1 = fit_C1(D, B) * 1e3
    X_1 = fit_X1(D, B)
    Y_1 = fit_Y1(D, B)
    d    = fit_d(D, B)
    W    = fit_W(D, B)

    # === Equivalent dynamic load ===
    P = X_1 * F_rad + Y_1 * F_ax

    # === Basic rating life (ISO 281) ===
    L10 = abs_smooth(safe_div(C_1, P)) ** (10.0 / 3.0)

    # === Convert to hours ===
    L10h = safe_div(L10 * 1e6, rpm_b * 60.0)

    # === Safety factor ===
    SB = (safe_div(L10h, L10h_min)) ** (3.0 / 10.0)

    # === Optional report ===
    if return_all:
        report = dict(
            d=d,
            D=D,
            B=B,
            C1=C_1,
            W=W,
            L10h=L10h,
            P_eq=P,
            X1=X_1,
            Y1=Y_1,
            beta=beta,
            b=b,
            N_p=N_p,
            m_n=m_n,
            rpm_b=rpm_b,
        )
        return SB, report

    return SB
