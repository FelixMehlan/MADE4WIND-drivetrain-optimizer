import numpy as np

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog
from gearopt.config.load_config import load_config


def calc_l10h(x, i_sts, geo, loads, par, return_all=False):
    """
    Smooth/differentiable bearing life (L10h) and safety SB (ISO 281 style).

    Parameters
    ----------
    x : array-like (10,)
        [dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta, i_st]
    i_sts : float
        Upstream stage ratio (not used here; kept for API uniformity).
    geo : dict
        Geometry dictionary from calc_geometry().
    loads : array-like (16,)
        From calc_loads_ldd(); uses F_rad=loads[10], F_ax=loads[11], rpm_b=loads[15].
    return_all : bool
        If True, returns (SB, report_dict). Otherwise returns SB (float).

    Returns
    -------
    SB : float
        Bearing life safety factor (dimensionless).
    report : dict (only if return_all=True)
        Detailed intermediate values.
    """

    # --- Load mechanical constants from config ---
    n0 = par["n0"]
    T0 = par["T0"]
    L10h_min = par["L10h_min"]
 
    # --- Design variables ---
    dm_n, z_s, z_p_cont, z_r_cont, Np_raw, db, x_s, x_p, eps_beta = x

    # === Smoothed basics ===
    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50], dtype=float)
    dm_n_sc = softclip(dm_n, 1.0, len(m_n_table), 20.0)
    m_n = np.interp(dm_n_sc, np.arange(1, len(m_n_table) + 1), m_n_table)

    b = db * m_n
    N_p = np.clip(Np_raw, 3.0, 5.0)

    arg_raw = safe_div(eps_beta * np.pi * m_n, b)
    beta = safe_asin(arg_raw)

    # --- Geometry unpack (from dict) ---
    d_i_p = geo["d_i_p"]  # planet gear bore diameter [mm]

    # --- Loads ---
    F_rad = loads["F_rad"]
    F_ax = loads["F_ax"]
    rpm_b = loads["rpm_b"]

    # --- Bearing fit functions ---
    fits = fit_bearing_catalog()
    fit_C1 = fits["fit_C1"]
    fit_d = fits["fit_d"]
    fit_X1 = fits["fit_X1"]
    fit_Y1 = fits["fit_Y1"]
    fit_W = fits["fit_W"]

    # --- Bearing geometry inputs to fits ---
    B = 0.5 * b    # mm, bearing width estimate
    D = d_i_p      # mm, inner diameter from gear geometry

    # Evaluate smooth fits
    C_1 = fit_C1(D, B) * 1e3     # convert to N
    d = fit_d(D, B)              # mm (useful to report)
    X_1 = fit_X1(D, B)
    Y_1 = fit_Y1(D, B)
    W = fit_W(D, B)

    # --- Equivalent dynamic bearing load (ISO) ---
    P = X_1 * F_rad + Y_1 * F_ax  # N
    
    # --- L10 basic rating life (in 10^6 revolutions) ---
    L10 = abs_smooth(safe_div(C_1, P)) ** (10.0 / 3.0)

    # --- Convert to hours using bearing speed ---
    L10h = safe_div(L10 * 1e6, rpm_b * 60.0)

    # --- Safety factor on life ---
    SB = (L10h/ L10h_min) ** (3.0 / 10.0)

    if return_all:
        report = {
            "d": d,
            "D": D,
            "B": B,
            "C1": C_1,
            "W": W,
            "L10h": L10h,
            "P_eq": P,
            "X1": X_1,
            "Y1": Y_1,
            "beta": beta,
            "b": b,
            "N_p": N_p,
            "m_n": m_n,
            "rpm_b": rpm_b,
        }
        return float(SB), report

    return float(SB)
