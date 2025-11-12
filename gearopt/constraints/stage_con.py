import numpy as np

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.safety.calc_sf import calc_sf
from gearopt.safety.calc_sh import calc_sh
from gearopt.safety.calc_shaft import calc_shaft
from gearopt.safety.calc_l10h import calc_l10h
from gearopt.efficiency.calc_efficiency import calc_efficiency
from gearopt.util.safe_div import safe_div
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.config.load_config import load_config


def stage_con(x, i_st, i_sts, par, data):
    """
    Full stage-level constraint evaluation for a planetary gear stage.
    Updated to use dictionary-based geometry instead of array indices.

    Returns
    -------
    C : np.ndarray
        Inequality constraints (<= 0 feasible).
    Ceq : np.ndarray
        Equality constraints (empty here).
    report : dict
        Structured output for postprocessing.
    """
    # --- parameters ---
    SF_min = par["SF_min"]
    SH_min = par["SH_min"]
    S_ShULS_min = par["S_ShULS_min"]

    # --- design variables ---
    dm_n, z_s, z_p_cont, z_r_cont, Np_raw, db, x_s, x_p, eps_beta = x

    # --- smoothed basics ---
    N_p = np.clip(Np_raw, 3.0, 5.0)

    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50])
    m_n = np.interp(dm_n, np.arange(1, len(m_n_table) + 1), m_n_table)

    b = db * m_n
    arg_raw = safe_div(eps_beta * np.pi * m_n, b)
    arg_safe = softclip(arg_raw, -0.999, 0.999, 10.0)
    beta = np.arcsin(arg_safe)
    beta_deg = np.rad2deg(beta)

    # --- geometry and loads ---
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # --- safety factors ---
    SF_s, SF_p, SF_r = calc_sf(x, i_sts, geo, loads, par)
    cond_SF_s = SF_min - SF_s
    cond_SF_p = SF_min - SF_p
    cond_SF_r = SF_min - SF_r

    SH_s, SH_p, SH_r = calc_sh(x, i_sts, geo, loads, par)
    cond_SH_s = SH_min - SH_s
    cond_SH_p = SH_min - SH_p
    cond_SH_r = SH_min - SH_r

    # --- bearings / shafts ---
    SB, reportB = calc_l10h(x, i_sts, geo, loads, par, return_all=True)
    cond_L10h = 1 - SB

    S_ShULS_s, S_ShULS_p, reportSh = calc_shaft(x, i_sts, geo, loads, par)
    tol = 1e-3
    cond_ShULS_s = S_ShULS_min - S_ShULS_s - tol
    cond_ShULS_p = S_ShULS_min - S_ShULS_p - tol

    # Shaft–gear fit constraints (use named fields)
    cond_dss = (reportSh["d_ss"] - geo["d_i_s"]) / geo["d_i_s"]       # shaft >= gear inner
    cond_dps = (reportSh["d_ps"] - reportB["d"]) / reportB["d"]       # planet shaft >= bearing bore

    # --- efficiency ---
    eta = calc_efficiency(x, i_sts, geo, loads, par)

    # --- geometric compatibility ---
    pi_N_p = np.pi / N_p
    cond_adj = safe_div(geo["d_a_p"] - 2 * geo["a"] * np.sin(pi_N_p), geo["d_a_p"])

    # mesh compatibility (avoid using int division!)
    mesh_residual = np.sin(np.pi * (z_r_cont + z_s) / max(N_p, 1e-12))
    tol_mesh = 1e-3
    cond_mesh = abs_smooth(mesh_residual) - tol_mesh

    # ratio tolerance
    i_12a = 1 + safe_div(z_r_cont, z_s)
    cond_ratio = safe_div(abs_smooth(i_12a - i_st), max(i_st, 1e-12)) - 0.02

    # profile shift bounds
    x_r = geo["x_r"]
    cond_x5 = -x_r - 0.6
    cond_x6 = x_r - 1.0

    # --- collect inequalities ---
    C = np.array([
        cond_adj, cond_ratio, cond_mesh,
        cond_SF_s, cond_SF_p, cond_SF_r,
        cond_SH_s, cond_SH_p, cond_SH_r,
        cond_x5, cond_x6,
        cond_L10h,
        cond_ShULS_s, cond_ShULS_p,
        cond_dss, cond_dps,
    ])

    Ceq = np.array([])

    # --- reporting ---
    report = {
        "gear": dict(
            m_n=m_n, z_s=z_s, z_p=z_p_cont, z_r=z_r_cont, b=b,
            x_s=x_s, x_p=x_p, x_r=x_r, i_st=i_st, beta_deg=beta_deg
        ),
        "bearing": reportB,
        "shaft": reportSh,
        "safety": dict(
            SF_s=SF_s, SF_p=SF_p, SF_r=SF_r,
            SH_s=SH_s, SH_p=SH_p, SH_r=SH_r,
            SB=SB, S_ShULS_s=S_ShULS_s, S_ShULS_p=S_ShULS_p
        ),
        "efficiency": dict(eta=eta),
    }

    return C, Ceq, report
