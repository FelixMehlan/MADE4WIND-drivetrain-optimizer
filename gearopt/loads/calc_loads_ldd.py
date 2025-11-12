import numpy as np
from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.data.load_tspec import load_tspec
from gearopt.config.load_config import load_config

def calc_loads_ldd(x, geo, i_sts, par, data, return_all=False):
    """
    Gradient-friendly load and duty distribution calculator.
    Translated from MATLAB calcLoadsLDD.m (smoothed and differentiable).

    Parameters
    ----------
    x : array-like (10,)
        Design vector [dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta, i_st]
    geo : array-like
        Geometry vector from calc_geometry().
    i_sts : float
        Cumulative gear ratio of higher stages.
    return_all : bool, optional
        If True, return detailed dictionary instead of array (default False).

    Returns
    -------
    loads : ndarray (16,) or dict
        If return_all=False:
            [T_s_ULS, T_p_ULS, T_s_FLS, T_p_FLS,
             F_t_s_SH, F_t_s_SF, F_t_s_SB,
             F_t_r_SH, F_t_r_SF, F_t_r_SB,
             F_rad, F_ax,
             rpm_r, rpm_s, rpm_p, rpm_b]
        If return_all=True:
            Dictionary with all intermediate and output variables.
    """
    # --- torque input ---
    T_spec = data

    # --- Parameters ---
    n0 = par["n0"]
    T0 = par["T0"]
    p_SF = par["p_SF"]
    p_SH = par["p_SH"]
    p_Sh = par["p_Sh"]
    T_max = par["T_max"]

    # --- Design variables ---
    dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta = x

    # --- Smooth basics ---
    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50], dtype=float)
    m_n = np.interp(dm_n, np.arange(1, len(m_n_table) + 1), m_n_table)

    b = db * m_n
    N_p = np.clip(Np_raw, 3.0, 5.0)
    beta = safe_asin(safe_div(eps_beta * np.pi * m_n, b))

    # --- Geometry subset ---
    alpha_t = geo["alpha_t"]          # transverse pressure angle
    d_p_s = geo["d_p_s"]              # sun gear pitch diameter
    d_p_p = geo["d_p_p"]              # planet gear pitch diameter
    d_p_r = geo["d_p_r"]              # ring gear pitch diameter
    a = geo["a"]                      # center distance

    # --- Torque spectrum ---
    w_i = T_spec[:, 0]
    tbar = T_spec[:, 1]

    T_eq_SH = (np.sum(w_i * tbar**p_SH)) ** (1 / p_SH) * T0
    T_eq_SF = (np.sum(w_i * tbar**p_SF)) ** (1 / p_SF) * T0
    T_eq_SB = (np.sum(w_i * tbar ** (10 / 3))) ** (3 / 10) * T0
    T_eq_Sh = (np.sum(w_i * tbar ** (1 / p_Sh))) ** (1 / p_Sh) * T0

    # --- Ratios ---
    z_s_safe = np.maximum(z_s, 1e-9)
    z_p_safe = np.maximum(z_p, 1e-9)
    Np_safe = np.maximum(N_p, 1e-9)

    i_sta = 1.0 + safe_div(z_r, z_s_safe)
    i_sta_inv = safe_div(1.0, i_sta)
    i_sts_inv = safe_div(1.0, i_sts)

    # --- Shaft torques ---
    T_s_SH = T_eq_SH * i_sta_inv * i_sts_inv
    T_s_SF = T_eq_SF * i_sta_inv * i_sts_inv
    T_s_SB = T_eq_SB * i_sta_inv * i_sts_inv
    T_s_ShFLS = T_eq_Sh * i_sta_inv * i_sts_inv
    T_s_ShULS = T_max * i_sta_inv * i_sts_inv

    # --- Ring torques ---
    T_r_SH = T_eq_SH * i_sts_inv - T_s_SH
    T_r_SF = T_eq_SF * i_sts_inv - T_s_SF
    T_r_SB = T_eq_SB * i_sts_inv - T_s_SB

    # --- Planet torques ---
    z_ratio = safe_div(z_p_safe, z_s_safe)
    Np_inv = safe_div(1.0, Np_safe)
    T_p_ShFLS = T_s_ShFLS * Np_inv * z_ratio
    T_p_ShULS = T_s_ShULS * Np_inv * z_ratio

    # --- Forces ---
    d_p_s_m = d_p_s * 1e-3
    d_p_r_m = d_p_r * 1e-3
    r_ps = 0.5 * d_p_s_m
    r_pr = 0.5 * d_p_r_m

    F_t_s_SH = np.abs(safe_div(T_s_SH, N_p * r_ps))
    F_t_s_SF = np.abs(safe_div(T_s_SF, N_p * r_ps))
    F_t_s_SB = np.abs(safe_div(T_s_SB, N_p * r_ps))

    F_t_r_SH = np.abs(safe_div(T_r_SH, N_p * r_pr))
    F_t_r_SF = np.abs(safe_div(T_r_SF, N_p * r_pr))
    F_t_r_SB = np.abs(safe_div(T_r_SB, N_p * r_pr))

    # --- Radial & axial forces ---
    F_r_s_SB = F_t_s_SB * np.tan(alpha_t) / np.cos(beta)
    F_r_r_SB = F_t_r_SB * np.tan(alpha_t) / np.cos(beta)
    F_a_s_SB = F_t_s_SB * np.tan(beta)
    F_a_r_SB = F_t_r_SB * np.tan(beta)

    F_x_SB = F_a_s_SB - F_a_r_SB
    F_y_SB = F_t_s_SB + F_t_r_SB
    F_z_SB = F_r_s_SB + F_r_r_SB

    F_rad = np.hypot(F_y_SB, F_z_SB)
    F_ax = np.abs(F_x_SB)

    # --- Speeds (rpm) ---
    rpm_r = n0 * N_p * i_sts
    rpm_s = (n0 * i_sta - n0) * N_p * i_sts
    rpm_p = n0 * safe_div(z_r, z_p_safe) * i_sts
    rpm_b = n0 * (1 + safe_div(z_r, z_p_safe)) * i_sts

        # --- Pack results into dict ---
    loads = {
        # --- Equivalent torques ---
        "T_eq_SH": T_eq_SH,
        "T_eq_SF": T_eq_SF,
        "T_eq_SB": T_eq_SB,
        "T_eq_Sh": T_eq_Sh,

        # --- Stage ratios ---
        "i_sta": i_sta,
        "i_sts": i_sts,

        # --- Shaft torques ---
        "T_s_SH": T_s_SH,
        "T_s_SF": T_s_SF,
        "T_s_SB": T_s_SB,
        "T_s_ShFLS": T_s_ShFLS,
        "T_s_ShULS": T_s_ShULS,

        # --- Ring torques ---
        "T_r_SH": T_r_SH,
        "T_r_SF": T_r_SF,
        "T_r_SB": T_r_SB,

        # --- Planet torques ---
        "T_p_ShFLS": T_p_ShFLS,
        "T_p_ShULS": T_p_ShULS,

        # --- Tangential forces ---
        "F_t_s_SH": F_t_s_SH,
        "F_t_s_SF": F_t_s_SF,
        "F_t_s_SB": F_t_s_SB,
        "F_t_r_SH": F_t_r_SH,
        "F_t_r_SF": F_t_r_SF,
        "F_t_r_SB": F_t_r_SB,

        # --- Resultant loads ---
        "F_rad": F_rad,
        "F_ax": F_ax,

        # --- Rotational speeds ---
        "rpm_r": rpm_r,
        "rpm_s": rpm_s,
        "rpm_p": rpm_p,
        "rpm_b": rpm_b,

        # --- Derived values ---
        "beta": beta,
        "b": b,
        "N_p": N_p,
    }

    # --- Return ---
    return loads if return_all else {
        # Minimal subset (for performance-critical routines)
        "T_s_ShULS": T_s_ShULS,
        "T_p_ShULS": T_p_ShULS,
        "T_s_ShFLS": T_s_ShFLS,
        "T_p_ShFLS": T_p_ShFLS,
        "F_t_s_SH": F_t_s_SH,
        "F_t_s_SF": F_t_s_SF,
        "F_t_s_SB": F_t_s_SB,
        "F_t_r_SH": F_t_r_SH,
        "F_t_r_SF": F_t_r_SF,
        "F_t_r_SB": F_t_r_SB,
        "F_rad": F_rad,
        "F_ax": F_ax,
        "rpm_r": rpm_r,
        "rpm_s": rpm_s,
        "rpm_p": rpm_p,
        "rpm_b": rpm_b,
    }
