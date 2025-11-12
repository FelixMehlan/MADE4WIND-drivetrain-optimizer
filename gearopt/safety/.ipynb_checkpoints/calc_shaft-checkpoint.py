import numpy as np

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog
from gearopt.config.load_config import load_config

def calc_shaft(x, i_sts, geo, loads, par, return_all=False):
    """
    Shaft ultimate limit state (ULS) safety factors.
    Translation of MATLAB calcShaft.m with smooth utilities.

    Parameters
    ----------
    x : array-like (10,)
        [dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta, i_st]
    par : array-like
        Parameter vector from parameters_default().
    i_sts : float
        Stage ratio (not directly used, included for interface consistency).
    geo : array-like
        Geometry vector from calc_geometry().
    loads : array-like
        Load vector from calc_loads_ldd().
    return_all : bool, optional
        If True, returns a detailed report dictionary.

    Returns
    -------
    S_ShULS_s : float
        Sun shaft safety factor (ULS).
    S_ShULS_p : float
        Planet shaft safety factor (ULS).
    reportSh : dict (if return_all=True)
        Detailed output dictionary.
    """

    # --- Parameters ---
    rho_steel = par["rho_steel"]
    sigma_Y = par["sigma_Y"]
    S_ShULS_min = par["S_ShULS_min"]  # minimum required ULS safety factor

    # --- Design variables ---
    dm_n, z_s, z_p_cont, z_r_cont, Np_raw, db, x_s, x_p, eps_beta = x

    # --- Smoothed basics ---
    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50], dtype=float)
    dm_n_sc = softclip(dm_n, 1.0, len(m_n_table), 20.0)
    m_n = np.interp(dm_n_sc, np.arange(1, len(m_n_table) + 1), m_n_table)

    b = db * m_n
    N_p = np.clip(Np_raw, 3.0, 5.0)

    arg_raw = safe_div(eps_beta * np.pi * m_n, b)
    beta = safe_asin(arg_raw)

    # --- Geometry ---
    d_i_p = geo["d_i_p"]  # MATLAB index 22 → Python index 21 (planet gear inner diameter)

    # --- Loads ---
    if loads is None or len(loads) < 2:
        raise ValueError("calc_shaft: 'loads' must contain at least two torque values.")

    T_s_ShULS = loads["T_s_ShULS"]  # sun shaft torque [N·m]
    T_p_ShULS = loads["T_p_ShULS"]  # planet shaft torque [N·m]

    # --- Bearing fit functions ---

    # load built-in hard-coded bearing catalog fits
    bearing_fits = fit_bearing_catalog()
    fit_d = bearing_fits["fit_d"]


    # --- Planet shaft diameter (from bearing fit) ---
    B = 0.5 * b  # bearing width proxy [mm]
    D = d_i_p
    d_ps = fit_d(D, B)
    d_ps = np.maximum(d_ps, 1e-6)

    # --- Sun shaft diameter (torsion ULS) ---
    T_s_Nmm = T_s_ShULS * 1e3  # convert to N·mm
    T_p_Nmm = T_p_ShULS * 1e3  # convert to N·mm

    d_ss = abs_smooth(
        safe_div(16 * np.sqrt(3) * T_s_Nmm * S_ShULS_min, np.pi * sigma_Y)
    ) ** (1 / 3)

    # --- Polar moments of inertia ---
    I_ss = (np.pi / 32.0) * d_ss**4
    I_ps = (np.pi / 32.0) * d_ps**4

    # --- Torsional stresses ---
    tau_ss = safe_div(T_s_Nmm * d_ss, 2 * I_ss)
    tau_ps = safe_div(T_p_Nmm * d_ps, 2 * I_ps)

    sigma_vm_ss = np.sqrt(3) * tau_ss
    sigma_vm_ps = np.sqrt(3) * tau_ps

    # --- Safety factors ---
    S_ShULS_s = safe_div(sigma_Y, sigma_vm_ss)
    S_ShULS_p = safe_div(sigma_Y, sigma_vm_ps)

    # --- Build report ---
    reportSh = dict(
        d_ss=d_ss,
        d_ps=d_ps,
        tau_ss=tau_ss,
        tau_ps=tau_ps,
        sigma_vm_ss=sigma_vm_ss,
        sigma_vm_ps=sigma_vm_ps,
        S_ShULS_s=S_ShULS_s,
        S_ShULS_p=S_ShULS_p,
        beta=beta,
        b=b,
        N_p=N_p,
        m_n=m_n,
    )

    if return_all:
        return reportSh

    return float(S_ShULS_s), float(S_ShULS_p), reportSh
