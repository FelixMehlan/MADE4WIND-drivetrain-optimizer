import numpy as np

from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.safety.calc_shaft import calc_shaft
from gearopt.efficiency.calc_efficiency import calc_efficiency
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog
from gearopt.config.load_config import load_config

def stage_weight(x, i_st, i_sts, par, data):
    """
    Stage weight calculation (Python version of stageWeight.m)

    Calculates the mass of a single gearbox stage and its main components
    (gears, shafts, bearings, carrier) using smooth continuous formulations.

    Parameters
    ----------
    x : array-like (10,)
        [dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta, i_st]
    par : array-like
        Parameter vector from parameters_default().
    i_sts : float
        Stage ratio.

    Returns
    -------
    W_st : float
        Total stage weight [kg].
    reportW_k : dict
        Detailed weight and geometry report.
    """

    # === Parameters ===
    rho_steel = par["rho_steel"]
    w = par["w"]
    
    # === Design variables ===
    dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta = x

    # === Smoothed basics ===
    N_p = np.clip(Np_raw, 3.0, 5.0)

    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50], dtype=float)
    m_n = np.interp(dm_n, np.arange(1, len(m_n_table) + 1), m_n_table)

    b = db * m_n
    arg_raw = eps_beta * np.pi * m_n / np.maximum(b, 1e-12)
    arg_safe = softclip(arg_raw, -0.999, 0.999, 10.0)
    beta = np.arcsin(arg_safe)
    beta_deg = np.degrees(beta)

    # === Geometry / Loads ===
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # === Bearing fits ===
    fits = fit_bearing_catalog()
    fit_d = fits["fit_d"]
    fit_W = fits["fit_W"]


    # === Bearing sizing ===
    b_b = 0.5 * b  # bearing width
    d_i_p = geo["d_i_p"] # planet gear inner diameter (MATLAB index 22)
    d_o_b = fit_d(d_i_p, b_b)

    # === Shaft sizing ===
    _, _, reportSh = calc_shaft(x, i_sts, geo, loads, par)
    d_ss = reportSh["d_ss"]
    d_ps = reportSh["d_ps"]

    # === Geometry values for weight (dictionary access) ===
    d_pw_s = geo["d_pw_s"]    # sun pitch diameter
    d_pw_p = geo["d_pw_p"]    # planet pitch diameter
    d_pw_r = geo["d_pw_r"]    # ring pitch diameter
    d_o_r  = geo["d_o_r"]     # ring outer diameter
    a      = geo["a"]         # center distance

    # === Constants ===
    pi_4_rho = np.pi * 0.25 * rho_steel
    k_PLC = 4.45  # empirical carrier weight coefficient

    # === Component weights ===
    W_sg = pi_4_rho * abs_smooth(d_pw_s**2 - d_ss**2) * b
    W_pg = pi_4_rho * abs_smooth(d_pw_p**2 - d_i_p**2) * b
    W_rg = pi_4_rho * abs_smooth(d_o_r**2 - d_pw_r**2) * b
    W_ss = pi_4_rho * d_ss**2 * (2 * b)
    W_ps = pi_4_rho * d_ps**2 * b

    # Bearing weight
    W_pb = fit_W(d_o_b, b_b) if callable(fit_W) else 0.1 * (d_o_b * b_b)

    # Carrier weight
    W_plc = rho_steel * np.pi * 0.25 * a**2 * b * k_PLC

    # === Total stage weight ===
    W_st = (
        W_sg + W_ss
        + N_p * (W_pg + 2 * W_pb + W_ps)
        + W_rg + W_plc
    )

    # === Efficiency ===
    eta = calc_efficiency(x, i_sts, geo, loads, par)

    # === Report structure ===
    reportW_k = {
        "gear": {
            "W_st": W_st,
            "W_sg": W_sg,
            "W_pg": W_pg,
            "W_rg": W_rg,
            "W_ss": W_ss,
            "W_ps": W_ps,
            "W_pb": W_pb,
            "W_plc": W_plc,
            "N_p": N_p,
            "total_planet_gears": N_p * W_pg,
            "total_planet_bearings": N_p * 2 * W_pb,
            "total_planet_shafts": N_p * W_ps,
            "eta": eta,
            "w": w,  # efficiency weighting factor
        },
        "dimensions": {
            "m_n": m_n,
            "b": b,
            "z_s": z_s,
            "z_p": z_p,
            "z_r": z_r,
            "d_ss": d_ss,
            "d_ps": d_ps,
            "d_o_b": d_o_b,
            "b_b": b_b,
        },
    }

    return float(W_st), reportW_k
