import numpy as np

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.smoothmin2 import smoothmin2
from gearopt.safety.calc_yf_hob import calc_yf_hob
from gearopt.safety.calc_yf_shaper import calc_yf_shaper
from gearopt.safety.calc_ynt import calc_ynt
from gearopt.safety.calc_yx import calc_yx
from gearopt.config.load_config import load_config


def calc_sf(x, i_sts, geo, loads, par, return_all=False):
    """
    ISO 6336-style root (bending) safety factors.
    Smooth, differentiable translation of calcSF.m (fully dict-based).

    Parameters
    ----------
    x : array-like (10,)
        [dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta, i_st]
    i_sts : float
        Upstream stage ratio (for compatibility; not directly used).
    geo : dict
        Geometry dictionary from calc_geometry().
    loads : dict
        Loads dictionary from calc_loads_ldd().
    return_all : bool, optional
        If True, return detailed dictionary of intermediate values.

    Returns
    -------
    (SF_s, SF_p, SF_r) : tuple of floats
        Root safety factors for sun, planet, and ring.
    """



    # --- Extract needed mechanical constants ---
    alpha_n = np.deg2rad(par["alpha_n_deg"])
    h_fPf = par["h_fPf"]
    rho_fPf = par["rho_fPf"]
    sigma_FE = par["sigma_FE"]
    Y_B = par["Y_B"]
    Y_DT = par["Y_DT"]
    Y_delta_rel_T = par["Y_delta_rel_T"]
    Y_R_rel_T = par["Y_R_rel_T"]
    Y_M_s = par["Y_M_s"]
    Y_M_p = par["Y_M_p"]
    Y_M_r = par["Y_M_r"]
    K_A = par["K_A"]
    K_V = par["K_V"]
    K_Falpha = par["K_Falpha"]
    K_Fbeta_sp = par["K_Fbeta_sp"]
    K_Fbeta_rp = par["K_Fbeta_rp"]

    # --- Design variables ---
    dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta = x

    # --- Smoothed module selection ---
    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50], dtype=float)
    m_n = np.interp(dm_n, np.arange(1, len(m_n_table) + 1), m_n_table)

    b = db * m_n
    N_p = np.clip(Np_raw, 3.0, 5.0)

    arg_raw = safe_div(eps_beta * np.pi * m_n, b)
    beta = safe_asin(arg_raw)

    # --- Geometry unpack ---
    d_p_s = geo["d_p_s"]
    d_p_p = geo["d_p_p"]
    d_p_r = geo["d_p_r"]
    d_a_s = geo["d_a_s"]
    d_a_p = geo["d_a_p"]
    d_a_r = geo["d_a_r"]
    x_r = geo["x_r"]
    eps_alpha_t_sp = geo["eps_alpha_t_sp"]
    eps_alpha_t_rp = geo["eps_alpha_t_rp"]

    # --- Loads unpack ---
    F_t_s_SH = loads["F_t_s_SH"]
    F_t_r_SH = loads["F_t_r_SH"]
    rpm_r = loads["rpm_r"]
    rpm_s = loads["rpm_s"]
    rpm_p = loads["rpm_p"]

    # --- Root geometry factors ---
    Y_F_s, Y_S_s = calc_yf_hob(d_p_s, d_a_s, z_s, x_s, eps_alpha_t_sp, m_n,
                                alpha_n, h_fPf, rho_fPf, beta, eps_beta)
    Y_F_sp, Y_S_sp = calc_yf_hob(d_p_p, d_a_p, z_p, x_p, eps_alpha_t_sp, m_n,
                                 alpha_n, h_fPf, rho_fPf, beta, eps_beta)
    Y_F_rp, Y_S_rp = calc_yf_hob(d_p_p, d_a_p, z_p, x_p, eps_alpha_t_rp, m_n,
                                 alpha_n, h_fPf, rho_fPf, beta, eps_beta)
    Y_F_r, Y_S_r = calc_yf_shaper(-d_p_r, -d_a_r, -z_r, x_r,
                                  eps_alpha_t_rp, m_n, alpha_n, h_fPf, rho_fPf, beta, eps_beta)

    # --- Helix/face load factors ---
    epsb_eff = np.minimum(eps_beta, 1.0)
    denom = (2.0 / 3.0) * np.pi
    Y_beta_sp = safe_div(1.0 - epsb_eff * beta / denom, np.cos(beta) ** 3)
    Y_beta_rp = Y_beta_sp

    # --- Base bending stresses ---
    bm = b * m_n

    sigma_F0_s = safe_div(F_t_s_SH, bm) * Y_F_s * Y_S_s * Y_beta_sp * Y_B * Y_DT
    sigma_F0_sp = safe_div(F_t_s_SH, bm) * Y_F_sp * Y_S_sp * Y_beta_sp * Y_B * Y_DT
    sigma_F0_rp = safe_div(F_t_r_SH, bm) * Y_F_rp * Y_S_rp * Y_beta_rp * Y_B * Y_DT
    sigma_F0_r = safe_div(F_t_r_SH, bm) * Y_F_r * Y_S_r * Y_beta_rp * Y_B * Y_DT

    # --- Mesh load factor ---
    K_gamma = np.interp(N_p, [3.0, 4.0, 5.0], [1.00, 1.20, 1.35])

    # --- Apply load factors ---
    sigma_F_s = sigma_F0_s * K_A * K_gamma * K_V * K_Fbeta_sp * K_Falpha
    sigma_F_sp = sigma_F0_sp * K_A * K_gamma * K_V * K_Fbeta_sp * K_Falpha
    sigma_F_rp = sigma_F0_rp * K_A * K_gamma * K_V * K_Fbeta_rp * K_Falpha
    sigma_F_r = sigma_F0_r * K_A * K_gamma * K_V * K_Fbeta_rp * K_Falpha

    # --- Life/size factors ---
    Y_X = calc_yx(m_n)
    Y_NT_s = calc_ynt(rpm_s)
    Y_NT_p = calc_ynt(rpm_p)
    Y_NT_r = calc_ynt(rpm_r)

    sigma_FG_s = sigma_FE * Y_NT_s * Y_delta_rel_T * Y_R_rel_T * Y_X * Y_M_s
    sigma_FG_sp = sigma_FE * Y_NT_p * Y_delta_rel_T * Y_R_rel_T * Y_X * Y_M_p
    sigma_FG_rp = sigma_FE * Y_NT_p * Y_delta_rel_T * Y_R_rel_T * Y_X * Y_M_p
    sigma_FG_r = sigma_FE * Y_NT_r * Y_delta_rel_T * Y_R_rel_T * Y_X * Y_M_r

    # --- Safety factors ---
    SF_s = safe_div(sigma_FG_s, sigma_F_s)
    SF_sp = safe_div(sigma_FG_sp, sigma_F_sp)
    SF_rp = safe_div(sigma_FG_rp, sigma_F_rp)
    SF_r = safe_div(sigma_FG_r, sigma_F_r)

    # Smooth-min for planets
    SF_p = smoothmin2(SF_sp, SF_rp, k=30.0)

    if not return_all:
        return float(SF_s), float(SF_p), float(SF_r)

    # --- Detailed output dictionary ---
    return dict(
        SF_s=SF_s, SF_p=SF_p, SF_r=SF_r,
        Y_F_s=Y_F_s, Y_S_s=Y_S_s,
        Y_F_sp=Y_F_sp, Y_S_sp=Y_S_sp,
        Y_F_rp=Y_F_rp, Y_S_rp=Y_S_rp,
        Y_F_r=Y_F_r, Y_S_r=Y_S_r,
        Y_beta_sp=Y_beta_sp, Y_beta_rp=Y_beta_rp,
        sigma_F0_s=sigma_F0_s, sigma_F0_sp=sigma_F0_sp,
        sigma_F0_rp=sigma_F0_rp, sigma_F0_r=sigma_F0_r,
        sigma_F_s=sigma_F_s, sigma_F_sp=sigma_F_sp,
        sigma_F_rp=sigma_F_rp, sigma_F_r=sigma_F_r,
        sigma_FG_s=sigma_FG_s, sigma_FG_sp=sigma_FG_sp,
        sigma_FG_rp=sigma_FG_rp, sigma_FG_r=sigma_FG_r,
        K_gamma=K_gamma, Y_X=Y_X,
        Y_NT_s=Y_NT_s, Y_NT_p=Y_NT_p, Y_NT_r=Y_NT_r,
        beta=beta, b=b, N_p=N_p,
    )
