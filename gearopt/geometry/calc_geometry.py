import numpy as np
from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.geometry.tipAlterationAnalytic import tipAlterationAnalytic
from gearopt.config.load_config import load_config

def calc_geometry(x, par):
    """
    Gradient-friendly gear geometry calculation.
    Vectorized and smooth (no discontinuities).
    """

    # --- Parameters ---
    alpha_n = np.deg2rad(par["alpha_n_deg"])
    s_R_s = par["s_R_s"]
    s_R_p = par["s_R_p"]
    s_R_r = par["s_R_r"]

    # --- Standard module table ---
    m_n_table = np.array([8, 10, 12, 16, 20, 25, 32, 40, 50])

    # --- Design variables ---
    dm_n, z_s, z_p, z_r, Np_raw, db, x_s, x_p, eps_beta = x

    # --- Smooth module interpolation ---
    idxs = np.arange(1, len(m_n_table) + 1)
    m_n = np.interp(dm_n, idxs, m_n_table)

    # --- Facewidth & planet count ---
    b = db * m_n
    N_p = np.clip(Np_raw, 3.0, 5.0)

    # --- Helix angle ---
    arg_raw = safe_div(eps_beta * np.pi * m_n, b)
    beta = safe_asin(arg_raw)

    # --- Precompute trig values ---
    cos_beta = np.cos(beta)
    tan_alpha_n = np.tan(alpha_n)
    cos_alpha_n = np.cos(alpha_n)

    m_t = m_n / cos_beta
    alpha_t = np.arctan(tan_alpha_n / cos_beta)
    beta_b = safe_asin(np.sin(beta) * cos_alpha_n)
    cos_alpha_t = np.cos(alpha_t)

    # --- Pitch & base diameters ---
    d_p_s = z_s * m_t
    d_p_p = z_p * m_t
    d_p_r = z_r * m_t
    d_b_s = d_p_s * cos_alpha_t
    d_b_p = d_p_p * cos_alpha_t
    d_b_r = d_p_r * cos_alpha_t

    # --- Working pressure angle (sun–planet) ---
    inv_alpha_wt_sp = (
        2 * tan_alpha_n * (x_s + x_p) / max(z_s + z_p, 1e-9)
        + np.tan(alpha_t) - alpha_t
    )
    alpha_guess = alpha_t + 0.1
    for _ in range(4):
        f = np.tan(alpha_guess) - alpha_guess - inv_alpha_wt_sp
        df = 1.0 / np.maximum(np.cos(alpha_guess) ** 2, 1e-9) - 1
        alpha_guess = alpha_guess - f / np.maximum(df, 1e-9)
    alpha_wt_sp = alpha_guess
    cos_alpha_wt_sp = np.cos(alpha_wt_sp)

    # --- Working pitch diameters ---
    d_pw_s = d_b_s / np.maximum(cos_alpha_wt_sp, 1e-9)
    d_pw_p = d_b_p / np.maximum(cos_alpha_wt_sp, 1e-9)

    # --- Center distance ---
    y_sp = 0.5 * (z_s + z_p) * (cos_alpha_t / np.maximum(cos_alpha_wt_sp, 1e-9) - 1)
    a = ((z_s + z_p) * 0.5 + y_sp) * m_t

    # --- Ring–planet mesh ---
    den_rp = np.maximum(z_r - z_p, 1e-9)
    d_pw_r = 2 * a * z_r / den_rp
    d_pw_p_rp = 2 * a * z_p / den_rp
    arg_rp = d_b_r / np.maximum(d_pw_r, 1e-9)
    alpha_wt_rp = safe_asin(safe_sqrt(1 - arg_rp**2))

    # --- Profile shift (ring) ---
    x_rp = (
        -(np.tan(alpha_wt_rp) - alpha_wt_rp - np.tan(alpha_t) + alpha_t)
        * 0.5
        / np.maximum(tan_alpha_n, 1e-9)
        * (z_r - z_p)
    )
    x_r = x_rp - x_p

    # --- Diameters ---
    k_tip_sp = tipAlterationAnalytic(d_b_s / 2, d_b_p / 2, a, alpha_t, m_n)
    d_a_s = d_p_s + 2 * (1 + x_s + k_tip_sp) * m_n
    d_a_p = d_p_p + 2 * (1 + x_p + k_tip_sp) * m_n
    d_a_r = d_p_r - 2 * (1 + x_r) * m_n
    d_f_s = d_a_s - 2 * (2.25 + k_tip_sp) * m_n
    d_f_p = d_a_p - 2 * (2.25 + k_tip_sp) * m_n
    d_f_r = d_a_r + 2 * 2.25 * m_n
    d_i_s = d_f_s - 2 * s_R_s * m_n
    d_i_p = d_f_p - 2 * s_R_p * m_n
    d_o_r = d_f_r + 2 * s_R_r * m_n

    # --- Contact ratios ---
    den_CR = np.maximum(2 * np.pi * m_t * cos_alpha_t, 1e-9)
    term_s = safe_sqrt(d_a_s**2 - d_b_s**2)
    term_p = safe_sqrt(d_a_p**2 - d_b_p**2)
    term_r = safe_sqrt(d_a_r**2 - d_b_r**2)
    eps_alpha_t_sp = (term_s + term_p - 2 * a * np.sin(alpha_wt_sp)) / den_CR
    eps_alpha_t_rp_raw = (term_p - term_r + 2 * a * np.sin(alpha_wt_rp)) / den_CR
    eps_alpha_t_rp = abs_smooth(eps_alpha_t_rp_raw)
    eps_1_sp = (term_s - d_pw_s * np.sin(alpha_wt_sp)) / den_CR
    eps_2_sp = (term_p - d_pw_p * np.sin(alpha_wt_sp)) / den_CR
    eps_1_rp_raw = (term_p - d_pw_p_rp * np.sin(alpha_wt_rp)) / den_CR
    eps_1_rp = softclip(eps_1_rp_raw, 0, eps_alpha_t_rp, 20)
    eps_2_rp_raw = (-term_r + d_pw_r * np.sin(alpha_wt_rp)) / den_CR
    eps_2_rp = softclip(eps_2_rp_raw, 0, 5, 20)

    # --- Output (dictionary instead of array) ---
    geo = {
        # General geometry
        "m_t": m_t,
        "alpha_t": alpha_t,
        "beta_b": beta_b,
        "alpha_wt_sp": alpha_wt_sp,
        "alpha_wt_rp": alpha_wt_rp,
    
        # Pitch, base, and bore diameters
        "d_p_s": d_p_s,
        "d_p_p": d_p_p,
        "d_p_r": d_p_r,
        "d_b_s": d_b_s,
        "d_b_p": d_b_p,
        "d_b_r": d_b_r,
    
        # Working pitch diameters
        "d_pw_s": d_pw_s,
        "d_pw_p": d_pw_p,
        "d_pw_r": d_pw_r,
    
        # Addendum and dedendum diameters
        "d_a_s": d_a_s,
        "d_a_p": d_a_p,
        "d_a_r": d_a_r,
        "d_f_s": d_f_s,
        "d_f_p": d_f_p,
        "d_f_r": d_f_r,
    
        # Bore and outer diameters
        "d_i_s": d_i_s,
        "d_i_p": d_i_p,
        "d_o_r": d_o_r,
    
        # Derived and auxiliary quantities
        "y_sp": y_sp,
        "a": a,
        "x_r": x_r,
    
        # Overlap and contact ratios
        "eps_alpha_t_sp": eps_alpha_t_sp,
        "eps_alpha_t_rp": eps_alpha_t_rp,
        "eps_beta": eps_beta,
        "eps_1_sp": eps_1_sp,
        "eps_2_sp": eps_2_sp,
        "eps_1_rp": eps_1_rp,
        "eps_2_rp": eps_2_rp,
    }
    
    return geo
