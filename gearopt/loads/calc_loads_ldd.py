import jax.numpy as jnp
from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.interp_linear import interp_linear

def calc_loads_ldd(x, geo, i_sts, par, data, return_all=False):
    # --- Parameters ---
    n0 = par["n0"]
    T0 = par["T0"]
    p_SF = par["p_SF"]
    p_SH = par["p_SH"]
    p_Sh = par["p_Sh"]
    T_max = par["T_max"]

    # --- Design variables ---
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # --- Smooth basics (module, b, N_p, beta) ---
    m_n_table = jnp.array([8, 10, 12, 16, 20, 25, 32, 40, 50], dtype=jnp.float32)
    idxs = jnp.arange(1, len(m_n_table) + 1)
    m_n = interp_linear(dm_n, idxs, m_n_table)

    b = db * m_n
    N_p = jnp.clip(Np_raw, 3.0, 5.0)
    beta = safe_asin(safe_div(eps_beta * jnp.pi * m_n, b))

    # --- Geometry subset ---
    alpha_t = geo["alpha_t"]
    d_p_s = geo["d_p_s"]
    d_p_r = geo["d_p_r"]
    a = geo["a"]

    # --- Torque spectrum (EXACTLY as NumPy version) ---
    w_i = data[:, 0]
    tbar = data[:, 1]

    T_eq_SH = (jnp.sum(w_i * tbar**p_SH)) ** (1.0 / p_SH) * T0
    T_eq_SF = (jnp.sum(w_i * tbar**p_SF)) ** (1.0 / p_SF) * T0
    T_eq_SB = (jnp.sum(w_i * tbar**(10.0 / 3.0))) ** (3.0 / 10.0) * T0
    T_eq_Sh = (jnp.sum(w_i * tbar**(1.0 / p_Sh))) ** (1.0 / p_Sh) * T0

    # --- Ratios ---
    z_s_safe = jnp.maximum(z_s, 1e-9)
    z_p_safe = jnp.maximum(z_p, 1e-9)
    Np_safe = jnp.maximum(N_p, 1e-9)

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

    # --- Tangential forces ---
    r_ps = 0.5 * (d_p_s * 1e-3)
    r_pr = 0.5 * (d_p_r * 1e-3)

    F_t_s_SH = jnp.abs(safe_div(T_s_SH, N_p * r_ps))
    F_t_s_SF = jnp.abs(safe_div(T_s_SF, N_p * r_ps))
    F_t_s_SB = jnp.abs(safe_div(T_s_SB, N_p * r_ps))

    F_t_r_SH = jnp.abs(safe_div(T_r_SH, N_p * r_pr))
    F_t_r_SF = jnp.abs(safe_div(T_r_SF, N_p * r_pr))
    F_t_r_SB = jnp.abs(safe_div(T_r_SB, N_p * r_pr))

    # --- Radial & axial forces ---
    F_r_s_SB = F_t_s_SB * jnp.tan(alpha_t) / jnp.cos(beta)
    F_r_r_SB = F_t_r_SB * jnp.tan(alpha_t) / jnp.cos(beta)

    F_a_s_SB = F_t_s_SB * jnp.tan(beta)
    F_a_r_SB = F_t_r_SB * jnp.tan(beta)

    F_x_SB = F_a_s_SB - F_a_r_SB
    F_y_SB = F_t_s_SB + F_t_r_SB
    F_z_SB = F_r_s_SB + F_r_r_SB

    F_rad = jnp.sqrt(F_y_SB**2 + F_z_SB**2)
    F_ax = jnp.abs(F_x_SB)

    # --- Speeds ---
    rpm_r = n0 * N_p * i_sts
    rpm_s = (n0 * i_sta - n0) * N_p * i_sts
    rpm_p = n0 * safe_div(z_r, z_p_safe) * i_sts
    rpm_b = n0 * (1.0 + safe_div(z_r, z_p_safe)) * i_sts

    # --- Pack outputs ---
    loads = {
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

    if return_all:
        loads.update({
            "T_eq_SH": T_eq_SH,
            "T_eq_SF": T_eq_SF,
            "T_eq_SB": T_eq_SB,
            "T_eq_Sh": T_eq_Sh,
            "i_sta": i_sta,
            "i_sts": i_sts,
            "beta": beta,
            "b": b,
            "N_p": N_p,
        })

    return loads
