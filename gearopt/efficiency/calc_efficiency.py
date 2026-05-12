import jax
import jax.numpy as jnp

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.interp_linear import interp_linear   # your JAX interp
from gearopt.config.load_config import load_config


def calc_efficiency(x, i_sts, geo, loads, par, return_all=False):
    """
    Fully JAX-differentiable gear mesh efficiency model.
    Equivalent to your NumPy version but JAX-safe.

    Parameters
    ----------
    x : array
    i_sts : float
    geo : dict
    loads : dict
    par : dict
    return_all : bool

    Returns
    -------
    eff : scalar (JAX array)
    OR dict of detailed values
    """

    # --- Parameters ---
    n0 = par["n0"]
    T0 = par["T0"]
    alpha_n = jnp.deg2rad(par["alpha_n_deg"])

    # --- Geometry extraction ---
    d_p_s = geo["d_p_s"]
    d_p_p = geo["d_p_p"]
    d_p_r = geo["d_p_r"]
    d_b_s = geo["d_b_s"]
    d_b_p = geo["d_b_p"]
    d_b_r = geo["d_b_r"]

    d_pw_s = geo["d_pw_s"]
    d_pw_p = geo["d_pw_p"]
    d_pw_r = geo["d_pw_r"]

    alpha_wt_sp = geo["alpha_wt_sp"]
    alpha_wt_rp = geo["alpha_wt_rp"]
    beta_b = geo["beta_b"]

    eps_alpha_t_sp = geo["eps_alpha_t_sp"]
    eps_alpha_t_rp = geo["eps_alpha_t_rp"]
    eps_1_sp = geo["eps_1_sp"]
    eps_2_sp = geo["eps_2_sp"]
    eps_1_rp = geo["eps_1_rp"]
    eps_2_rp = geo["eps_2_rp"]
    a = geo["a"]

    # --- Loads ---
    F_rad = loads["F_rad"]
    F_ax = loads["F_ax"]
    rpm_r = loads["rpm_r"]
    rpm_s = loads["rpm_s"]
    rpm_p = loads["rpm_p"]
    rpm_b = loads["rpm_b"]

    # --- Design variables (same ordering as rest of JAX code) ---
    z_s, z_p_cont, z_r_cont, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # ===========================================================
    #              SMOOTH GEOMETRIC BASICS
    # ===========================================================

    # smooth module interpolation
    m_n_table = jnp.array([8., 10., 12., 16., 20., 25., 32., 40., 50.])
    idxs = jnp.arange(1, m_n_table.shape[0] + 1)
    m_n = interp_linear(dm_n, idxs, m_n_table)

    b = db * m_n
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    beta = safe_asin(safe_div(eps_beta * jnp.pi * m_n, b))

    # --- Precomputed trig ---
    cos_beta_b = jnp.cos(beta_b)

    sin_alpha_wt_sp = jnp.sin(alpha_wt_sp)
    sin_alpha_wt_rp = jnp.sin(alpha_wt_rp)

    # ===========================================================
    #                   OIL PROPERTIES
    # ===========================================================
    nu_40, nu_100 = 220.0, 17.5
    T1, T2, T_op = 313.0, 373.0, 343.0

    log_nu_40 = jnp.log10(jnp.log10(nu_40 + 0.7))
    log_nu_100 = jnp.log10(jnp.log10(nu_100 + 0.7))

    A = safe_div(log_nu_40 - log_nu_100, jnp.log10(T1) - jnp.log10(T2))
    B = log_nu_40 - A * jnp.log10(T1)

    log_nu_op = A * jnp.log10(T_op) + B
    nu_op = 10.0 ** (10.0 ** log_nu_op) - 0.7

    rho_ref, alpha_th = 0.895, 0.0007
    rho_op = rho_ref * (1 - alpha_th * (70 - 15))
    eta_oil = nu_op * rho_op

    # ===========================================================
    #       CURVATURE RADII + SLIDING VELOCITIES
    # ===========================================================
    R_a = 0.6  # µm

    rho_s = 0.5 * d_pw_s * sin_alpha_wt_sp
    rho_p = 0.5 * d_pw_p * sin_alpha_wt_sp
    rho_r = 0.5 * d_pw_r * sin_alpha_wt_rp

    rho_C_sp = safe_div(1.0, safe_div(1.0, rho_s) + safe_div(1.0, rho_p))
    rho_C_rp = safe_div(1.0, jnp.abs(safe_div(1.0, rho_r) - safe_div(1.0, rho_p)))

    # ===========================================================
    #             SPEEDS & SLIDING VELOCITIES
    # ===========================================================
    i_sta = 1.0 + safe_div(z_r_cont, z_s)
    omega_base = i_sts * (n0 / 60.0) * 2.0 * jnp.pi

    v_p = safe_div(z_r_cont, z_s) * omega_base * d_pw_s * 0.5
    v_s = safe_div(z_r_cont, z_p_cont) * omega_base * d_pw_p * 0.5
    v_r = omega_base * d_pw_r * 0.5

    V_sumC_sp = abs_smooth(v_p + v_s) * 1e-3 * sin_alpha_wt_sp
    V_sumC_rp = abs_smooth(v_r + v_p) * 1e-3 * sin_alpha_wt_rp

    # ===========================================================
    #                    TANGENTIAL FORCES
    # ===========================================================
    T_s = safe_div(T0, i_sta * i_sts)
    T_r = safe_div(T0, i_sts) - T_s

    F_tb_sp = safe_div(T_s, N_p * (0.5 * d_p_s * 1e-3)) * safe_div(d_p_s, d_b_s)
    F_tb_rp = safe_div(T_r, N_p * (0.5 * d_p_r * 1e-3)) * safe_div(d_p_r, d_b_r)

    # ===========================================================
    #                   LOAD FACTORS
    # ===========================================================
    X_L_sp = (safe_div(F_tb_sp, b)) ** -0.0651
    X_L_rp = (safe_div(F_tb_rp, b)) ** -0.0651

    # ===========================================================
    #                 FRICTION COEFFICIENTS
    # ===========================================================
    mu_m_sp = (
        0.048
        * (safe_div(F_tb_sp, b * V_sumC_sp * rho_C_sp)) ** 0.2
        * (eta_oil ** -0.05)
        * (R_a ** 0.25)
        * X_L_sp
    )

    mu_m_rp = (
        0.048
        * (safe_div(F_tb_rp, b * V_sumC_rp * rho_C_rp)) ** 0.2
        * (eta_oil ** -0.05)
        * (R_a ** 0.25)
        * X_L_rp
    )

    # ===========================================================
    #                     POWER LOSS FACTORS
    # ===========================================================
    H_v_sp = safe_div((z_p_cont / z_s + 1) * jnp.pi, z_p_cont * cos_beta_b) * \
             (1 - eps_alpha_t_sp + eps_1_sp**2 + eps_2_sp**2)

    H_v_rp = safe_div((-z_r_cont / z_p_cont + 1) * jnp.pi, -z_r_cont * cos_beta_b) * \
             (1 - eps_alpha_t_rp + eps_1_rp**2 + eps_2_rp**2)

    # ===========================================================
    #                       POWER LOSSES
    # ===========================================================
    omega_s = (i_sta - 1.0) * n0 * i_sts / 60.0 * 2.0 * jnp.pi
    omega_rp = abs_smooth(-n0 * i_sts / 60.0 * 2.0 * jnp.pi)

    P_mesh_sp = T_s * omega_s
    P_mesh_rp = T_r * omega_rp

    P_V_sp = P_mesh_sp * mu_m_sp * H_v_sp
    P_V_rp = P_mesh_rp * mu_m_rp * H_v_rp

    P_V = P_V_sp + P_V_rp

    # ===========================================================
    #                     EFFICIENCY
    # ===========================================================
    P0 = T0 * (n0 / 60.0) * 2 * jnp.pi
    eff_raw = 1.0 - safe_div(P_V, P0)
    eff = jnp.clip(eff_raw, 0.0, 1.0)

    # ===========================================================
    #                   RETURN VALUES
    # ===========================================================
    if return_all:
        return dict(
            eff=eff,
            mu_m_sp=mu_m_sp,
            mu_m_rp=mu_m_rp,
            H_v_sp=H_v_sp,
            H_v_rp=H_v_rp,
            P_mesh_sp=P_mesh_sp,
            P_mesh_rp=P_mesh_rp,
            P_V_sp=P_V_sp,
            P_V_rp=P_V_rp,
            P_V=P_V,
            P0=P0,
            eta_oil=eta_oil,
            rho_C_sp=rho_C_sp,
            rho_C_rp=rho_C_rp,
            V_sumC_sp=V_sumC_sp,
            V_sumC_rp=V_sumC_rp,
            F_tb_sp=F_tb_sp,
            F_tb_rp=F_tb_rp,
            T_s=T_s,
            T_r=T_r,
            omega_s=omega_s,
            omega_rp=omega_rp,
        )

    return eff
