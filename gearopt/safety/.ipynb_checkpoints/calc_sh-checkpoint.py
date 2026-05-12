import jax.numpy as jnp

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.smoothmin2 import smoothmin2
from gearopt.util.interp_linear import interp_linear

from gearopt.safety.calc_zeps import calc_zeps
from gearopt.safety.calc_zbd import calc_zbd
from gearopt.safety.calc_znt import calc_znt
from gearopt.safety.calc_zr import calc_zr


def calc_sh(x, i_sts, geo, loads, par, return_all=False):
    """
    JAX-compatible ISO 6336 contact (pitting) safety factors.
    Fully differentiable; no numpy; no Python floats inside computation.
    """

    # ===== Parameters =====
    alpha_n = jnp.deg2rad(par["alpha_n_deg"])
    h_a = par["h_a"]
    s_R_s = par["s_R_s"]
    s_R_p = par["s_R_p"]
    s_R_r = par["s_R_r"]

    K_A = par["K_A"]
    K_V = par["K_V"]
    K_Halpha = par["K_Halpha"]
    K_Hbeta_sp = par["K_Hbeta_sp"]
    K_Hbeta_rp = par["K_Hbeta_rp"]
    K_gamma = par["K_gamma"]
    calc_K_gamma = par["calc_K_gamma"]

    Z_L = par["Z_L"]
    Z_V = par["Z_V"]
    Z_W = par["Z_W"]
    Z_X = par["Z_X"]

    sigma_Hlim = par["sigma_Hlim"]
    E_steel = par["E_steel"]
    nu_steel = par["nu_steel"]
    Rz = par["Rz"]
    

    # ===== Design variables =====
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # ==== Smooth module table lookup ====
    m_n_table = jnp.array([8.,10.,12.,16.,20.,25.,32.,40.,50.])
    idxs       = jnp.arange(1, m_n_table.size + 1)
    m_n        = interp_linear(dm_n, idxs, m_n_table)

    b   = db * m_n
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    arg = safe_div(eps_beta * jnp.pi * m_n, b)
    beta = safe_asin(arg)

    # ===== Geometry unpack =====
    alpha_t     = geo["alpha_t"]
    beta_b      = geo["beta_b"]
    alpha_wt_sp = geo["alpha_wt_sp"]
    alpha_wt_rp = geo["alpha_wt_rp"]

    d_p_s = geo["d_p_s"]
    d_p_p = geo["d_p_p"]
    d_p_r = geo["d_p_r"]
    d_b_s = geo["d_b_s"]
    d_b_p = geo["d_b_p"]
    d_b_r = geo["d_b_r"]

    d_a_s = geo["d_a_s"]
    d_a_p = geo["d_a_p"]
    d_a_r = geo["d_a_r"]

    eps_alpha_t_sp = geo["eps_alpha_t_sp"]
    eps_alpha_t_rp = geo["eps_alpha_t_rp"]

    x_r = geo["x_r"]

    # ===== Loads =====
    F_t_s = loads["F_t_s_SH"]
    F_t_r = loads["F_t_r_SH"]
    rpm_s = loads["rpm_s"]
    rpm_p = loads["rpm_p"]
    rpm_r = loads["rpm_r"]

    # =====================================================
    #   ISO CONTACT GEOMETRY FACTORS
    # =====================================================

    Z_H_sp = safe_sqrt(
        safe_div(2.0 * jnp.cos(beta_b) * jnp.cos(alpha_wt_sp),
                 (jnp.cos(alpha_t)**2) * jnp.sin(alpha_wt_sp))
    )

    Z_H_rp = safe_sqrt(
        safe_div(2.0 * jnp.cos(beta_b) * jnp.cos(alpha_wt_rp),
                 (jnp.cos(alpha_t)**2) * jnp.sin(alpha_wt_rp))
    )

    Z_E = safe_sqrt(safe_div(E_steel, 2.0 * jnp.pi * (1.0 - nu_steel**2)))

    Z_eps_sp = calc_zeps(eps_alpha_t_sp, eps_beta)
    Z_eps_rp = calc_zeps(eps_alpha_t_rp, eps_beta)

    Z_beta = safe_div(1.0, safe_sqrt(jnp.cos(beta)))

    # =====================================================
    #   Base Hertzian Stress
    # =====================================================

    sigma_H0_sp = (
        Z_H_sp * Z_E * Z_eps_sp * Z_beta *
        safe_sqrt(
            safe_div(F_t_s * (z_s + z_p),
                     d_p_s * b * z_p)
        )
    )

    sigma_H0_rp = (
        Z_H_rp * Z_E * Z_eps_rp * Z_beta *
        safe_sqrt(
            safe_div(F_t_r * ( -z_r + z_p),
                     d_p_p * b * (-z_r))
        )
    )

    # ===== Size & boundary factors =====
    Z_B_sp, Z_D_sp = calc_zbd()
    Z_B_rp, _      = calc_zbd()
    Z_D_rp         = 1.0   # ring planets: size correction = 1

    # ===== Planet load sharing factor =====
    if calc_K_gamma:
        K_gamma = interp_linear(
            N_p,
            jnp.array([3.0, 4.0, 5.0]),
            jnp.array([1.00, 1.20, 1.35])
        )

    mult_sp = safe_sqrt(K_A * K_gamma * K_V * K_Hbeta_sp * K_Halpha)
    mult_rp = safe_sqrt(K_A * K_gamma * K_V * K_Hbeta_rp * K_Halpha)

    sigma_HB_sp = Z_B_sp * sigma_H0_sp * mult_sp
    sigma_HD_sp = Z_D_sp * sigma_H0_sp * mult_sp
    sigma_HB_rp = Z_B_rp * sigma_H0_rp * mult_rp
    sigma_HD_rp = Z_D_rp * sigma_H0_rp * mult_rp

    # ===== Roughness & life factors =====
    Z_R_sp = calc_zr(Rz, d_b_s, d_b_p, alpha_wt_sp)
    Z_R_rp = calc_zr(Rz, d_b_p, -d_b_r, alpha_wt_rp)

    Z_NT_s = calc_znt(rpm_s)
    Z_NT_p = calc_znt(rpm_p)
    Z_NT_r = calc_znt(rpm_r)

    sigma_HG_s  = sigma_Hlim * Z_NT_s * Z_L * Z_V * Z_R_sp * Z_W * Z_X
    sigma_HG_sp = sigma_Hlim * Z_NT_p * Z_L * Z_V * Z_R_sp * Z_W * Z_X
    sigma_HG_rp = sigma_Hlim * Z_NT_p * Z_L * Z_V * Z_R_rp * Z_W * Z_X
    sigma_HG_r  = sigma_Hlim * Z_NT_r * Z_L * Z_V * Z_R_rp * Z_W * Z_X

    # ===== Safety factors =====
    SH_s  = safe_div(sigma_HG_s,  sigma_HB_sp)
    SH_sp = safe_div(sigma_HG_sp, sigma_HD_sp)
    SH_rp = safe_div(sigma_HG_rp, sigma_HB_rp)
    SH_r  = safe_div(sigma_HG_r,  sigma_HD_rp)

    # Planet mesh: smooth minimum
    SH_p = smoothmin2(SH_sp, SH_rp, k=30.0)

    if not return_all:
        return SH_s, SH_p, SH_r

    # ===== Full report =====
    return dict(
        SH_s=SH_s, SH_p=SH_p, SH_r=SH_r,
        Z_H_sp=Z_H_sp, Z_H_rp=Z_H_rp,
        Z_E=Z_E, Z_beta=Z_beta,
        sigma_H0_sp=sigma_H0_sp,
        sigma_H0_rp=sigma_H0_rp,
        sigma_HB_sp=sigma_HB_sp,
        sigma_HB_rp=sigma_HB_rp,
        sigma_HG_s=sigma_HG_s,
        sigma_HG_r=sigma_HG_r,
        Z_R_sp=Z_R_sp,
        Z_R_rp=Z_R_rp,
        rpm_s=rpm_s, rpm_p=rpm_p, rpm_r=rpm_r,
        beta=beta, b=b, N_p=N_p, m_n=m_n
    )
