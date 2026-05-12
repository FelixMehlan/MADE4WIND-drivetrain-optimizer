import jax.numpy as jnp

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.interp_linear import interp_linear
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog


def calc_shaft(x, i_sts, geo, loads, par, return_all=False):
    """
    JAX-differentiable shaft ULS calculation.
    Fully replaces original NumPy implementation.
    """

    # === Parameters ===
    rho_steel = par["rho_steel"]
    sigma_Y = par["sigma_Y"]
    S_ShULS_min = par["S_ShULS_min"]

    # === Design variables ===
    # (same unpacking order as your revised convention)
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # === Smooth module interpolation ===
    m_n_table = jnp.array([8., 10., 12., 16., 20., 25., 32., 40., 50.])
    idxs = jnp.arange(1, m_n_table.shape[0] + 1)

    dm_n_sc = softclip(dm_n, 1.0, float(len(m_n_table)), 20.0)
    m_n = interp_linear(dm_n_sc, idxs, m_n_table)

    # === Basic geometry ===
    b = db * m_n
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    arg_raw = safe_div(eps_beta * jnp.pi * m_n, b)
    beta = safe_asin(arg_raw)

    # === Geometry values ===
    d_i_p = geo["d_i_p"]

    # === Loads ===
    T_s_ShULS = loads["T_s_ShULS"]
    T_p_ShULS = loads["T_p_ShULS"]

    # === Bearing fits (these are vectorizable, so JAX-safe) ===
    fits = fit_bearing_catalog()
    fit_d = fits["fit_d"]

    B = 0.5 * b
    D = d_i_p

    d_ps = fit_d(D, B)
    d_ps = jnp.maximum(d_ps, 1e-6)

    # === Convert torques to N·mm ===
    T_s_Nmm = T_s_ShULS * 1e3
    T_p_Nmm = T_p_ShULS * 1e3

    # === Shaft diameter from torsion ULS ===
    d_ss = abs_smooth(
        safe_div(16 * jnp.sqrt(3) * T_s_Nmm * S_ShULS_min,
                 jnp.pi * sigma_Y)
    ) ** (1 / 3)

    # === Polar moment of inertia ===
    I_ss = (jnp.pi / 32.0) * d_ss**4
    I_ps = (jnp.pi / 32.0) * d_ps**4

    # === Shear stresses ===
    tau_ss = safe_div(T_s_Nmm * d_ss, 2 * I_ss)
    tau_ps = safe_div(T_p_Nmm * d_ps, 2 * I_ps)

    sigma_vm_ss = jnp.sqrt(3.0) * tau_ss
    sigma_vm_ps = jnp.sqrt(3.0) * tau_ps

    # === Safety factors ===
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

    # JAX: return tracers/scalars directly (no float() conversion)
    return S_ShULS_s, S_ShULS_p, reportSh

