import jax.numpy as jnp
from jax import jit

from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.interp_linear import interp_linear

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.safety.calc_shaft import calc_shaft
from gearopt.efficiency.calc_efficiency import calc_efficiency
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog


@jit
def stage_weight(x, i_st, i_sts, par, data):
    """
    JAX version of stage_weight()
    """

    # === Parameters ===
    rho_steel = par["rho_steel"]
    w_eff = par["w"]
    W_gb0 = par["W_gb0"]
    
    # === Unpack design variables ===
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # === Smooth planet count ===
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    # === Smooth module interpolation ===
    m_n_table = jnp.array([8., 10., 12., 16., 20., 25., 32., 40., 50.])
    idxs = jnp.arange(1, len(m_n_table) + 1)
    m_n = interp_linear(dm_n, idxs, m_n_table)

    # === Facewidth ===
    b = db * m_n

    # === Helix angle (smoothed via softclip) ===
    arg_raw = eps_beta * jnp.pi * m_n / jnp.maximum(b, 1e-12)
    arg_safe = softclip(arg_raw, -0.999, 0.999, 10.0)
    beta = jnp.arcsin(arg_safe)

    # === Geometry + Loads (JAX versions) ===
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # === Bearing fits (JAX versions) ===
    fits = fit_bearing_catalog()
    fit_d = fits["fit_d"]
    fit_W = fits["fit_W"]

    # Bearing width
    b_b = 0.5 * b
    d_i_p = geo["d_i_p"]
    d_o_b = fit_d(d_i_p, b_b)

    # === Shaft sizing ===
    _, _, reportSh = calc_shaft(x, i_sts, geo, loads, par)
    d_ss = reportSh["d_ss"]
    d_ps = reportSh["d_ps"]

    # === Geometry subset ===
    d_pw_s = geo["d_pw_s"]
    d_pw_p = geo["d_pw_p"]
    d_pw_r = geo["d_pw_r"]
    d_o_r  = geo["d_o_r"]
    a      = geo["a"]

    # === Constants ===
    pi_4_rho = jnp.pi * 0.25 * rho_steel
    k_PLC = 4.45

    # === Component weights ===
    W_sg = pi_4_rho * abs_smooth(d_pw_s**2 - d_ss**2) * b
    W_pg = pi_4_rho * abs_smooth(d_pw_p**2 - d_i_p**2) * b
    W_rg = pi_4_rho * abs_smooth(d_o_r**2 - d_pw_r**2) * b
    W_ss = pi_4_rho * d_ss**2 * (2 * b)
    W_ps = pi_4_rho * d_ps**2 * b

    # === Bearing weight ===
    W_pb = fit_W(d_o_b, b_b)

    # === Carrier weight ===
    W_plc = rho_steel * jnp.pi * 0.25 * a**2 * b * k_PLC

    # === Total weight ===
    W_st = (
        W_sg + W_ss
        + N_p * (W_pg + 2.0 * W_pb + W_ps)
        + W_rg + W_plc
    )

    # === Efficiency ===
    eta = calc_efficiency(x, i_sts, geo, loads, par)

    # === Objective ===
    J = w_eff*W_st/W_gb0 + (1-w_eff)*(1-eta)*100
    
    # === Report dictionary (JAX-friendly) ===
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
            "eta": eta,
            "w": w_eff,
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

    return J, reportW_k
