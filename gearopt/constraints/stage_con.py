import jax
import jax.numpy as jnp

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.safety.calc_sf import calc_sf
from gearopt.safety.calc_sh import calc_sh
from gearopt.safety.calc_shaft import calc_shaft
from gearopt.safety.calc_l10h import calc_l10h
from gearopt.efficiency.calc_efficiency import calc_efficiency

from gearopt.util.safe_div import safe_div
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.interp_linear import interp_linear


def stage_con(x, i_st, i_sts, par, data):
    """
    JAX-compatible planetary stage constraint function.
    No NumPy. No Python floats. Fully differentiable.
    """
    calc_SB = True
    calc_SH = True
    calc_SF = True
    calc_Sh = True
    
    # === Parameters ===
    SF_min = par["SF_min"]
    SH_min = par["SH_min"]
    S_ShULS_min = par["S_ShULS_min"]

    # === Design variables ===
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # === Smoothed basics ===
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    m_n_table = jnp.array([8., 10., 12., 16., 20., 25., 32., 40., 50.])
    idxs = jnp.arange(1, m_n_table.size + 1)
    m_n = interp_linear(dm_n, idxs, m_n_table)

    b = db * m_n

    arg_raw = safe_div(eps_beta * jnp.pi * m_n, b)
    arg_safe = softclip(arg_raw, -0.999, 0.999, 10.0)
    beta = jnp.arcsin(arg_safe)
    beta_deg = beta * (180.0 / jnp.pi)

    # === Geometry + Loads (all must now be JAX-safe) ===
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # === Safety factors ===
    if calc_SF:
        SF_s, SF_p, SF_r = calc_sf(x, i_sts, geo, loads, par)       
    else:
        SF_s = SF_min
        SF_p = SF_min
        SF_r = SF_min

    if calc_SH:
        SH_s, SH_p, SH_r = calc_sh(x, i_sts, geo, loads, par)
    else:
        SH_s = SH_min
        SH_p = SH_min
        SH_r = SH_min

    cond_SF_s = SF_min - SF_s
    cond_SF_p = SF_min - SF_p
    cond_SF_r = SF_min - SF_r

    cond_SH_s = SH_min - SH_s
    cond_SH_p = SH_min - SH_p
    cond_SH_r = SH_min - SH_r

    # === Bearing life ===
    if calc_SB:
        SB, reportB = calc_l10h(x, i_sts, geo, loads, par, return_all=True)
    else:
        _, reportB = calc_l10h(x, i_sts, geo, loads, par, return_all=True)
        SB = 1.0
        
    cond_L10h = 1.0 - SB

    # === Shaft ULS ===
    if calc_Sh:
        S_ShULS_s, S_ShULS_p, reportSh = calc_shaft(x, i_sts, geo, loads, par)
        tol = 1e-3
        cond_ShULS_s = S_ShULS_min - S_ShULS_s - tol
        cond_ShULS_p = S_ShULS_min - S_ShULS_p - tol
    
        # Fit constraints
        cond_dss = safe_div(reportSh["d_ss"] - geo["d_i_s"], geo["d_i_s"])
        cond_dps = safe_div(reportSh["d_ps"] - reportB["d"], reportB["d"])
    else:
        _, _, reportSh = calc_shaft(x, i_sts, geo, loads, par)
        S_ShULS_s = S_ShULS_min
        S_ShULS_p = S_ShULS_min
        tol = 1e-3
        cond_ShULS_s = S_ShULS_min - S_ShULS_s - tol
        cond_ShULS_p = S_ShULS_min - S_ShULS_p - tol 
        cond_dss = 0.0
        cond_dps = 0.0
 

    # === Efficiency ===
    eta = calc_efficiency(x, i_sts, geo, loads, par)

    # === Geometric compatibility ===
    pi_Np = jnp.pi / N_p
    cond_adj = safe_div(geo["d_a_p"] - 2 * geo["a"] * jnp.sin(pi_Np),
                        geo["d_a_p"])

    # Mesh compatibility
    mesh_residual = jnp.sin(jnp.pi * (z_r + z_s) / N_p)
    cond_mesh = abs_smooth(mesh_residual) - 1e-3

    # Ratio tolerance
    i_12a = 1.0 + safe_div(z_r, z_s)
    cond_ratio = safe_div(abs_smooth(i_12a - i_st), jnp.maximum(i_st, 1e-12)) - 0.02

    # Profile shift constraints
    x_r = geo["x_r"]
    cond_x5 = -x_r - 0.6
    cond_x6 = x_r - 1.0

    # === Collect inequality constraints ===
    C = jnp.array([
        cond_adj,
        cond_ratio,
        cond_mesh,
        cond_SF_s, cond_SF_p, cond_SF_r,
        cond_SH_s, cond_SH_p, cond_SH_r,
        cond_x5, cond_x6,
        cond_L10h,
        cond_ShULS_s, cond_ShULS_p,
        cond_dss, cond_dps,
    ])

    Ceq = jnp.array([])

    # === Report dictionary ===
    report = {
        "gear": dict(
            m_n=m_n, z_s=z_s, z_p=z_p, z_r=z_r,
            b=b, x_s=x_s, x_p=x_p, x_r=x_r,
            i_st=i_st, beta_deg=beta_deg,
        ),
        "bearing": reportB,
        "shaft": reportSh,
        "safety": dict(
            SF_s=SF_s, SF_p=SF_p, SF_r=SF_r,
            SH_s=SH_s, SH_p=SH_p, SH_r=SH_r,
            SB=SB, S_ShULS_s=S_ShULS_s, S_ShULS_p=S_ShULS_p,
        ),
        "efficiency": dict(eta=eta),
    }

    return C, Ceq, report
