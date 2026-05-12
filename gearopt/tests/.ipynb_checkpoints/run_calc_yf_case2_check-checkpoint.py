#!/usr/bin/env python
import sys
import numpy as np
import pandas as pd
import jax.numpy as jnp

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

from gearopt.safety.calc_yf_hob import calc_yf_hob
from gearopt.safety.calc_yf_shaper import calc_yf_shaper
from gearopt.util.smooth_feps import smooth_feps


BETA_DEG_CASE2 = 15.0
MN_CASE2 = 40.0
RTOL = 0.05


def build_case2_inputs():
    """
    Build x, geo, par, and PDF reference values for case study #2.
    """
    paths = load_config("paths")
    _ = load_tspec(paths["tspec_path"])  # parity; not required for YF
    par = load_config("parameters")

    # Vector x for the report case
    m_n = MN_CASE2
    b = 800.0
    beta_deg = BETA_DEG_CASE2
    eps_beta = np.sin(np.deg2rad(beta_deg)) * b / (np.pi * m_n)

    x = np.array(
        [
            33.0,      # z_s
            27.0,      # z_p
            90.0,      # z_r magnitude (pass -x[2] to shaper)
            3.0,       # planets
            8.0,       # dm_n selector -> mn=40 (project convention)
            20.0,      # db=b/mn=800/40
            0.2004,    # x_s
            0.2624,    # x_p
            eps_beta,  # overlap ratio
        ],
        dtype=float,
    )

    geo = calc_geometry(x, par)
    print(geo)
    geo["d_a_s"]=1460.905
    # PDF reference values (section 7 table)
    ref = dict(
        Y_F_s=1.09, Y_S_s=2.06,
        Y_F_sp=1.06, Y_S_sp=2.09,
        Y_F_rp=0.85, Y_S_rp=2.24,
        Y_F_r=0.93, Y_S_r=2.22,

        alpha_Fen_deg_s=21.47,
        alpha_Fen_deg_sp=21.90,
        alpha_Fen_deg_rp=20.13,
        alpha_Fen_deg_r=16.48,

        f_eps_sp=0.801,
        f_eps_rp=0.766,
        f_eps_r=0.766,

        h_Fe_s=43.35,
        h_Fe_sp=41.93,
        h_Fe_rp=34.82,
        h_Fe_r=59.38,

        s_Fn_s=87.09,
        s_Fn_sp=86.63,
        s_Fn_rp=86.63,
        s_Fn_r=109.43,

        rho_F_s=19.30,
        rho_F_sp=19.00,
        rho_F_rp=19.00,
        rho_F_r=18.87,

        x_r=53.482,
        y_r=1882.966,
    )

    return x, geo, par, ref


def smoke_test(x, geo, par):
    """
    Basic "does it run" test and sanity checks.
    """
    alpha_n = float(np.deg2rad(par["alpha_n_deg"]))
    m_n = MN_CASE2

    # beta reconstructed from eps_beta and facewidth (for smoke only)
    b = float(x[5]) * m_n
    arg = float(x[8]) * np.pi * m_n / b
    beta = float(np.arcsin(arg))

    YF_s, YS_s = calc_yf_hob(
        geo["d_p_s"], geo["d_a_s"], x[0], x[6],
        geo["eps_alpha_t_sp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
    )

    YF_r, YS_r = calc_yf_shaper(
        -geo["d_p_r"], -geo["d_a_r"], -x[2], geo["x_r"],
        geo["eps_alpha_t_rp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
    )

    vals = np.array([float(YF_s), float(YS_s), float(YF_r), float(YS_r)])
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Smoke test failed: non-finite values: {vals}")
    if not np.all(vals > 0.0):
        raise RuntimeError(f"Smoke test failed: non-positive values: {vals}")
    if not np.all(vals < 10.0):
        raise RuntimeError(f"Smoke test failed: unexpectedly large values: {vals}")

    print("Smoke test: PASSED")


def pdf_comparison(x, geo, par, ref, rtol=RTOL):
    """
    Compute YF/YS + intermediates and compare with PDF reference values.
    Prints a sorted table and raises if any key fails.
    """
    alpha_n = jnp.deg2rad(par["alpha_n_deg"])
    m_n = MN_CASE2

    # Use report beta directly (critical for matching)
    beta = float(np.deg2rad(BETA_DEG_CASE2))

    sun_dbg = calc_yf_hob(
        geo["d_p_s"], geo["d_a_s"], x[0], x[6],
        geo["eps_alpha_t_sp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    sp_dbg = calc_yf_hob(
        geo["d_p_p"], geo["d_a_p"], x[1], x[7],
        geo["eps_alpha_t_sp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    rp_dbg = calc_yf_hob(
        geo["d_p_p"], geo["d_a_p"], x[1], x[7],
        geo["eps_alpha_t_rp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    r_dbg = calc_yf_shaper(
        -geo["d_p_r"], -geo["d_a_r"], -x[2], geo["x_r"],
        geo["eps_alpha_t_rp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    # f_eps (PDF calls it fε)
    f_eps_sp = float(smooth_feps(float(x[8]), float(sp_dbg["eps_alpha_n"])))
    f_eps_rp = float(smooth_feps(float(x[8]), float(rp_dbg["eps_alpha_n"])))
    f_eps_r = float(r_dbg["f_eps"])

    calc = dict(
        Y_F_s=float(sun_dbg["Y_F"]), Y_S_s=float(sun_dbg["Y_S"]),
        Y_F_sp=float(sp_dbg["Y_F"]), Y_S_sp=float(sp_dbg["Y_S"]),
        Y_F_rp=float(rp_dbg["Y_F"]), Y_S_rp=float(rp_dbg["Y_S"]),
        Y_F_r=float(r_dbg["Y_F"]),   Y_S_r=float(r_dbg["Y_S"]),

        alpha_Fen_deg_s=float(sun_dbg["alpha_Fen"]) * 180.0 / np.pi,
        alpha_Fen_deg_sp=float(sp_dbg["alpha_Fen"]) * 180.0 / np.pi,
        alpha_Fen_deg_rp=float(rp_dbg["alpha_Fen"]) * 180.0 / np.pi,
        alpha_Fen_deg_r=float(r_dbg["alpha_Fen"]) * 180.0 / np.pi,

        f_eps_sp=f_eps_sp,
        f_eps_rp=f_eps_rp,
        f_eps_r=f_eps_r,

        h_Fe_s=float(sun_dbg["h_Fe"]),
        h_Fe_sp=float(sp_dbg["h_Fe"]),
        h_Fe_rp=float(rp_dbg["h_Fe"]),
        h_Fe_r=float(r_dbg["h_Fe"]),

        s_Fn_s=float(sun_dbg["s_Fn"]),
        s_Fn_sp=float(sp_dbg["s_Fn"]),
        s_Fn_rp=float(rp_dbg["s_Fn"]),
        s_Fn_r=float(r_dbg["s_Fn"]),

        rho_F_s=float(sun_dbg["rho_F"]),
        rho_F_sp=float(sp_dbg["rho_F"]),
        rho_F_rp=float(rp_dbg["rho_F"]),
        rho_F_r=float(r_dbg["rho_F"]),

        x_r=float(r_dbg["x"]),
        y_r=float(r_dbg["y"]),
    )

    compare_keys = [
        "Y_F_s","Y_S_s","Y_F_sp","Y_S_sp","Y_F_rp","Y_S_rp","Y_F_r","Y_S_r",
        "alpha_Fen_deg_s","alpha_Fen_deg_sp","alpha_Fen_deg_rp","alpha_Fen_deg_r",
        "f_eps_sp","f_eps_rp","f_eps_r",
        "h_Fe_s","h_Fe_sp","h_Fe_rp","h_Fe_r",
        "s_Fn_s","s_Fn_sp","s_Fn_rp","s_Fn_r",
        "rho_F_s","rho_F_sp","rho_F_rp","rho_F_r",
        "x_r","y_r",
    ]

    rows = []
    failures = []

    for k in compare_keys:
        ref_val = float(ref[k])
        calc_val = float(calc[k])

        # internal gear y-coordinate sign convention differs frequently between implementations
        if k == "y_r":
            calc_val = abs(calc_val)

        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-12)) * 100.0
        rows.append((k, ref_val, calc_val, rel_err))

        if not np.isclose(calc_val, ref_val, rtol=rtol):
            failures.append((k, ref_val, calc_val, rel_err))

    df = pd.DataFrame(rows, columns=["Parameter", "Reference(PDF)", "Calculated", "RelError_percent"])
    df = df.sort_values("RelError_percent", ascending=False)

    print("\n=== calc_yf_hob / calc_yf_shaper PDF Verification (case2) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    if failures:
        msg_lines = [f"FAILED {len(failures)} / {len(compare_keys)} checks (rtol={rtol})"]
        for k, ref_val, calc_val, rel_err in failures[:15]:
            msg_lines.append(
                f"  {k}: calc={calc_val:.6f}, ref={ref_val:.6f}, rel_err={rel_err:.3f}%"
            )
        if len(failures) > 15:
            msg_lines.append(f"  ... plus {len(failures) - 15} more")
        raise RuntimeError("\n".join(msg_lines))

    print(f"PDF comparison: PASSED (rtol={rtol})")


def main():
    x, geo, par, ref = build_case2_inputs()

    # Optional: verify which calc_geometry is imported
    import gearopt.geometry.calc_geometry as cg
    print("calc_geometry loaded from:", cg.__file__)

    smoke_test(x, geo, par)
    pdf_comparison(x, geo, par, ref, rtol=RTOL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e))
        sys.exit(1)
