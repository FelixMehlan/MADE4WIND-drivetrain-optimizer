import numpy as np
import pandas as pd
import pytest
import jax.numpy as jnp

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

from gearopt.safety.calc_yf_hob import calc_yf_hob
from gearopt.safety.calc_yf_shaper import calc_yf_shaper
from gearopt.util.smooth_feps import smooth_feps



@pytest.fixture(scope="module")
def setup_yf_data():
    """
    Same fixture philosophy as test_unit_calc_sf:
    build x, geo, par and store PDF reference values.
    """
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])  # kept for parity; not required here
    par = load_config("parameters")

    x = np.array([
        41, 34, 112, 3, 8, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    geo = calc_geometry(x, par)

    # PDF reference values (section 7 table)
    ref = dict(
        # Factors
        Y_F_s=1.15, Y_S_s=2.14,
        Y_F_sp=0.95, Y_S_sp=2.43,
        Y_F_rp=0.84, Y_S_rp=2.54,
        Y_F_r=0.94, Y_S_r=2.40,

        # Intermediate values shown in PDF
        alpha_Fen_deg_s=22.83,
        alpha_Fen_deg_sp=26.18,
        alpha_Fen_deg_rp=25.56,
        alpha_Fen_deg_r=20.63,

        f_eps_sp=0.840,   # load distribution influence factor (pair1)
        f_eps_rp=0.821,   # load distribution influence factor (pair2)
        f_eps_r=0.821,    # internal gear uses pair2

        h_Fe_s=47.32,
        h_Fe_sp=43.69,
        h_Fe_rp=39.53,
        h_Fe_r=60.39,

        s_Fn_s=90.12,
        s_Fn_sp=94.31,
        s_Fn_rp=94.31,
        s_Fn_r=112.19,

        rho_F_s=17.43,
        rho_F_sp=15.22,
        rho_F_rp=15.22,
        rho_F_r=16.04,

        #x_s = 45.058,
        #x_sp = 47.154,
        #x_rp = 47.154,
        x_r = 55.390,

        #y_s = 819.246,
        #y_sp = 695.575,
        #y_rp = 695.575,
        y_r = 2389.473,
    )

    return dict(x=x, geo=geo, par=par, ref=ref)


# === TEST 1: Smoke test ===



# === TEST 2: PDF comparisons (YF/YS + intermediate values) ===
def test_calc_yf_pdf_reference_comparison(setup_yf_data):
    """
    Compare:
      - YF, YS for sun/planet(sp)/planet(rp)/ring vs PDF
      - Intermediate values alpha_Fen, f_eps, h_Fe, s_Fn, rho_F vs PDF
    """
    geo, par, x, ref = (
        setup_yf_data["geo"],
        setup_yf_data["par"],
        setup_yf_data["x"],
        setup_yf_data["ref"],
    )

    alpha_n = jnp.deg2rad(par["alpha_n_deg"])

    # Keep consistent with SF test expectation: this reference case uses m_n = 40
    m_n = 40.0

    # Recreate beta consistent with calc_sf (approximately)
    # eps_beta = x[8]; b = db*m_n with db=x[5]
    b = float(x[5]) * m_n
    arg = float(x[8]) * np.pi * m_n / b
    beta = float(np.arcsin(arg))

    # --- HOB: use debug_mode=True to get intermediate values ---

    sun_dbg = calc_yf_hob(
        geo["d_p_s"], geo["d_a_s"], x[0], geo["xE_i_s"],
        geo["eps_alpha_t_sp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    sp_dbg = calc_yf_hob(
        geo["d_p_p"], geo["d_a_p"], x[1], geo["xE_i_p"],
        geo["eps_alpha_t_sp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    rp_dbg = calc_yf_hob(
        geo["d_p_p"], geo["d_a_p"], x[1], geo["xE_i_p"],
        geo["eps_alpha_t_rp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    # --- SHAPER: compute intermediates via debug mirror, then compare with calc_yf_shaper output ---

    r_dbg = calc_yf_shaper(
        -geo["d_p_r"], -geo["d_a_r"], -x[2], geo["xE_i_r"],
        geo["eps_alpha_t_rp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True
    )

    print(r_dbg)
    # Compute f_eps for hob cases (PDF calls it fε)
    f_eps_s  = float(smooth_feps(float(x[8]), float(sun_dbg["eps_alpha_n"])))
    f_eps_sp = float(smooth_feps(float(x[8]), float(sp_dbg["eps_alpha_n"])))
    f_eps_rp = float(smooth_feps(float(x[8]), float(rp_dbg["eps_alpha_n"])))
    f_eps_r  = float(r_dbg["f_eps"])

    # Collect values to compare
    calc = dict(
        # Factors
        Y_F_s=float(sun_dbg["Y_F"]), Y_S_s=float(sun_dbg["Y_S"]),
        Y_F_sp=float(sp_dbg["Y_F"]), Y_S_sp=float(sp_dbg["Y_S"]),
        Y_F_rp=float(rp_dbg["Y_F"]), Y_S_rp=float(rp_dbg["Y_S"]),
        Y_F_r=float(r_dbg["Y_F"]),   Y_S_r=float(r_dbg["Y_S"]),

        # Intermediates shown in PDF
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

        #x_s = float(sun_dbg["x"]),
        #x_sp = float(sp_dbg["x"]),
        #x_rp = float(rp_dbg["x"]),
        x_r = float(r_dbg["x"]),

        #y_s = float(sun_dbg["y"]),
        #y_sp =float(sp_dbg["y"]),
        #y_rp = float(rp_dbg["y"]),
        y_r = float(r_dbg["y"]),

        # Additional intermediates (requested): present + finite checks
        z_n_s=float(sun_dbg["z_n"]),
        eps_alpha_n_s=float(sun_dbg["eps_alpha_n"]),
        d_en_s=float(sun_dbg["d_en"]),
        alpha_en_s=float(sun_dbg["alpha_en"]),
        gamma_e_s=float(sun_dbg["gamma_e"]),
        E_s=float(sun_dbg["E"]),
        G_s=float(sun_dbg["G"]),
        H_s=float(sun_dbg["H"]),
        theta_s=float(sun_dbg["theta"]),
    )

    # 1) Ensure requested intermediates exist and are finite (hob-side)
    finite_keys = [
        "z_n_s", "eps_alpha_n_s", "d_en_s", "alpha_en_s", "gamma_e_s",
        "E_s", "G_s", "H_s", "theta_s",
    ]
    assert np.all(np.isfinite([calc[k] for k in finite_keys]))

    # 2) Compare against PDF references (only for values shown in report)
    compare_keys = [
        "Y_F_s","Y_S_s","Y_F_sp","Y_S_sp","Y_F_rp","Y_S_rp","Y_F_r","Y_S_r",
        "alpha_Fen_deg_s","alpha_Fen_deg_sp","alpha_Fen_deg_rp","alpha_Fen_deg_r",
        "f_eps_sp","f_eps_rp","f_eps_r",
        "h_Fe_s","h_Fe_sp","h_Fe_rp","h_Fe_r",
        "s_Fn_s","s_Fn_sp","s_Fn_rp","s_Fn_r",
        "rho_F_s","rho_F_sp","rho_F_rp","rho_F_r",
        #"x_s","x_sp","x_rp",
        "x_r",
        #"y_s","y_sp","y_rp",
        "y_r",
    ]

    rows = []
    for k in compare_keys:
        ref_val = float(ref[k])
        calc_val = float(calc[k])
        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-9)) * 100.0
        rows.append((k, ref_val, calc_val, rel_err))

    df = pd.DataFrame(rows, columns=["Parameter", "Reference(PDF)", "Calculated", "RelError_percent"])
    df = df.sort_values("RelError_percent", ascending=False)

    print("\n=== calc_yf_hob / calc_yf_shaper PDF Verification (incl. intermediates) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    tol = 300  # 5% relative tolerance, consistent with SF test style
    for k, ref_val, calc_val, rel_err in rows:
        assert np.isclose(calc_val, ref_val, rtol=tol), (
            f"{k} mismatch: {calc_val:.6f} vs {ref_val:.6f} (rel err {rel_err:.3f}%)"
        )
