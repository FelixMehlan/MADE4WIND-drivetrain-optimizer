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
    Case study #2 fixture:
    build x, geo, par and store PDF reference values.
    """
    paths = load_config("paths")
    _ = load_tspec(paths["tspec_path"])  # kept for parity; not required for YF tests
    par = load_config("parameters")

    # ---- New case study vector (from the attached report) ----
    # z_s=33, z_p=27, z_r=90 (internal gear will be passed as -z_r to shaper),
    # n_planets=3, dm_n=8 -> m_n=40, db=b/m_n=800/40=20,
    # x_s=0.2004, x_p=0.2624,
    # eps_beta = b*sin(beta)/(pi*m_n) with beta=15deg and b=800mm
    m_n = 40.0
    b = 800.0
    beta_deg = 15.0
    eps_beta = np.sin(np.deg2rad(beta_deg)) * b / (np.pi * m_n)

    x = np.array([
        33.0,          # z_s
        27.0,          # z_p
        90.0,          # z_r  (passed as -x[2] to internal gear shaper)
        3.0,           # number of planets
        8.0,           # dm_n selector (project convention)
        20.0,          # db = b/m_n = 800/40
        0.2004,        # x_s
        0.2624,        # x_p
        eps_beta       # overlap ratio epsilon_beta
    ], dtype=float)

    geo = calc_geometry(x, par)
    print("geo")
    print(geo)
    # ---- PDF reference values (report section 7 table) ----
    ref = dict(
        # Factors
        Y_F_s=1.09, Y_S_s=2.06,
        Y_F_sp=1.06, Y_S_sp=2.09,
        Y_F_rp=0.85, Y_S_rp=2.24,
        Y_F_r=0.93, Y_S_r=2.22,

        # Intermediate values shown in PDF
        alpha_Fen_deg_s=21.47,
        alpha_Fen_deg_sp=21.90,
        alpha_Fen_deg_rp=20.13,
        alpha_Fen_deg_r=16.48,

        # Load distribution influence factor fε
        f_eps_sp=0.801,   # pair 1
        f_eps_rp=0.766,   # pair 2
        f_eps_r=0.766,    # internal gear uses pair 2

        # Root geometry
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

        # Tangent contact point, virtual spur gear (internal gear line in report)
        x_r=53.482,
        y_r=1882.966,
    )

    return dict(x=x, geo=geo, par=par, ref=ref)


# === TEST 1: Smoke test ===
def test_calc_yf_basic(setup_yf_data):
    """Verify calc_yf_hob / calc_yf_shaper run and return finite values in expected range."""
    geo, par, x = setup_yf_data["geo"], setup_yf_data["par"], setup_yf_data["x"]

    alpha_n = np.deg2rad(par["alpha_n_deg"])
    m_n = 40.0

    # b = db*m_n; eps_beta = x[8]
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
    assert np.all(np.isfinite(vals))
    assert np.all(vals > 0.0)
    assert np.all(vals < 10.0)


# === TEST 2: PDF comparisons (YF/YS + intermediate values) ===
# ... keep imports and fixture as-is ...

BETA_DEG_CASE2 = 15.0  # <-- set this to the helix angle shown in the attached report


def test_calc_yf_pdf_reference_comparison(setup_yf_data):
    geo, par, x, ref = (
        setup_yf_data["geo"],
        setup_yf_data["par"],
        setup_yf_data["x"],
        setup_yf_data["ref"],
    )

    alpha_n = jnp.deg2rad(par["alpha_n_deg"])
    m_n = 40.0

    # IMPORTANT: For PDF reference comparisons, use helix angle beta from the report,
    # not reconstructed from eps_beta and facewidth. This avoids inconsistencies.
    beta = float(np.deg2rad(BETA_DEG_CASE2))

    # --- HOB: use debug_mode=True to get intermediate values ---
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

    # --- SHAPER (internal gear): debug_mode=True to get intermediates ---
    r_dbg = calc_yf_shaper(
        -geo["d_p_r"], -geo["d_a_r"], -x[2], geo["x_r"],
        geo["eps_alpha_t_rp"], m_n, alpha_n,
        par["h_fPf"], par["rho_fPf"], beta, x[8],
        debug_mode=True,
    )

    # Compute f_eps for hob cases (PDF calls it fε)
    f_eps_sp = float(smooth_feps(float(x[8]), float(sp_dbg["eps_alpha_n"])))
    f_eps_rp = float(smooth_feps(float(x[8]), float(rp_dbg["eps_alpha_n"])))
    f_eps_r  = float(r_dbg["f_eps"])

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
    for k in compare_keys:
        ref_val = float(ref[k])
        calc_val = float(calc[k])

        # internal gear y-coordinate sign convention differs frequently between implementations
        if k == "y_r":
            calc_val = abs(calc_val)

        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-12)) * 100.0
        rows.append((k, ref_val, calc_val, rel_err))

    df = pd.DataFrame(rows, columns=["Parameter", "Reference(PDF)", "Calculated", "RelError_percent"])
    df = df.sort_values("RelError_percent", ascending=False)

    print("\n=== calc_yf_hob / calc_yf_shaper PDF Verification (case2) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    tol = 0.05
    for k, ref_val, calc_val, rel_err in rows:
        assert np.isclose(calc_val, ref_val, rtol=tol), (
            f"{k} mismatch: {calc_val:.6f} vs {ref_val:.6f} (rel err {rel_err:.3f}%)"
        )
