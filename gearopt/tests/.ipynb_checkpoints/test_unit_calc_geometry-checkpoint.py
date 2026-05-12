import numpy as np
import pandas as pd
from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

def test_calc_geometry_runs():
    """Ensure calc_geometry runs and returns a valid dictionary."""
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    x = np.array([
        41, 34, 112, 3, 8, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    geo = calc_geometry(x,par)

    # --- Updated for dict output ---
    assert isinstance(geo, dict), "calc_geometry should return a dict"
    assert len(geo) > 0, "geo dictionary should not be empty"
    assert all(np.isfinite(v) for v in geo.values()), "All geometry values must be finite"


def test_calc_geometry_reference_comparison():
    """Compare calc_geometry output to KISSsoft reference values."""
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    x = np.array([
        41, 34, 112, 3, 8, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    geo = calc_geometry(x,par)

    ref = {
        "a": 1600.000,
        "m_n": 40.0000,
        "m_t": 41.4110,
        "alpha_t": np.deg2rad(20.647),
        "alpha_n": np.deg2rad(20.0000),
        "beta": np.deg2rad(15.0000),
        "beta_b": np.deg2rad(14.076),
        "z_s": 41,
        "z_p": 34,
        "z_r": 112,
        "i_st": 3.732,
        "alpha_wt_sp": np.deg2rad(24.743),
        "alpha_wt_rp": np.deg2rad(19.161),
        "d_p_s": 1697.853,
        "d_p_p": 1407.976,
        "d_p_r": 4638.037,
        "d_b_s": 1588.802,
        "d_b_p": 1317.543,
        "d_b_r": 4340.142,
        "d_a_s": 1798.263,
        "d_a_p": 1552.561,
        "d_a_r": 4602.755,
        "d_f_s": 1627.439,
        "d_f_p": 1381.737,
        "d_f_r": 4782.755,
        "b": 750,
        "x_r": -0.5590,
        "eps_alpha_t_sp": 1.333,
        "eps_beta": 1.545,
        "eps_total": 2.877,
        "eps_1_sp": 0.453,
        "eps_2_sp": 0.880,
        "eps_1_rp": 1.394,
        "xE_i_s": 0.3544,
        "xE_i_p": 0.9103,
        "xE_i_r": -0.5858,
        "Asn_i_s_mm": -0.450,
        "Asn_i_p_mm": -0.340,
        "Asn_i_r_mm": -0.780,
    }

    # --- Extract only available keys ---
    common_keys = [k for k in ref.keys() if k in geo]
    ref_vals = np.array([ref[k] for k in common_keys])
    calc_vals = np.array([geo[k] for k in common_keys])

    # --- Compute relative errors ---
    rel_err = np.abs(calc_vals - ref_vals) / np.maximum(np.abs(ref_vals), 1e-12) * 100

    df = pd.DataFrame({
        "Parameter": common_keys,
        "Reference": ref_vals,
        "Calculated": calc_vals,
        "RelError_percent": rel_err,
    }).sort_values("RelError_percent", ascending=False)

    print("\n=== Geometry Comparison (KISSsoft vs calc_geometry) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    tol = 1e-2  # 1% tolerance
    for name, val, ref_val, err in zip(common_keys, calc_vals, ref_vals, rel_err):
        assert np.isfinite(val), f"{name} produced NaN/Inf"
        assert np.isclose(val, ref_val, rtol=tol), f"{name} mismatch: {val} vs {ref_val} (rel err {err:.3g}%)"
