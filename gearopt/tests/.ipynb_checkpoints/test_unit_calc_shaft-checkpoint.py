import numpy as np
import pandas as pd
import pytest

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.safety.calc_shaft import calc_shaft
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

@pytest.fixture(scope="module")
def setup_data():
    """Prepare common test data for shaft verification."""
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    # --- Design vector (same as other tests) ---
    x = np.array([
        8.0, 41.0, 34.0, 112.0, 3.0, 18.75,
        0.3698, 0.9220,
        np.sin(np.deg2rad(15)) * 750 / (40 * np.pi)
    ])

    # --- Parameters and setup ---
    i_sts = 1.0

    # --- Geometry and loads ---
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # --- Reference outputs (validated MATLAB version) ---
    ref = dict(
        S_ShULS_s=1.10,
        S_ShULS_p=37.11,
        d_ss=461.88,
        d_ps=972.20,
    )


    return dict(x=x, geo=geo, loads=loads, i_sts=i_sts, ref=ref, par=par)


def test_calc_shaft_reference(setup_data):
    """Full verification of calc_shaft() against validated MATLAB reference."""

    x = setup_data["x"]
    geo = setup_data["geo"]
    loads = setup_data["loads"]
    i_sts = setup_data["i_sts"]
    ref = setup_data["ref"]
    par = setup_data["par"]

    # --- Run Device Under Test (DUT) ---
    S_ShULS_s, S_ShULS_p, report = calc_shaft(x, i_sts, geo, loads, par)

    # --- Extract computed values ---
    calc_vals = np.array([
        S_ShULS_s,
        S_ShULS_p,
        report["d_ss"],
        report["d_ps"],
    ])

    ref_vals = np.array([
        ref["S_ShULS_s"],
        ref["S_ShULS_p"],
        ref["d_ss"],
        ref["d_ps"],
    ])

    params = ["S_ShULS_s", "S_ShULS_p", "d_ss", "d_ps"]

    # --- Compute relative error (%) ---
    rel_err = np.abs((calc_vals - ref_vals) / np.maximum(np.abs(ref_vals), 1e-9)) * 100

    # --- Build comparison table ---
    df = pd.DataFrame({
        "Parameter": params,
        "Reference": ref_vals,
        "Calculated": calc_vals,
        "RelError_percent": rel_err,
    }).sort_values("RelError_percent", ascending=False)

    print("\n=== calc_shaft Verification ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # --- Verification ---
    tol = 0.05  # 5% tolerance
    for name, refv, calc, err in zip(params, ref_vals, calc_vals, rel_err):
        assert np.isclose(calc, refv, rtol=tol), (
            f"{name} mismatch: {calc:.4f} vs {refv:.4f} (rel err {err:.3f}%)"
        )
