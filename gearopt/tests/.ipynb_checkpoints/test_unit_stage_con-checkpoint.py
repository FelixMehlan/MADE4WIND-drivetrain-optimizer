import numpy as np
import pandas as pd
import pytest

from gearopt.config.parameters_default import parameters_default
from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.constraints.stage_con import stage_con  # your Python port
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

@pytest.fixture(scope="module")
def setup_data():
    """Prepare validated test setup for stage_con."""
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    x = np.array([
        8, 41, 34, 112, 3, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / (40 * np.pi)
    ])
    i_st = 3.6742
    i_sts = 1.0

    # Geometry and loads
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)


    # Reference constraint values (validated MATLAB baseline)
    ref = {
        "cond_adj": -0.78513,
        "cond_ratio": -0.0043483,
        "cond_mesh": -0.000999,
        "cond_SF_s": -0.93857,
        "cond_SF_p": -0.35107,
        "cond_SF_r": -1.5504,
        "cond_SH_s": -0.28012,
        "cond_SH_p": -0.32302,
        "cond_SH_r": -1.4737,
        "cond_x5": -0.041082,
        "cond_x6": -1.5589,
        "cond_L10h": -0.20909,
        "cond_ShULS_s": -0.001,
        "cond_ShULS_p": -36.011,
        "cond_dss": -0.6984,
        "cond_dps": 0.0,
    }

    return dict(x=x, geo=geo, i_sts=i_sts, ref=ref, i_st=i_st, par=par, data=data)


def test_stage_con_reference(setup_data):
    """Full verification of stage_con() against validated MATLAB results."""

    x = setup_data["x"]
    geo = setup_data["geo"]
    i_sts = setup_data["i_sts"]
    ref = setup_data["ref"]
    i_st = setup_data["i_st"]
    par = setup_data["par"]
    data = setup_data["data"]
    
    # Run DUT (Device Under Test)
    C, Ceq, report = stage_con(x, i_st, i_sts, par, data)

    # Map constraint names
    params = [
        "cond_adj", "cond_ratio", "cond_mesh",
        "cond_SF_s", "cond_SF_p", "cond_SF_r",
        "cond_SH_s", "cond_SH_p", "cond_SH_r",
        "cond_x5", "cond_x6", "cond_L10h",
        "cond_ShULS_s", "cond_ShULS_p", "cond_dss", "cond_dps"
    ]

    # Extract values
    calc_vals = np.array(C).flatten()
    ref_vals = np.array([ref[p] for p in params])
    rel_err = np.abs((calc_vals - ref_vals) / np.maximum(np.abs(ref_vals), 1e-9)) * 100

    # Build readable comparison table
    df = pd.DataFrame({
        "Constraint": params,
        "Reference": ref_vals,
        "Calculated": calc_vals,
        "RelError_percent": rel_err,
    }).sort_values("RelError_percent", ascending=False)

    print("\n=== stage_con Constraint Verification ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    # Tolerance check
    tol = 1e-3  # same as MATLAB test
    for name, refv, calc, err in zip(params, ref_vals, calc_vals, rel_err):
        assert np.isclose(calc, refv, rtol=tol), (
            f"{name} mismatch: {calc:.6f} vs {refv:.6f} (rel err {err:.3e}%)"
        )

    # Sanity checks
    assert Ceq is None or len(Ceq) == 0, "Ceq should be empty"
    assert len(C) == 16, "Expected 16 inequality constraints"
    assert isinstance(report, dict), "Report must be a dictionary"
    assert "gear" in report, "Missing 'gear' in report"
    assert "safety" in report, "Missing 'safety' in report"

    # Feasibility summary
    feasible = np.all(C <= 1e-6)
    print(f"Feasibility: {int(feasible)} (1 = feasible)")
    print(f"Max constraint violation: {np.max(C):.4e}")
