import numpy as np
import pandas as pd

from gearopt.constraints.stage_weight import stage_weight
from gearopt.config.parameters_default import parameters_default
from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.util.fit_bearing_catalog import fit_bearing_catalog
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

def test_stage_weight_reference():
    """Full verification of stage_weight() against validated MATLAB reference results."""
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    # --- Design vector ---
    x = np.array([
        8, 41, 34, 112, 3, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    # --- Parameters, geometry, loads ---
    i_sts = 1
    i_st = 3.6742
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # --- Run DUT ---
    W_st, reportW_k = stage_weight(x, i_st, i_sts, par, data)

    # --- Extract calculated values ---
    calc_vals = np.array([
        W_st,
        reportW_k["gear"]["W_ss"],
        reportW_k["gear"]["W_rg"],
        reportW_k["gear"]["W_pg"],
        reportW_k["gear"]["W_pb"],
        reportW_k["gear"]["W_plc"],
        reportW_k["gear"]["W_ps"],
        reportW_k["gear"]["W_sg"],
    ])

    # --- Reference values (validated MATLAB “Calculated” column) ---
    ref_vals = np.array([
        1.1284e+05,   # W_st
        1972.9,     # W_ss
        20894,      # W_rg
        2086.9,     # W_pg
        792.53,      # W_pb
        52677,      # W_plc
        4370.5,     # W_ps
        13164,      # W_sg
    ])

    params = ["W_st", "W_ss", "W_rg", "W_pg", "W_pb", "W_plc", "W_ps", "W_sg"]

    # --- Compute relative error (%) ---
    rel_err = np.abs((calc_vals - ref_vals) / np.maximum(np.abs(ref_vals), 1e-9)) * 100

    # --- Display comparison table ---
    df = pd.DataFrame({
        "Parameter": params,
        "Reference": ref_vals,
        "Calculated": calc_vals,
        "RelError_percent": rel_err,
    }).sort_values("RelError_percent", ascending=False)

    print("\n=== stage_weight Verification ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # --- Verification threshold ---
    tol = 0.05  # 5% relative tolerance
    for name, refv, calc, err in zip(params, ref_vals, calc_vals, rel_err):
        assert np.isclose(calc, refv, rtol=tol), (
            f"{name} mismatch: {calc:.4f} vs {refv:.4f} (rel err {err:.3f}%)"
        )

    # --- Sanity checks ---
    assert W_st > 0, "Stage weight must be positive."
    assert 0.9 <= reportW_k["gear"]["eta"] <= 1.0, "Stage efficiency must be plausible."
