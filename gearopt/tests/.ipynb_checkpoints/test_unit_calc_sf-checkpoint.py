import numpy as np
import pandas as pd
import pytest

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.safety.calc_sf import calc_sf
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

@pytest.fixture(scope="module")
def setup_data():
    """Prepare shared test data for SF comparison."""
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

    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # Reference values (from validated KISSsoft or MATLAB)
    ref = dict(
        Y_F_s=1.15, Y_S_s=2.14,
        Y_F_sp=0.95, Y_S_sp=2.43,
        Y_F_rp=0.84, Y_S_rp=2.54,
        Y_F_r=0.94, Y_S_r=2.40,
        sigma_F_s=236.28, sigma_F_sp=220.17,
        sigma_F_rp=179.98, sigma_F_r=190.69,
        sigma_FG_s=581.95, sigma_FG_sp=416.77,
        sigma_FG_rp=416.77, sigma_FG_r=598.50,
        SF_s=2.46, SF_sp=1.89, SF_rp=2.32, SF_p=1.89, SF_r=3.14
    )

    return dict(x=x, geo=geo, loads=loads, i_sts=i_sts, ref=ref, par=par)


# === TEST 1: Smoke test ===
def test_calc_sf_basic(setup_data):
    """Verify calc_sf runs and produces finite values in expected range."""
    x, geo, loads, i_sts, par = (
        setup_data["x"], 
        setup_data["geo"], setup_data["loads"], setup_data["i_sts"], setup_data["par"]
    )

    SF_s, SF_p, SF_r = calc_sf(x, i_sts, geo, loads, par)

    assert np.all(np.isfinite([SF_s, SF_p, SF_r]))
    assert np.all(np.array([SF_s, SF_p, SF_r]) > 0.1)
    assert np.all(np.array([SF_s, SF_p, SF_r]) < 10.0)


# === TEST 2: Detailed comparison against reference ===
def test_calc_sf_reference_comparison(setup_data):
    """Compare calc_sf outputs with validated MATLAB/KISSsoft references."""
    x, geo, loads, i_sts, ref, par = (
        setup_data["x"], setup_data["geo"],
        setup_data["loads"], setup_data["i_sts"], setup_data["ref"], setup_data["par"]
    )

    # Use full detail mode
    result = calc_sf(x, i_sts, geo, loads, par, return_all=True)

    # Build aligned comparison table
    rows = []
    for key, ref_val in ref.items():
        if key not in result:
            print(f"Warning: '{key}' missing in result dict")
            continue
        calc_val = float(result[key])
        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-9)) * 100
        rows.append((key, ref_val, calc_val, rel_err))

    df = pd.DataFrame(rows, columns=["Parameter", "Reference", "Calculated", "RelError_percent"])
    df = df.sort_values("RelError_percent", ascending=False)

    print("\n=== calc_sf Detailed Verification ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # Numeric verification
    tol = 0.05  # 5% tolerance
    for key, ref_val, calc_val, rel_err in rows:
        assert np.isclose(calc_val, ref_val, rtol=tol), (
            f"{key} mismatch: {calc_val:.4f} vs {ref_val:.4f} (rel err {rel_err:.3f}%)"
        )
