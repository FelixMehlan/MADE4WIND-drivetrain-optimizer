import numpy as np
import pandas as pd

from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

def test_calc_loads_ldd_basic():
    """
    Smoke test: ensures function runs, returns valid outputs, and magnitudes make sense.
    """
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    
    # Simplified dummy input
    x = np.array([
        41, 34, 112, 3, 8, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    geo = calc_geometry(x, par)
    i_sts = 3.0

    # Call with default (dict-based) return
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # If loads is a dict, flatten to array for sanity checks
    if isinstance(loads, dict):
        vals = np.array(list(loads.values()), dtype=float)
    else:
        vals = loads

    # Basic integrity checks
    assert vals.size >= 16, "Expected at least 16 load components"
    assert np.all(np.isfinite(vals)), "Output contains NaN or Inf"
    assert np.all(vals >= 0), "All loads should be nonnegative magnitudes"


def test_calc_loads_ldd_reference_comparison():
    """
    Full verification of calc_loads_ldd() against validated MATLAB reference.
    Uses return_all=True for detailed introspection.
    """
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    # --- Design vector (same as geometry tests) ---
    x = np.array([
        41, 34, 112, 3, 8, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    # --- Parameters and setup ---
    i_sts = 1.0

    # --- Geometry ---
    geo = calc_geometry(x,par)

    # --- Reference outputs (validated MATLAB results) ---
    ref = {
        "T_s_ShULS": 1.0155e7, "T_p_ShULS": 2.807e6,
        "T_s_ShFLS": 5.0773e6, "T_p_ShFLS": 1.4035e6,
        "F_t_s_SH": 1.9936e6, "F_t_s_SF": 1.9936e6, "F_t_s_SB": 1.9936e6,
        "F_t_r_SH": 1.9936e6, "F_t_r_SF": 1.9936e6, "F_t_r_SB": 1.9936e6,
        "F_rad": 4.2799e6, "F_ax": 1.1642e-10,
        "rpm_r": 22.68, "rpm_s": 61.955, "rpm_p": 24.904, "rpm_b": 32.464
    }

    # --- Run function under test ---
    result = calc_loads_ldd(x, geo, i_sts, par, data, return_all=True)

    assert isinstance(result, dict), "calc_loads_ldd should return a dict when return_all=True"

    # --- Match shared keys between ref and result ---
    rows = []
    for key, ref_val in ref.items():
        if key not in result:
            print(f" Warning: '{key}' missing in result dict")
            continue
        calc_val = float(result[key])
        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-9)) * 100
        rows.append((key, ref_val, calc_val, rel_err))

    df = pd.DataFrame(rows, columns=["Parameter", "Reference", "Calculated", "RelError_percent"])
    df = df.sort_values("RelError_percent", ascending=False)

    print("\n=== calc_loads_ldd Verification (vs MATLAB Reference) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.6f}"))

    # --- Validation ---
    rel_tol = 0.01   # 1% tolerance
    abs_tol_Fax = 0.1
    
    for key, ref_val, calc_val, rel_err in rows:
    
        # Special rule for axial force
        if key == "F_ax":
            assert np.isclose(calc_val, ref_val, atol=abs_tol_Fax), (
                f"{key} mismatch: {calc_val:.6e} vs {ref_val:.6e} "
                f"(abs err {abs(calc_val-ref_val):.6e}, tol={abs_tol_Fax})"
            )
            continue
    
        # General rule: relative tolerance
        assert np.isclose(calc_val, ref_val, rtol=rel_tol), (
            f"{key} mismatch: {calc_val:.6e} vs {ref_val:.6e} "
            f"(rel err {rel_err:.3f}%)"
        )

