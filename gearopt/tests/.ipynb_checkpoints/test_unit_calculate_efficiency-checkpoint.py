import numpy as np
import pandas as pd

from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.efficiency.calc_efficiency import calc_efficiency
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

def test_calc_efficiency_reference():
    """
    Compare calculated gear meshing efficiency and losses against
    KISSsoft reference values. Uses calc_efficiency(return_all=True)
    to extract detailed intermediate results.
    """
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    # === Design vector (same as other tests) ===
    x = np.array([
        8.0, 41.0, 34.0, 112.0, 3.0, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / (40 * np.pi)
    ])

    # --- Parameters and load spectrum ---
    i_sts = 1.0

    # --- Geometry and loads ---
    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # --- KISSsoft reference values (Power Loss Report 14.3) ---
    ref = {
        "mu_m_sp": 0.034,
        "mu_m_rp": 0.026,
        "H_v_sp": 0.113,
        "H_v_rp": 0.103,
        "P_mesh_sp": 10_980.392e3,
        "P_mesh_rp": 10_980.392e3,
        "P_V_sp": 13.874e3 * 3,
        "P_V_rp": 9.769e3 * 3,
        "P_V": 70.929e3,
        "eff": 0.99527,  # 99.527%
    }

    # --- Compute efficiency with full output ---
    result = calc_efficiency(x, i_sts, geo, loads, par, return_all=True)

    # --- Build comparison table for shared keys ---
    rows = []
    for key, ref_val in ref.items():
        if key not in result:
            print(f" Warning: '{key}' not found in result dictionary.")
            continue
        calc_val = float(result[key])
        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-9)) * 100
        rows.append((key, ref_val, calc_val, rel_err))

    # --- Convert to DataFrame for readability ---
    df = pd.DataFrame(rows, columns=["Parameter", "Reference", "Calculated", "RelError_percent"])
    df = df.sort_values("RelError_percent", ascending=False)

    print("\n=== Efficiency Comparison (calc_efficiency vs KISSsoft) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # --- Verification ---
    rel_tol = 0.10  # 10% tolerance
    for key, ref_val, calc_val, rel_err in rows:
        assert np.isclose(calc_val, ref_val, rtol=rel_tol), (
            f"{key} mismatch: {calc_val:.6g} vs {ref_val:.6g} "
            f"(rel err {rel_err:.2f}%)"
        )

