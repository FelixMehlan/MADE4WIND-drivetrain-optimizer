import numpy as np
import pytest
from gearopt.safety.calc_sh import calc_sh
from gearopt.geometry.calc_geometry import calc_geometry
from gearopt.loads.calc_loads_ldd import calc_loads_ldd
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

@pytest.fixture(scope="module")
def setup_data():
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")
    # === Geometry setup ===
    x = np.array([
        41, 34, 112, 3, 8, 18.75,
        0.3698, 0.9220, np.sin(np.deg2rad(15)) * 750 / 40 / np.pi
    ])

    i_sts = 1


    geo = calc_geometry(x, par)
    loads = calc_loads_ldd(x, geo, i_sts, par, data)

    # === KISSsoft reference values ===
    ref = {
        "Z_H_sp": 2.19, "Z_H_rp": 2.52, "Z_E": 189.81,
        "Z_eps_sp": 0.866, "Z_eps_rp": 0.847, "Z_beta": 1.017,
        "b_eff": 750.0,
        "sigma_H0_sp": 681.71, "sigma_H0_rp": 473.47,
        "sigma_HB_sp": 918.54, "sigma_HD_sp": 918.54,
        "sigma_HB_rp": 593.90, "sigma_HD_rp": 542.16,
        "sigma_HG_s": 1403.59, "sigma_HG_sp": 1443.39,
        "sigma_HG_rp": 1469.95, "sigma_HG_r": 1474.17,
        "Z_L": 1.020, "Z_V": 0.963, "Z_R_sp": 1.038, "Z_R_rp": 1.057,
        "Z_W": 1.000, "Z_X": 1.000,
        "Z_NT_s": 0.918, "Z_NT_p": 0.944, "Z_NT_r": 0.947,
        "SH_s": 1.53, "SH_sp": 1.57, "SH_rp": 2.47,
        "SH_p": 1.57, "SH_r": 2.72,
        "K_A": 1.25, "K_gammaN": 1.00, "K_V": 1.00,
        "K_Hbeta_sp": 1.20, "K_Hbeta_rp": 1.04, "K_Halpha": 1.00,
    }

    return x, i_sts, geo, loads, ref, par


def test_calc_sh_detailed(setup_data):
    x, i_sts, geo, loads, ref, par = setup_data

    # === Compute with debug output ===
    result = calc_sh(x, i_sts, geo, loads, par, return_all=True)

    # --- Compare all fields that exist in reference ---
    rows = []
    for key, ref_val in ref.items():
        if key not in result:
            continue
        calc_val = float(result[key])
        rel_err = abs((calc_val - ref_val) / max(abs(ref_val), 1e-9)) * 100
        rows.append((key, ref_val, calc_val, rel_err))

    # --- Sort by largest error for readability ---
    rows.sort(key=lambda r: r[3], reverse=True)

    # --- Display table ---
    print("\n=== calc_sh Detailed Comparison (vs KISSsoft) ===")
    print(f"{'Parameter':<15}{'Reference':>12}{'Calculated':>14}{'RelErr [%]':>12}")
    print("-" * 55)
    for key, ref_val, calc_val, rel_err in rows:
        print(f"{key:<15}{ref_val:12.3f}{calc_val:14.3f}{rel_err:12.3f}")

    # --- Verify all within tolerance ---
    rel_tol = 0.05  # 5%
    for key, ref_val, calc_val, rel_err in rows:
        assert np.isclose(calc_val, ref_val, rtol=rel_tol), \
            f"{key} mismatch: {calc_val:.3f} vs {ref_val:.3f} ({rel_err:.2f}%)"
