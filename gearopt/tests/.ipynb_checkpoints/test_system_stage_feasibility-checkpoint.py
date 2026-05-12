import numpy as np
import pytest
from gearopt.optim.stage_feasibility import stage_feasibility
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

@pytest.fixture(scope="module")
def setup_data():
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")

    # Verified stage ratio
    i_sts = 1
    i_st = 3.6742
    lbI = opts["discrete"]["lbI"]
    ubI = opts["discrete"]["ubI"]
    
    candidates = stage_feasibility(lbI, ubI, i_st)
    return {"lbI": lbI, "ubI": ubI, "i_st": i_st, "candidates": candidates}

def test_candidate_structure(setup_data):
    candidates = setup_data["candidates"]
    assert candidates.size > 0, "No feasible candidates found"
    assert candidates.shape[1] == 4, "Unexpected number of columns"

def test_ratio_consistency(setup_data):
    candidates = setup_data["candidates"]
    i_st = setup_data["i_st"]

    i_calc = 1 + candidates[:, 2] / candidates[:, 0]
    rel_err = np.abs((i_calc - i_st) / i_st)
    assert np.all(rel_err <= 0.1), f"Some ratios exceed tolerance (max {rel_err.max():.3f})"

def test_adjacency_constraint(setup_data):
    candidates = setup_data["candidates"]
    h_a = 1.0

    for row in candidates:
        z_s, z_p, N_p = row[0], row[1], row[3]
        a = 0.5 * (z_s + z_p)
        d_ap = z_p + 2 * h_a
        cond_adj = (d_ap - 2 * a * np.sin(np.pi / N_p)) / d_ap
        assert cond_adj < 0, f"Adjacency violated (cond_adj = {cond_adj:.4f})"
