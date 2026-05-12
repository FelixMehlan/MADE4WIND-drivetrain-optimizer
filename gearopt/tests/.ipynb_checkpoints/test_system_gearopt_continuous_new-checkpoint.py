import pytest
import numpy as np
from gearopt.optim.stage_feasibility import stage_feasibility
from gearopt.optim.gearopt_continuous import gearopt_continuous
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

@pytest.fixture(scope="module")
def setup_data():
    paths = load_config("paths")
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters")
    opts = load_config("options")

    # Verified stage ratio
    stage_id = 1
    i_sts = 1
    i_st = 3.6742
    lbI = opts["discrete"]["lbI"]
    ubI = opts["discrete"]["ubI"]
    m_n = 8
    # Feasible integer tuple
    feasInt = stage_feasibility(lbI, ubI, i_st)
    assert feasInt.size > 0, "No feasible integer combinations found"
    xI_base = feasInt[0, :].reshape(-1, 1)
    xI = np.concatenate([xI_base, np.array([[m_n]])], axis=0)
    return {
        "xI": xI,
        "i_st": i_st,
        "i_sts": i_sts,
        "data":data,
        "par":par,
        "opts":opts
    }


def test_continuous_optimization_runs(setup_data):
    """Test that optimization runs and returns valid results."""
    td = setup_data
    Wbest, xCbest, feasible, Cmax, C,_,_,_ = gearopt_continuous(
        td["xI"], td["i_st"], td["i_sts"],td["par"], td["data"], td["opts"]
    )
    print(C)
    assert np.isfinite(Wbest), "Wbest must be finite"
    assert Wbest > 0, "Weight must be positive"
    assert (Cmax < 1e-3) or (not feasible), f"Constraint violation too high (Cmax={Cmax:.3e})"
    
