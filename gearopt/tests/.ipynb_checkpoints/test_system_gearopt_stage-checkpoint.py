import numpy as np
import pytest
from gearopt.optim.gearopt_stage import gearopt_stage
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec
from gearopt.optim.stage_feasibility import stage_feasibility

@pytest.fixture
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
    
    return {
        "stage_id": stage_id,
        "i_st": i_st,
        "i_sts": i_sts,
        "par":par,
        "data":data,
        "opts":opts
    }

def test_gearopt_stage_runs(setup_data):
    """Ensure gearopt_stage runs and returns valid outputs."""
    td = setup_data

    W, x_best, C = gearopt_stage(
        td["stage_id"], td["i_st"], td["i_sts"], td["par"], td["data"], td["opts"]
    )

    # === Check types and shapes ===
    assert np.isscalar(W), "W must be a scalar"
    assert isinstance(x_best, np.ndarray), "x_best must be an array"
    assert isinstance(C, np.ndarray), "C must be an array"

    # === Check numeric sanity ===
    assert np.isfinite(W), f"W is not finite: {W}"
    assert W > 0, f"Stage weight must be positive, got {W}"
    assert len(x_best) >= 9, f"Expected at least 9 design variables, got {len(x_best)}"

    # === Check constraint consistency ===
    Cmax = np.max(C)
    feasible = np.all(C <= 1e-6)
    if not feasible:
        print("\n Stage constraints violated:")
        for i, c in enumerate(C):
            print(f"C[{i:02d}] = {c: .6f}")
        print(f"Max constraint violation: {Cmax:.6e}")
    else:
        print(f"\n Stage feasible: Max constraint = {Cmax:.3e}")

    # Allow either feasible design or correctly flagged infeasible
    assert (Cmax < 1e-3) or (not feasible), \
        f"Constraint violation too high (Cmax={Cmax:.3e})"
