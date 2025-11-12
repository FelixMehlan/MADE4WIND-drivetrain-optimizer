import numpy as np
from gearopt.optim.gearopt_system import gearopt_system
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

def test_gearopt_system_runs():
    """Test that gearopt_system runs and returns valid results."""
    main_config_file = "optimization_config.yaml"

    best, logbook = gearopt_system(main_config_file)

    # --- Type checks ---
    assert isinstance(best, dict), "best must be a dict"
    assert isinstance(logbook, dict), "logbook must be a dict"

    # --- Sanity checks ---
    assert np.isfinite(best["W"]), f"Total weight not finite: {best['W']}"
    assert best["W"] > 0, f"Total weight must be positive"
    assert "stage" in best and isinstance(best["stage"], list)
    assert "ratios" in best
    assert "feasible" in best
