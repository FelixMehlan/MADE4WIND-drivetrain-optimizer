import numpy as np
from concurrent.futures import ThreadPoolExecutor

from gearopt.optim.gearopt_stage import gearopt_stage
from gearopt.optim.gearopt_ratios import gearopt_ratios  # placeholder for future GA version
from gearopt.config.load_config import load_config
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec

def gearopt_system(main_config_file):
    """
    Optimize or evaluate gearbox design across all stages.

    Parameters
    ----------
    i_gb : float
        Target total gearbox ratio.
    use_variable_ratios : bool, optional
        If True, performs outer GA optimization for stage ratios.
        Otherwise assumes equal ratios for all stages.

    Returns
    -------
    best : dict
        Dictionary with best design summary:
            - "W" : total weight
            - "stage" : list of per-stage solutions
            - "ratios" : per-stage ratio vector
            - "Cmax" : max constraint violation
            - "feasible" : overall feasibility flag
    logbook : dict
        Metadata or optimization history (for GA).
    """
    paths = load_config("paths",main_config_file)
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters",main_config_file)
    opts = load_config("options",main_config_file)
    syst = load_config("system",main_config_file)
    
    N_st = par["N_st"]
    i_gb = par["i_gb"]
    use_variable_ratios = syst["use_variable_ratios"]

    # ============================================================
    # CASE 1: Variable ratio optimization (outer GA)
    # ============================================================
    if use_variable_ratios:
        if N_st == 1:
            # Degenerate case: only one stage, no ratio optimization needed
            W1, x1, C1 = gearopt_stage(
                stage_id=1, i_st=i_gb, i_sts=1, par=par, data=data, opts=opts
            )
            best = {
                "W": W1,
                "stage": [dict(x=x1, W=W1, C=C1, ratio=i_gb)],
                "ratios": np.array([i_gb]),
            }
            logbook = {"s_best": None, "W_best": W1, "best": best}
            return best, logbook
        else:
            # Run GA-based ratio optimization
            best, logbook = gearopt_ratios(i_gb , par=par, data=data, opts=opts)
            return best, logbook

    # ============================================================
    # CASE 2: Fixed equal per-stage ratios
    # ============================================================
    else:
        # --- Equal per-stage ratios ---
        r_tgt = i_gb ** (1.0 / N_st)
        ratios = np.full(N_st, r_tgt)
        i_sts = np.ones(N_st)
        for k in range(1, N_st):
            i_sts[k] = i_sts[k - 1] * ratios[k - 1]

        # --- Run per-stage optimizations in parallel ---
        stage_results = [None] * N_st
        with ThreadPoolExecutor(max_workers=min(N_st, 4)) as executor:
            futures = []
            for k in range(N_st):
                futures.append(
                    executor.submit(
                        gearopt_stage,
                        k + 1,
                        ratios[k],
                        i_sts[k],
                        par,
                        data,
                        opts
                    )
                )
            for k, f in enumerate(futures):
                stage_results[k] = f.result()

        # --- Aggregate results ---
        total_weight = 0.0
        feasible = True
        Cmax = -np.inf
        stage_info = []

        for k in range(N_st):
            Wk, xk, Ck = stage_results[k]
            total_weight += Wk
            if Ck is not None and np.size(Ck) > 0:
                Cmax = max(Cmax, np.max(Ck))
                feasible = feasible and np.all(Ck <= 0)
            stage_info.append(
                {"x": xk, "W": Wk, "C": Ck, "ratio": ratios[k]}
            )

        # --- Store outputs ---
        best = {
            "W": total_weight,
            "stage": stage_info,
            "ratios": ratios,
            "Cmax": Cmax,
            "feasible": feasible,
        }

        logbook = {
            "ratios": ratios,
            "W_best": total_weight,
            "best": best,
        }

        return best, logbook
