import numpy as np
from scipy.optimize import differential_evolution
from gearopt.config.load_config import load_config
from gearopt.optim.fitness_ratios import fitness_ratios


def gearopt_ratios(i_gb, par, data, opts, verbose=True):
    """
    Multi-stage GA optimization over stage ratios (variable-ratio mode).

    Parameters
    ----------
    i_gb : float
        Total gearbox ratio.
    par : ndarray
        Parameter vector (contains number of stages in par[3]).
    lbI, ubI : ndarray
        Integer variable bounds.
    lbC, ubC : ndarray
        Continuous variable bounds.

    Returns
    -------
    best : dict
        Best design dictionary containing total weight, per-stage data, ratios, etc.
    logbook : dict
        Optimization history with GA variable states.
    """

    # --- Load configuration ---
   
    N_st = par["N_st"]
    if N_st < 2:
        raise ValueError("gearopt_ratios requires at least 2 stages.")

    # === Define ratio search bounds in log space ===
    r_tgt = i_gb ** (1.0 / N_st)
    s_min = np.log(0.75 * r_tgt)
    s_max = np.log(1.5 * r_tgt)
    S = np.log(i_gb)

    nvars = N_st - 1  # free ratios (s1...s_{N-1}), last derived to meet total S
    bounds = [(s_min, s_max)] * nvars

    # === Fitness function wrapper ===
    def fitness(s_free):
        return fitness_ratios(
            np.asarray(s_free).ravel(),
            S,
            s_min,
            s_max,
            par,
            data,
            opts
        )[0]  # [0] returns weight (objective)

     # --- Parse optimizer options from config ---
    maxiter = opts["ratios"]["maxiter"]
    popsize = opts["ratios"]["popsize"]
    mutation = opts["ratios"]["mutation"]
    recombination = opts["ratios"]["recombination"]
    tol = opts["ratios"]["tol"]
    disp = opts["ratios"]["disp"]
    polish = opts["ratios"]["polish"]

    # --- Run optimizer ---
    result = differential_evolution(
        fitness,
        bounds,
        strategy="best1bin",
        popsize=popsize,
        maxiter=maxiter,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        tol=tol,
        disp=disp,
    )

    s_best = result.x

    # === Get full best design info from fitness_ratios ===
    _, best = fitness_ratios(
        s_best, S, s_min, s_max, par, data, opts
    )

    # === Logbook (metadata) ===
    logbook = {
        "s_best": s_best,
        "W_best": best["W"],
        "best": best,
        "nfev": result.nfev,
        "nit": result.nit,
        "success": result.success,
    }

    return best, logbook
