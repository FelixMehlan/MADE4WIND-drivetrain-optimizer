import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from gearopt.optim.gearopt_stage import gearopt_stage
from gearopt.config.load_config import load_config
from gearopt.optim.stage_ratio_band_from_bounds import stage_ratio_band_from_bounds

# === Global cache replacement for MATLAB persistent map ===
_ratio_cache = {}
_cache_stats = {"hit": 0, "miss": 0}


def fitness_ratios(svec, S, smin, smax, par, data, opts, 
                   obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, obj_rest_val_and_grad,
                   par_static, data_static,
                   verbose=False):
    """
    Evaluate gearbox weight for a given N-stage ratio configuration.

    Parameters
    ----------
    svec : array_like
        Log-space free ratios (s1...s_{N-1}).
    S : float
        Log of total gearbox ratio.
    smin, smax : float
        Log-space bounds for stage ratios.
        
    Returns
    -------
    f : float
        Penalized objective (total weight + infeasibility penalty).
    out : dict
        Detailed result including per-stage data, feasibility, and ratios.
    """

    
    global _ratio_cache, _cache_stats

    N_st = par["N_st"]
    svec = np.asarray(svec, dtype=float).ravel()
    
    # --- Derive full ratios ---
    s_last = S - np.sum(svec)
    s_all = np.concatenate([svec, [s_last]])
    ratios = np.exp(s_all)
    i_sts = np.ones(N_st)
    for k in range(1, N_st):
        i_sts[k] = i_sts[k - 1] * ratios[k - 1]

    

    lbI = opts["discrete"]["lbI"]
    ubI = opts["discrete"]["ubI"]

    # --- Integer-feasibility precheck for all stages ---
    for k in range(N_st):
        rmin, rmax = stage_ratio_band_from_bounds(lbI, ubI)
        if not (rmin <= ratios[k] <= rmax):
            # Entire gearbox is infeasible — no need to run any optimizers
            out = {
                "W": np.inf,
                "stage": None,
                "ratios": ratios,
                "Cmax": np.inf,
                "feasible": False,
                "fail_reason": "stage_ratio_out_of_integer_bounds",
                "fail_stage": k + 1,
            }
            f = np.inf
            return f, out

    
    # --- Cache key ---
    key = "_".join([f"{r:.4f}" for r in ratios])

    if key in _ratio_cache:
        entry = _ratio_cache[key]
        _cache_stats["hit"] += 1
        if (_cache_stats["hit"] + _cache_stats["miss"]) % 25 == 0:
            reuse = 100 * _cache_stats["hit"] / (
                _cache_stats["hit"] + _cache_stats["miss"]
            )
            print(f"[Cache] hits={_cache_stats['hit']}, misses={_cache_stats['miss']} ({reuse:.1f}% reuse)")
        return entry["f"], entry["out"]

    # --- Soft penalty for last-stage bound violations ---
    penal = 0.0
    if s_last < smin:
        penal = (smin - s_last) ** 2 * 1e4
    elif s_last > smax:
        penal = (s_last - smax) ** 2 * 1e4

    if verbose:
        print(f"Evaluating ratios: {np.round(np.exp(s_all), 3)}, penalty={penal:.2f}")
        
    # --- Solve each stage independently ---
    stage_data = [None] * N_st
    use_parallel = opts["ratios"]["parallel"]
    
    if use_parallel:
        max_workers = opts.get("n_workers", min(N_st, 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    gearopt_stage,
                    k + 1,
                    ratios[k],
                    i_sts[k],
                    par,
                    data,
                    opts,
                    obj_val_and_grad, con_val_jit, con_jac_jit,
                    obj_hess_jit, obj_rest_val_and_grad,
                    par_static, data_static,
                )
                for k in range(N_st)
            ]
            for k, f_res in enumerate(futures):
                Wk, xk, Ck, feask = f_res.result()
                stage_data[k] = dict(x=xk, W=Wk, C=Ck, feas=feask, ratio=ratios[k])
    else:
        # Serial evaluation (debug-safe)
        for k in range(N_st):
            Wk, xk, Ck, feask = gearopt_stage(
                k + 1,
                ratios[k],
                i_sts[k],
                par,
                data,
                opts,
                obj_val_and_grad, con_val_jit, con_jac_jit,
                obj_hess_jit, obj_rest_val_and_grad,
                par_static, data_static,
            )
            stage_data[k] = dict(x=xk, W=Wk, C=Ck, feas=feask, ratio=ratios[k])


    # --- Aggregate results ---
    Wsum = sum(stage["W"] for stage in stage_data)
    
    # Overall feasibility: all stages feasible
    feas = all(bool(stage.get("feas", False)) for stage in stage_data)
    
    # Still compute a diagnostic Cmax (optional, but useful for penalty magnitude)
    all_C = np.concatenate([
        np.atleast_1d(s["C"]) for s in stage_data if s.get("C") is not None
    ])
    all_C = all_C[np.isfinite(all_C)]
    Cmax = float(np.max(all_C)) if all_C.size else float("inf")


    # --- Penalize infeasibility ---
    #f = Wsum + penal + (not feas) * 1e7 * (1 + max(0.0, Cmax))
    f = Wsum + penal
    # --- Collect outputs ---
    out = {
        "W": Wsum,
        "stage": stage_data,
        "ratios": ratios,
        "Cmax": Cmax,
        "feasible": bool(feas),
    }

    # --- Cache entry ---
    _ratio_cache[key] = {"f": f, "out": out}
    _cache_stats["miss"] += 1

    return f, out
