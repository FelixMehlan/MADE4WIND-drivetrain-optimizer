from gearopt.optim.gearopt_continuous import gearopt_continuous
from gearopt.optim._cache import _inner_cache
import numpy as np


def gearopt_discrete(idx, feasible_int, i_st, i_sts, par, data, opts,
                    obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, obj_rest_val_and_grad,
                    par_static, data_static,verbose=False):
    """
    Discrete-level objective evaluation for gearbox optimization.

    Now loops over module numbers m_n from ubI→lbI in integer steps.
    Terminates early when no feasible continuous solution is found.
    """

    global _inner_cache

    # --------------------------
    # 1. Check cache
    # --------------------------
    key = (int(idx), round(float(i_st), 6), round(float(i_sts), 6))

    if key in _inner_cache:
        return _inner_cache[key]["Wpen"]

    # --------------------------
    # 2. Extract full integer bounds
    # --------------------------
    lbI = np.atleast_1d(opts["discrete"]["lbI"])
    ubI = np.atleast_1d(opts["discrete"]["ubI"])

    # --------------------------
    # 3. Extract integer tuple WITHOUT m_n
    # --------------------------
    xI_base = feasible_int[:, int(idx)].astype(float)
    if verbose:
        print(f"\n[DISCRETE]: Trying integer tuple xI = {np.array2string(xI_base, precision=3)}")
    # --------------------------
    # 4. Construct descending module list:
    #    m_n_list = [ubI[4], ubI[4]-1, ..., lbI[4]]
    # --------------------------
    m_n_list = range(int(ubI[4]), int(lbI[4]) - 1, -1)

    results_mn = []

    # --------------------------
    # 5. Loop over modules, descending
    # --------------------------
    for m_n in m_n_list:
        
        # full integer vector for this test
        xI_full = np.concatenate([xI_base, [m_n]])
        
        # run continuous optimization
        W, xC, feasible, Cmax, C, _ , _ , lenFeas = gearopt_continuous(
            xI_full, i_st, i_sts, par, data, opts,
            obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, obj_rest_val_and_grad,
            par_static, data_static
        )

        if feasible:
            Wpen = W
        else:
            Wpen = W + 1e7 * (1 + max(0, Cmax))

        results_mn.append({
            "m_n":      m_n,
            "xI":  xI_full,
            "xC":       xC,
            "W":        W,
            "Wpen":     Wpen,
            "feasible": feasible,
            "Cmax":     Cmax,
            "C":        C
        })

        # --------------------------
        # EARLY TERMINATION
        # Stop descending once continuous optimizer fails
        # --------------------------
        if not feasible:
            break

    # --------------------------
    # 6. Select best result
    # --------------------------
    feasibles = [r for r in results_mn if r["feasible"]]

    if len(feasibles) > 0:
        best = min(feasibles, key=lambda r: r["W"])
    else:
        best = min(results_mn, key=lambda r: r["Wpen"])

    # --------------------------
    # 7. Cache and return
    # --------------------------
    _inner_cache[key] = best
    return best["W"]
