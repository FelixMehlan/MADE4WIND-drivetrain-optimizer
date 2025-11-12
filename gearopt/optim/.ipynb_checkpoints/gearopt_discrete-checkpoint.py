from gearopt.optim.gearopt_continuous import gearopt_continuous

from gearopt.optim._cache import _inner_cache

def gearopt_discrete(idx, feasible_int, i_st, i_sts, par, data, opts):
    """
    Discrete-level objective evaluation for gearbox optimization.
    Equivalent to MATLAB's gearopt_discrete.m.
    """

    global _inner_cache

    # --- Check cache ---
    if idx in _inner_cache:
        return _inner_cache[idx]["Wpen"]

    # --- Extract integer design tuple ---
    xI = feasible_int[:, int(idx)]

    # --- Inner continuous optimization ---
    W, xC, feasible, Cmax, C = gearopt_continuous(xI, i_st, i_sts, par, data, opts)

    # --- Penalize infeasibility ---
    if feasible:
        Wpen = W
    else:
        Wpen = W + 1e6 * (1 + max(0, Cmax))

    # --- Store result in cache ---
    sol = dict(xI=xI, xC=xC, W=W, Wpen=Wpen, feasible=feasible, Cmax=Cmax)
    _inner_cache[idx] = sol

    # --- Return penalized objective value ---
    return Wpen
