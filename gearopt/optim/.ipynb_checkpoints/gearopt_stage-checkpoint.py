from scipy.optimize import differential_evolution
from gearopt.optim.gearopt_discrete import gearopt_discrete
from gearopt.constraints.stage_con import stage_con
from gearopt.optim.stage_feasibility import stage_feasibility
from gearopt.optim._cache import _inner_cache


def gearopt_stage(stage_id, i_st, i_sts, par, data, opts):

    # Extract and coerce optimizer parameters safely
    maxiter = opts["discrete"]["maxiter"]
    popsize = opts["discrete"]["popsize"]
    mutation = opts["discrete"]["mutation"]
    recombination = opts["discrete"]["recombination"]
    polish = opts["discrete"]["polish"]
    disp = opts["discrete"]["disp"]
    lbI = opts["discrete"]["lbI"]
    ubI = opts["discrete"]["ubI"]

    rmin, rmax = stage_ratio_band_from_bounds(lbI, ubI)
    if not (rmin <= i_st <= rmax):
        return np.inf, np.full(9, np.nan), np.inf

    feas_int = stage_feasibility(lbI, ubI, i_st).T
    if feas_int.size == 0:
        return np.inf, np.nan, np.inf

    # Discrete optimizer using differential evolution or custom GA
    def fitness(idx):
        return gearopt_discrete(int(round(idx[0])), feas_int, i_st, i_sts, par, data, opts)

    # --- Define iteration callback (has access to local fitness) ---
    def callback(xk, convergence):
        fval = fitness(xk)
        print(f"Iter {callback.iter:3d} | f = {fval:10.4f} | conv = {convergence:8.3e} | x = {xk}")
        callback.iter += 1
        return False
    callback.iter = 0

    bounds = [(0, feas_int.shape[1] - 1)]

    

    result = differential_evolution(
        fitness,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        disp=disp,
        callback=callback,
    )

    idx_best = int(round(result.x[0]))
    sol = _inner_cache[idx_best]

    W = sol["W"]
    x_best = np.concatenate([sol["xI"], sol["xC"]])
    C, _, report = stage_con(x_best, i_st, i_sts, par, data)
    return W, x_best, C

import numpy as np

def stage_ratio_band_from_bounds(lbI, ubI):
    """
    Compute approximate feasible stage ratio range from integer bounds.
    
    Parameters
    ----------
    lbI : array-like
        Lower integer bounds [m_n, z_s, z_p, z_r, N_p]
    ubI : array-like
        Upper integer bounds [m_n, z_s, z_p, z_r, N_p]
    
    Returns
    -------
    (rmin, rmax) : tuple of floats
        Minimum and maximum feasible stage ratios.
    """
    zs_min, zs_max = lbI[1], ubI[1]
    zp_min, zp_max = lbI[2], ubI[2]
    zr_max = ubI[3]

    # Effective max planet teeth (limited by ring geometry)
    zp_max_eff = min(zp_max, np.floor((zr_max - zs_min) / 2.0))

    rmin = 2 + 2 * (zp_min / zs_max)
    rmax = 2 + 2 * (zp_max_eff / zs_min)
    return rmin, rmax