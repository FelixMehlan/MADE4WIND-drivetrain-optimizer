from scipy.optimize import differential_evolution
from gearopt.optim.gearopt_discrete import gearopt_discrete
from gearopt.constraints.stage_con import stage_con
from gearopt.optim.stage_feasibility import stage_feasibility
from gearopt.optim._cache import _inner_cache
from gearopt.optim.stage_ratio_band_from_bounds import stage_ratio_band_from_bounds
import numpy as np

def gearopt_stage(stage_id, i_st, i_sts, par, data, opts,
                  obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, obj_rest_val_and_grad,par_static, data_static):
    #clear cache
    _inner_cache.clear()
    
    # Extract and coerce optimizer parameters safely
    maxiter = opts["discrete"]["maxiter"]
    popsize = opts["discrete"]["popsize"]
    mutation = opts["discrete"]["mutation"]
    recombination = opts["discrete"]["recombination"]
    polish = opts["discrete"]["polish"]
    disp = opts["discrete"]["disp"]
    lbI = opts["discrete"]["lbI"]
    ubI = opts["discrete"]["ubI"]

    #rmin, rmax = stage_ratio_band_from_bounds(lbI, ubI)
    #if not (rmin <= i_st <= rmax):
    #    return np.inf, np.full(9, np.nan), np.inf, False

    feas_int = stage_feasibility(lbI, ubI, i_st).T
    if feas_int.size == 0:
        return np.inf, np.nan, np.inf, False

    bounds = [(0, len(feas_int.T) - 1)]

    # Discrete optimizer using differential evolution or custom GA
    def fitness(idx):
        return gearopt_discrete(int(round(idx[0])), feas_int, i_st, i_sts, par, data, opts,
                                obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, 
                                obj_rest_val_and_grad,
                                par_static, data_static)

    # --- Define iteration callback (has access to local fitness) ---
    def callback(xk, convergence):
        print(f"Iter {callback.iter:3d} | conv = {convergence:8.3e} | x = {xk}")
        callback.iter += 1
        return False
    callback.iter = 0
  

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
    key_best = (idx_best, round(float(i_st), 6), round(float(i_sts), 6))
    sol = _inner_cache[key_best]

    #W_out = sol["W"] if sol.get("feasible", False) else sol["Wpen"]
    W_out = sol["W"]
    xI = np.atleast_1d(sol.get("xI", np.array([], dtype=float)))
    xC = np.atleast_1d(sol.get("xC", np.array([], dtype=float)))
    
    # If xC is just a scalar nan/inf placeholder, treat as failure
    if xC.ndim != 1 or xC.size == 0:
        return np.inf, np.full(9, np.nan), np.inf, False
    
    x_best = np.concatenate([xI, xC])
    C = sol["C"]
    feas = sol["feasible"]

    return W_out, x_best, C, feas