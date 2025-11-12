from scipy.optimize import minimize
import numpy as np
from gearopt.constraints.stage_weight import stage_weight
from gearopt.constraints.stage_con import stage_con
from gearopt.util.scale_variables import scale_variables
from gearopt.util.unscale_variables import unscale_variables
from gearopt.config.load_config import load_config

def gearopt_continuous(xI, i_st, i_sts, par, data, opts):
    lbC = opts["continuous"]["lbC"]
    ubC = opts["continuous"]["ubC"]
    optsC = opts["continuous"]
    solver_keys = {"disp", "maxiter", "ftol"}
    solver_opts = {k: optsC[k] for k in solver_keys if k in optsC}

    # Ensure arrays
    
    xI, lbC, ubC = map(np.atleast_1d, (xI, lbC, ubC))
    x0C = (lbC + ubC) / 2

    x0_scaled, lb_s, ub_s, scale = scale_variables(x0C, lbC, ubC)

    def fun_scaled(x_scaled):
        x_phys = np.concatenate([
            np.ravel(xI),                        # flatten 5x1 → (5,)
            np.ravel(unscale_variables(x_scaled, lbC, ubC))  # flatten (4,)
        ])
        W_st, _ = stage_weight(x_phys, i_st, i_sts,  par, data)
        return W_st

    def nonl_scaled(x_scaled):
        x_phys = np.concatenate([
            np.ravel(xI),                        # flatten 5x1 → (5,)
            np.ravel(unscale_variables(x_scaled, lbC, ubC))  # flatten (4,)
        ])
        C, Ceq, _ = stage_con(x_phys, i_st, i_sts,  par, data)
        return C, Ceq

    res = minimize(
        fun_scaled, x0_scaled, bounds=list(zip(lb_s, ub_s)),
        constraints={'type': 'ineq', 'fun': lambda x: -nonl_scaled(x)[0]},
        options=solver_opts
    )

    xCbest = unscale_variables(res.x, lbC, ubC)
    C, _ = nonl_scaled(res.x)
    feasible = np.all(C <= 1e-6)
    Cmax = np.max(C)
    return res.fun, xCbest, feasible, Cmax, C
