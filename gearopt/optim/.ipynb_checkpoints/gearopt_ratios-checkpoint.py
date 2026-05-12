import numpy as np
from scipy.optimize import differential_evolution

from gearopt.optim.fitness_ratios import fitness_ratios


def gearopt_ratios(
    i_gb,
    par,
    data,
    opts,
    obj_val_and_grad,
    con_val_jit,
    con_jac_jit,
    obj_hess_jit,
    obj_rest_val_and_grad,
    par_static,
    data_static,
    verbose=True,
):
    """
    Multi-stage differential-evolution optimization over stage ratios.

    The ratio layer delegates each candidate ratio split to ``fitness_ratios``.
    Since ``fitness_ratios`` calls the stage/discrete/continuous stack, the JAX
    objective/constraint wrappers must be passed through explicitly.

    Parameters
    ----------
    i_gb : float
        Total gearbox ratio.
    par : dict
        Parameter dictionary; must contain ``N_st``.
    data : array_like
        Loaded technical specification data.
    opts : dict
        Options dictionary.
    obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, obj_rest_val_and_grad
        JAX wrappers used by the continuous layer.
    par_static, data_static
        Hashable/static versions of ``par`` and ``data`` used by the JAX layer.
    verbose : bool, optional
        Enables diagnostic output in the ratio fitness.

    Returns
    -------
    best : dict
        Best design dictionary containing total weight, per-stage data, ratios,
        constraint diagnostics, and feasibility.
    logbook : dict
        Optimization metadata.
    """

    N_st = int(par["N_st"])
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
        f, _ = fitness_ratios(
            svec=np.asarray(s_free, dtype=float).ravel(),
            S=S,
            smin=s_min,
            smax=s_max,
            par=par,
            data=data,
            opts=opts,
            obj_val_and_grad=obj_val_and_grad,
            con_val_jit=con_val_jit,
            con_jac_jit=con_jac_jit,
            obj_hess_jit=obj_hess_jit,
            obj_rest_val_and_grad=obj_rest_val_and_grad,
            par_static=par_static,
            data_static=data_static,
            verbose=False,
        )
        return f

    # --- Parse optimizer options from config ---
    ratio_opts = opts["ratios"]
    maxiter = ratio_opts["maxiter"]
    popsize = ratio_opts["popsize"]
    mutation = ratio_opts["mutation"]
    recombination = ratio_opts["recombination"]
    tol = ratio_opts["tol"]
    disp = ratio_opts["disp"]
    polish = ratio_opts["polish"]

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
        svec=s_best,
        S=S,
        smin=s_min,
        smax=s_max,
        par=par,
        data=data,
        opts=opts,
        obj_val_and_grad=obj_val_and_grad,
        con_val_jit=con_val_jit,
        con_jac_jit=con_jac_jit,
        obj_hess_jit=obj_hess_jit,
        obj_rest_val_and_grad=obj_rest_val_and_grad,
        par_static=par_static,
        data_static=data_static,
        verbose=verbose,
    )

    # === Logbook (metadata) ===
    logbook = {
        "s_best": s_best,
        "W_best": best["W"],
        "best": best,
        "nfev": result.nfev,
        "nit": result.nit,
        "success": result.success,
        "message": result.message,
        "fun": result.fun,
    }

    return best, logbook
