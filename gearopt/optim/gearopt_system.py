import numpy as np
from concurrent.futures import ThreadPoolExecutor

from gearopt.optim.gearopt_stage import gearopt_stage
from gearopt.optim.gearopt_ratios import gearopt_ratios
from gearopt.config.load_config import load_config
from gearopt.data.load_tspec import load_tspec
from gearopt.optim.jax_wrappers import jax_wrappers


def _build_jax_wrappers(par, data, opts):
    """
    Build the JAX wrappers once at system level and pass them down through all
    optimization layers.
    """
    par_static = tuple(par.items())
    data_static = tuple(np.asarray(data).ravel())

    wrappers = jax_wrappers(par_static, data_static, opts)

    return {
        "obj_val_and_grad": wrappers["obj_val_and_grad"],
        "con_val_jit": wrappers["con_val_jit"],
        "con_jac_jit": wrappers["con_jac_jit"],
        "obj_hess_jit": wrappers["obj_hess_jit"],
        "obj_rest_val_and_grad": wrappers["obj_rest_val_and_grad"],
        "par_static": par_static,
        "data_static": data_static,
    }


def gearopt_system(main_config_file):
    """
    Optimize or evaluate gearbox design across all stages.

    Parameters
    ----------
    main_config_file : str
        Main configuration file passed through to ``load_config``.

    Returns
    -------
    best : dict
        Dictionary with best design summary:
            - ``W`` : total weight
            - ``stage`` : list of per-stage solutions
            - ``ratios`` : per-stage ratio vector
            - ``Cmax`` : max constraint violation
            - ``feasible`` : overall feasibility flag
    logbook : dict
        Metadata or optimization history.
    """
    paths = load_config("paths", main_config_file)
    data = load_tspec(paths["tspec_path"])
    par = load_config("parameters", main_config_file)
    opts = load_config("options", main_config_file)
    syst = load_config("system", main_config_file)

    N_st = int(par["N_st"])
    i_gb = par["i_gb"]
    use_variable_ratios = syst["use_variable_ratios"]

    jax_args = _build_jax_wrappers(par, data, opts)

    # ============================================================
    # CASE 1: Variable ratio optimization (outer ratio layer)
    # ============================================================
    if use_variable_ratios:
        if N_st == 1:
            # Degenerate case: only one stage, no ratio optimization needed.
            W1, x1, C1, feas1 = gearopt_stage(
                stage_id=1,
                i_st=i_gb,
                i_sts=1,
                par=par,
                data=data,
                opts=opts,
                **jax_args,
            )
            best = {
                "W": W1,
                "stage": [dict(x=x1, W=W1, C=C1, feas=feas1, ratio=i_gb)],
                "ratios": np.array([i_gb]),
                "Cmax": float(np.max(np.atleast_1d(C1))) if C1 is not None else np.inf,
                "feasible": bool(feas1),
            }
            logbook = {"s_best": None, "W_best": W1, "best": best}
            return best, logbook

        # Run DE-based ratio optimization.
        best, logbook = gearopt_ratios(
            i_gb,
            par=par,
            data=data,
            opts=opts,
            **jax_args,
        )
        return best, logbook

    # ============================================================
    # CASE 2: Fixed equal per-stage ratios
    # ============================================================
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
                    opts,
                    jax_args["obj_val_and_grad"],
                    jax_args["con_val_jit"],
                    jax_args["con_jac_jit"],
                    jax_args["obj_hess_jit"],
                    jax_args["obj_rest_val_and_grad"],
                    jax_args["par_static"],
                    jax_args["data_static"],
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
        Wk, xk, Ck, feask = stage_results[k]
        total_weight += Wk
        if Ck is not None and np.size(Ck) > 0:
            Cmax = max(Cmax, float(np.max(np.atleast_1d(Ck))))
        feasible = feasible and bool(feask)
        stage_info.append(
            {"x": xk, "W": Wk, "C": Ck, "feas": bool(feask), "ratio": ratios[k]}
        )

    best = {
        "W": total_weight,
        "stage": stage_info,
        "ratios": ratios,
        "Cmax": Cmax,
        "feasible": bool(feasible),
    }

    logbook = {
        "ratios": ratios,
        "W_best": total_weight,
        "best": best,
    }

    return best, logbook
