from scipy.optimize import minimize, NonlinearConstraint, BFGS # <-- ADD BFGS HERE
import numpy as np
from scipy.stats.qmc import Sobol

import jax
import jax.numpy as jnp

from gearopt.constraints.stage_weight import stage_weight
from gearopt.constraints.stage_con import stage_con

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Values in x were outside bounds during a minimize step, clipping to bounds",
    category=RuntimeWarning,
    module=r"scipy\.optimize\._slsqp_py",
)

def gearopt_continuous(xI, i_st, i_sts, par, data, opts,
                             obj_val_and_grad, con_val_jit, con_jac_jit, obj_hess_jit, obj_rest_val_and_grad, par_static, data_static,
                             *, # <--- EVERYTHING AFTER THIS MUST BE PASSED BY KEYWORD
                             n_start_feas=None, n_start_full=None,
                      verbose=False):
    """
    Continuous optimization of db, x_s, x_p, eps using analytic JAX gradients and Hessian.
    Uses trust-constr solver for enhanced robustness and speed.
    """
    # =========================================================================
    # 0. Preparation
    # =========================================================================
    W_pen = 1e7

    lbC = np.asarray(opts["continuous"]["lbC"], float)
    ubC = np.asarray(opts["continuous"]["ubC"], float)
    ctol = np.asarray(opts["continuous"]["ctol"], float)

    i_db, i_xs, i_xp, i_eps = 0, 1, 2, 3

    db_feas0  = lbC[i_db] + 0.9*(ubC[i_db] - lbC[i_db]) 
    eps_feas0 = lbC[i_eps] + 0.1*(ubC[i_eps] - lbC[i_eps])
    
    if n_start_feas is None:
        n_start_feas = opts["continuous"].get("n_start_feas", 32)
    if n_start_full is None:
        n_start_full = opts["continuous"].get("n_start_full", 1)
    
    # Solver options
    solver_opts = {
        "maxiter": int(opts["continuous"].get("maxiter", 50)),
        "ftol": float(opts["continuous"].get("ftol", 1e-6)),
        "disp": bool(opts["continuous"].get("disp", False)),
    }
    rest_maxiter = int(opts["continuous"].get("rest_maxiter", 20))
    rest_ftol    = float(opts["continuous"].get("rest_ftol", 1e-9))
    
    xI = np.asarray(xI, float).ravel()
    bounds_phys = list(zip(lbC, ubC))

    C_REPAIR_THRESHOLD = 10000.0 * ctol
    LAMBDA_PROXIMITY = 1e-3 # Weight for proximity penalty: min(Violation) + lambda * min(||x - x_fail||^2)
    
    # =========================================================================
    # 1. Feasibility Seed Generator
    # =========================================================================
    lb_feas = np.array([lbC[i_xs], lbC[i_xp]])
    ub_feas = np.array([ubC[i_xs], ubC[i_xp]])
    
    sobol = Sobol(d=2, scramble=True)
    
    X_candidate = lb_feas + sobol.random(n_start_feas) * (ub_feas - lb_feas)

    xI_j = jnp.asarray(xI, jnp.float32) 

    def check_feasibility(xs0, xp0):
        xC = np.array([db_feas0, xs0, xp0, eps_feas0])
        xC_j = jnp.asarray(xC, jnp.float32) 
        
        C_j = con_val_jit(xI_j, xC_j, i_st, i_sts)
        C = np.asarray(C_j, float)
        
        if np.all(C <= ctol):
            return xC.copy(), float(np.max(C)), C
        return None, float(np.max(C)), C


    # =========================================================================
    # 2. JAX-compiled objective and constraints in physical xC space
    # =========================================================================

    # --- Objective NumPy wrappers --------------------------------------------
    def obj_np(xC_np):
        xC_j = jnp.asarray(xC_np, jnp.float32) 
        val, _ = obj_val_and_grad(xI_j, xC_j, i_st, i_sts)    
        return float(val)

    def obj_grad_np(xC_np):
        xC_j = jnp.asarray(xC_np, jnp.float32) 
        _, g = obj_val_and_grad(xI_j, xC_j, i_st, i_sts)
        return np.asarray(g, float)

    def obj_hess_np(xC_np):
        xC_j = jnp.asarray(xC_np, jnp.float32) 
        H = obj_hess_jit(xI_j, xC_j, i_st, i_sts)
        return np.asarray(H, float)
        
    # --- Constraints NumPy wrappers ------------------------------------------
    def ineq_np(xC_np):
        xC_j = jnp.asarray(xC_np, jnp.float32) 
        C = con_val_jit(xI_j, xC_j, i_st, i_sts)
        return -np.asarray(C, float)

    def ineq_jac_np(xC_np):
        xC_j = jnp.asarray(xC_np, jnp.float32) 
        J = con_jac_jit(xI_j, xC_j, i_st, i_sts)
        return -np.asarray(J, float)

    def obj_rest_np(xC_np, x_fail_np_j): # Pass x_fail_np_j explicitly
        xC_j = jnp.asarray(xC_np, jnp.float32)
        val, _ = obj_rest_val_and_grad(xC_j, x_fail_np_j, xI_j, i_st, i_sts)
        return float(val)

    def obj_rest_grad_np(xC_np, x_fail_np_j): # Pass x_fail_np_j explicitly
        xC_j = jnp.asarray(xC_np, jnp.float32)
        _, g = obj_rest_val_and_grad(xC_j, x_fail_np_j, xI_j, i_st, i_sts)
        return np.asarray(g, float)

    # Create NonlinearConstraint object for trust-constr
    constraints = NonlinearConstraint(
        fun=ineq_np,
        lb=0.0,
        ub=np.inf,
        jac=ineq_jac_np, 
        # BRUTE-FORCE FIX: Use the explicit BFGS object, which is always valid for the constraint Hessian.
        hess=BFGS() 
    )
    

    # =========================================================================
    # 3. Sequential Search and Optimization
    # =========================================================================
    results = []
    # NEW: Store all successful feasible seeds and their initial weight
    feasible_seeds = [] 
     
    for n_run in range(n_start_full):
        
        found_feasible_seed = False
        
        for i_seed in range(n_run, len(X_candidate)): 
            xs0, xp0 = X_candidate[i_seed]
            
            feasible_seed, Cmax_test, C_test = check_feasibility(xs0, xp0)
            
            if feasible_seed is not None:
                x0C = feasible_seed
                found_feasible_seed = True
                
                # --- CALCULATE AND STORE SEED WEIGHT ---
                Wseed = obj_np(x0C)
                feasible_seeds.append({
                    "W": Wseed,
                    "xC": x0C,
                    "Cmax": Cmax_test, # Should be <= ctol
                    "C": C_test,
                    "feasible": True
                })
                
                # --- RUN trust-constr OPTIMIZATION ---
                res = minimize(
                    obj_np,
                    x0C,
                    method="SLSQP", 
                    jac=obj_grad_np, # <--- SWITCH BACK to the JAX gradient wrapper
                    bounds=bounds_phys,
                    constraints=(
                        {
                            'type': 'ineq', 
                            'fun': ineq_np,
                            'jac': ineq_jac_np # <--- ADD JAX Constraint Jacobian wrapper back
                        } 
                    ),
                    options=solver_opts,
                )
                if verbose:
                    print("res.success", res.success, "status", res.status, "message", res.message)
                    print("res.x", res.x)

                                
                # --- POST-OPTIMIZATION FEASIBILITY CHECK & RESTORATION ---
                xCbest = np.asarray(res.x, float)
                x_phys = np.concatenate([xI, xCbest])
                par_dict = dict(par)
                data_array = np.asarray(data, dtype=np.float32).reshape(1, -1)
                C, _, _ = stage_con(x_phys, i_st, i_sts, par_dict, data_array) 
                C = np.asarray(C, float)
                Cmax_final = float(np.max(C))
                feasible_final = np.all(C <= ctol)
                if verbose:
                    WSLSQP, _ = stage_weight(x_phys, i_st, i_sts, par_dict, data_array) 
                    print(
                        f"\n[SLSQP: FINISH] "
                        f"W = {WSLSQP:.2e}, "
                        f"x = {np.array2string(x_phys, precision=3)}, "
                        f"C_max = {Cmax_final:.2e}"
                    )
                # ---------------------------------------------------------
                # NEW: Feasibility Restoration Block
                # ---------------------------------------------------------
                if (not feasible_final) and (Cmax_final > ctol) and (Cmax_final <= C_REPAIR_THRESHOLD):
                    
                    # --- DIAGNOSTIC: Restoration Triggered ---
                    if verbose:
                        print(f"\n[REPAIR: START] Triggered on xI #. C_max_initial = {Cmax_final:.2e} (>{ctol:.2e}).")
                    
                    # 1. Convert failed point to JAX array
                    x_fail_np_j = jnp.asarray(xCbest, jnp.float32)
                    
                    # 2. Run Restoration Minimization
                    res_rest = minimize(
                        lambda xC: obj_rest_np(xC, x_fail_np_j), # Use lambda to pass the constant x_fail_np_j
                        xCbest, 
                        method="SLSQP", 
                        jac=lambda xC: obj_rest_grad_np(xC, x_fail_np_j), # Use lambda here too
                        bounds=bounds_phys,
                        options={"maxiter": rest_maxiter, "ftol": rest_ftol, "disp": False},
                    )
                
                    # 3. Check the restored point
                    xCrestored = np.asarray(res_rest.x, float)
                    x_phys_restored = np.concatenate([xI, xCrestored])
                    
                    # Recalculate constraints and objective using the final restored point
                    C_restored, _, _ = stage_con(x_phys_restored, i_st, i_sts, par_dict, data_array)
                    C_restored = np.asarray(C_restored, float)
                    Cmax_restored = float(np.max(C_restored))
                    
                    # --- DIAGNOSTIC: Restoration Result ---
                    if verbose:
                        print(f"[REPAIR: RESULT] Iterations: {res_rest.nit}, Status: {res_rest.message}")
                        print(f"[REPAIR: CHECK] C_max_restored = {Cmax_restored:.2e}. Repair successful if <= {ctol:.2e}.")
                    
                    # Overwrite solution if restoration was successful
                    if Cmax_restored <= ctol:
                        xCbest = xCrestored
                        Cmax_final = Cmax_restored
                        C = C_restored
                        feasible_final = True
                        
                        # Rerun the original objective evaluation (W) at the fixed point
                        Wfixed = obj_np(xCbest)
                        res.fun = Wfixed # Update the weight with the true minimum
                        
                        # --- DIAGNOSTIC: Repair SUCCESS ---
                        if verbose:
                            print(f"[REPAIR: SUCCESS] Solution fixed and updated. New W = {Wfixed:.2f}")
                
                    else:
                        if verbose:
                        # --- DIAGNOSTIC: Repair FAILURE ---
                            print("[REPAIR: FAILURE] Restoration failed to achieve target C_max.")
                
                
                # ---------------------------------------------------------
                # END: Feasibility Restoration Block
                # ---------------------------------------------------------

                results.append(dict(
                    W=float(res.fun),
                    xC=xCbest,
                    feasible=feasible_final,
                    Cmax=Cmax_final,
                    C=C,
                ))
                
                break 
                
        if not found_feasible_seed and n_run == 0:
            return W_pen, None, False, Cmax_test, C_test, False, False, 0
        elif not found_feasible_seed:
            break
            
    # Feasibility A & B logic remains the same
    feasA = len(results) > 0
    feasB = any(r["feasible"] for r in results)

    # =========================================================================
    # 4. Final Aggregation and Return (MODIFIED)
    # =========================================================================
    if not feasB:
        # If no feasible result was produced by the SLSQP runs (even after repair),
        # we return the best *initial feasible seed* found.
        
        # Since we only run minimization when a feasible seed is found, 
        # and results is non-empty (feasA is true), feasible_seeds must be non-empty.
        
        # NEW LOGIC: Compare all initial feasible seeds found and return the one with the lowest W.
        if feasible_seeds:
            best_seed = min(feasible_seeds, key=lambda r: r["W"])
            lenFeas = len(results) 
            
            # We return the feasible seed's results
            return (
                best_seed["W"],
                best_seed["xC"],
                best_seed["feasible"], # True
                best_seed["Cmax"],
                best_seed["C"],
                feasA,
                False, # Feasibility B is technically true if a feasible seed was found
                lenFeas
            )
        else:
            # Fallback for the unlikely case where no feasible seed was found AND no optimization ran.
            # This branch should only be hit if n_run > 0 and the seed search failed, which is already
            # handled by the inner break condition, but kept for robustness.
            worst = min(results, key=lambda r: r["Cmax"])
            lenFeas = len(results) 
            return W_pen, worst["xC"], False, worst["Cmax"], worst["C"], feasA, False, lenFeas
    
    
    # If feasB is True (at least one optimization produced a feasible result)
    best = min([r for r in results if r["feasible"]], key=lambda r: r["W"])
    
    # The best overall feasible result must be the minimum of (Best SLSQP Result, Best Feasible Seed)
    # We already stored the best feasible seed, so we compare it now.
    if feasible_seeds:
        best_seed = min(feasible_seeds, key=lambda r: r["W"])
        if best_seed["W"] < best["W"]:
            best = best_seed
    
    
    return (
        best["W"],
        best["xC"],
        best["feasible"],
        best["Cmax"],
        best["C"],
        feasA,
        feasB,
        len(results), 
    )