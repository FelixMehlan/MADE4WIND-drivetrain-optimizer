# gearopt/optim/jax_wrappers.py

import os
import contextlib
import numpy as np
import jax
import jax.numpy as jnp

from gearopt.constraints.stage_weight import stage_weight
from gearopt.constraints.stage_con import stage_con

def jax_wrappers(par_static, data_static, opts):
    I_ST = 3.6742
    I_STS = 1.0
    def build_jax_wrappers(par_static, data_static, lambda_proximity=1e-3, jit=True):
        par_dict = dict(par_static)
        data_array = jnp.asarray(data_static, dtype=jnp.float32).reshape(1, -1)
    
        def obj_jax(xI, xC, i_st, i_sts):
            x_phys = jnp.concatenate([xI, xC])
            W, _ = stage_weight(x_phys, i_st, i_sts, par_dict, data_array)
            return W
    
        def con_jax(xI, xC, i_st, i_sts):
            x_phys = jnp.concatenate([xI, xC])
            C, _, _ = stage_con(x_phys, i_st, i_sts, par_dict, data_array)
            return C
    
        def obj_restoration_jax(xC, x_fail, xI, i_st, i_sts):
            x_phys = jnp.concatenate([xI, xC])
            C, _, _ = stage_con(x_phys, i_st, i_sts, par_dict, data_array)
            v = jnp.maximum(0.0, C)
            violation_penalty = jnp.sum(v * v)
            proximity_penalty = lambda_proximity * jnp.sum((xC - x_fail) ** 2)
            return violation_penalty + proximity_penalty
    
        wrappers = {}
    
        if jit:
            wrappers["obj_val_and_grad"] = jax.jit(
                jax.value_and_grad(obj_jax, argnums=1),
                static_argnums=(2, 3),
            )
            wrappers["con_val_jit"] = jax.jit(con_jax, static_argnums=(2, 3))
            wrappers["con_jac_jit"] = jax.jit(
                jax.jacobian(con_jax, argnums=1),
                static_argnums=(2, 3),
            )
            wrappers["obj_hess_jit"] = jax.jit(
                jax.hessian(obj_jax, argnums=1),
                static_argnums=(2, 3),
            )
    
            #IMPORTANT FIX: xI is NOT static. Only i_st, i_sts are static.
            wrappers["obj_rest_val_and_grad"] = jax.jit(
                jax.value_and_grad(obj_restoration_jax, argnums=0),
                static_argnums=(3, 4),
            )
        else:
            wrappers["obj_val_and_grad"] = jax.value_and_grad(obj_jax, argnums=1)
            wrappers["con_val_jit"] = con_jax
            wrappers["con_jac_jit"] = jax.jacobian(con_jax, argnums=1)
            wrappers["obj_hess_jit"] = jax.hessian(obj_jax, argnums=1)
            wrappers["obj_rest_val_and_grad"] = jax.value_and_grad(obj_restoration_jax, argnums=0)
    
        return wrappers
    
    
    def warmup_jax_wrappers(
        *,
        wrappers,
        xI_full,
        lbC,
        ubC,
        i_st,
        i_sts,
        silence=True,
    ):
        """
        Force a single compile of each wrapper using representative inputs.
        Returns the warm-up inputs used (handy for debugging).
        """
        xI_warm = jnp.asarray(np.asarray(xI_full, dtype=np.float32).ravel(), jnp.float32)
    
        lbC = np.asarray(lbC, dtype=np.float32).ravel()
        ubC = np.asarray(ubC, dtype=np.float32).ravel()
        xC_warm = jnp.asarray((lbC + ubC) / 2.0, jnp.float32)
        x_fail = xC_warm
    
        def _calls():
            _ = wrappers["obj_val_and_grad"](xI_warm, xC_warm, i_st, i_sts)
            _ = wrappers["con_val_jit"](xI_warm, xC_warm, i_st, i_sts)
            _ = wrappers["con_jac_jit"](xI_warm, xC_warm, i_st, i_sts)
            _ = wrappers["obj_hess_jit"](xI_warm, xC_warm, i_st, i_sts)
            _ = wrappers["obj_rest_val_and_grad"](xC_warm, x_fail, xI_warm, i_st, i_sts)
    
        if silence:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                _calls()
        else:
            _calls()
    
        return {"xI_warm": xI_warm, "xC_warm": xC_warm, "x_fail": x_fail}
    
    
    def print_jax_cache_sizes(wrappers):
        def cache_size(f):
            return getattr(f, "_cache_size", lambda: "n/a")()
    
        print("obj cache:", cache_size(wrappers["obj_val_and_grad"]))
        print("con cache:", cache_size(wrappers["con_val_jit"]))
        print("jac cache:", cache_size(wrappers["con_jac_jit"]))
        print("hess cache:", cache_size(wrappers["obj_hess_jit"]))
        print("rest cache:", cache_size(wrappers["obj_rest_val_and_grad"]))

    wrappers = build_jax_wrappers(
            par_static=par_static,
            data_static=data_static,
            lambda_proximity=1e-3,
            jit=True,
        )
    
    print("\nWarming up JAX wrappers...")
    lbC = np.atleast_1d(opts["continuous"]["lbC"]).astype(float)
    ubC = np.atleast_1d(opts["continuous"]["ubC"]).astype(float)
    XI_FULL_WARMUP = np.array([38.0, 31.0, 102.0, 4.0, 8.0], dtype=float)
    
    _ = warmup_jax_wrappers(
        wrappers=wrappers,
        xI_full=XI_FULL_WARMUP,
        lbC=lbC,
        ubC=ubC,
        i_st=I_ST,
        i_sts=I_STS,
        silence=True,
    )
    print("Warm-up complete.\n")

    return wrappers