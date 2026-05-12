import jax.numpy as jnp

def calc_zeps(eps_alpha, eps_beta):
    """
    Smooth contact ratio factor Z_eps (ISO 6336, JAX-differentiable).

    Smooth approximation of:
        if eps_beta < 1:
            Z_eps = sqrt(((4 - eps_alpha)/3 * (1 - eps_beta) + eps_beta/eps_alpha))
        else:
            Z_eps = sqrt(1/eps_alpha)

    Fully JAX-compatible (no Python branching).
    """

    k = 20.0  # transition sharpness

    eps_alpha_safe = jnp.maximum(eps_alpha, 1e-6)

    # --- Branch 1: eps_beta < 1 ---
    Z_low = jnp.sqrt(
        jnp.maximum(4.0 - eps_alpha, 0.0) / 3.0 * (1.0 - eps_beta)
        + eps_beta / eps_alpha_safe
    )

    # --- Branch 2: eps_beta > 1 ---
    Z_high = jnp.sqrt(1.0 / eps_alpha_safe)

    # --- Smooth transition ---
    s = 0.5 * (1.0 + jnp.tanh(k * (eps_beta - 1.0)))

    # --- Smooth blend ---
    return (1.0 - s) * Z_low + s * Z_high
