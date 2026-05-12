import jax.numpy as jnp

def smooth_feps(eps_beta, eps_alpha_n):
    """
    JAX-differentiable smooth transition function for contact/efficiency models.

    Parameters
    ----------
    eps_beta : float or array-like (JAX tracers allowed)
        Face contact ratio.
    eps_alpha_n : float or array-like
        Normal contact ratio.

    Returns
    -------
    f_eps : jnp.ndarray
        Smooth transition factor.
    """

    eps_alpha_n_safe = jnp.maximum(eps_alpha_n, 1e-6)

    sa2 = 0.5 + 0.5 * jnp.tanh(50.0 * (2.0 - eps_alpha_n))
    sb0 = 0.5 + 0.5 * jnp.tanh(-50.0 * eps_beta)
    sb1 = 0.5 + 0.5 * jnp.tanh(50.0 * (eps_beta - 1.0))

    f1 = 1.0
    f2 = 0.7
    f3 = 1.0 - eps_beta + eps_beta / eps_alpha_n_safe
    f4 = (1.0 - eps_beta) / 2.0 + eps_beta / eps_alpha_n_safe
    f5 = eps_alpha_n_safe ** (-0.5)

    f_eps = (
        sb0 * (sa2 * f1 + (1.0 - sa2) * f2)
        + (1.0 - sb0 - sb1) * (sa2 * f3 + (1.0 - sa2) * f4)
        + sb1 * f5
    )

    return f_eps
