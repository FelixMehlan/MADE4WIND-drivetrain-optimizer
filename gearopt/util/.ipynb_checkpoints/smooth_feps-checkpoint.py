import numpy as np

def smooth_feps(eps_beta, eps_alpha_n):
    """
    Smooth transition function for contact or efficiency modeling.

    Parameters
    ----------
    eps_beta : float or array_like
        Contact ratio in the face direction.
    eps_alpha_n : float or array_like
        Normal contact ratio in the transverse direction.

    Returns
    -------
    f_eps : ndarray
        Smoothed transition factor.
    """
    eps_beta = np.asarray(eps_beta)
    eps_alpha_n = np.asarray(eps_alpha_n)
    eps_alpha_n_safe = np.maximum(eps_alpha_n, 1e-6)

    sa2 = 0.5 + 0.5 * np.tanh(50 * (2 - eps_alpha_n))
    sb0 = 0.5 + 0.5 * np.tanh(-50 * eps_beta)
    sb1 = 0.5 + 0.5 * np.tanh(50 * (eps_beta - 1))

    f1, f2 = 1.0, 0.7
    f3 = 1 - eps_beta + eps_beta / eps_alpha_n_safe
    f4 = (1 - eps_beta) / 2 + eps_beta / eps_alpha_n_safe
    f5 = eps_alpha_n_safe ** (-0.5)

    f_eps = (
        sb0 * (sa2 * f1 + (1 - sa2) * f2)
        + (1 - sb0 - sb1) * (sa2 * f3 + (1 - sa2) * f4)
        + sb1 * f5
    )

    return f_eps
