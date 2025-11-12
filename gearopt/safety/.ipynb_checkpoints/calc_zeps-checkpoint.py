import numpy as np

def calc_zeps(eps_alpha, eps_beta):
    """
    Smooth contact ratio factor Z_eps (ISO 6336, differentiable form).

    Matches the piecewise definition:
        if eps_beta < 1:
            Z_eps = sqrt(((4 - eps_alpha)/3 * (1 - eps_beta) + eps_beta/eps_alpha))
        else:
            Z_eps = sqrt(1/eps_alpha)
    but ensures continuity and differentiability at eps_beta = 1.

    Parameters
    ----------
    eps_alpha : float or ndarray
        Transverse contact ratio.
    eps_beta : float or ndarray
        Overlap (face) contact ratio.

    Returns
    -------
    Z_eps : float or ndarray
        Smooth contact ratio factor.
    """
    k = 20.0  # transition sharpness

    eps_alpha_safe = np.maximum(eps_alpha, 1e-6)

    # Evaluate both branches
    Z_low = np.sqrt(
        np.maximum(4 - eps_alpha, 0) / 3.0 * (1 - eps_beta)
        + eps_beta / eps_alpha_safe
    )
    Z_high = np.sqrt(1.0 / eps_alpha_safe)

    # Smooth transition between branches
    s = 0.5 * (1.0 + np.tanh(k * (eps_beta - 1.0)))  # 0 for eps_beta<1, 1 for eps_beta>1

    # Blend smoothly
    Z_eps = (1.0 - s) * Z_low + s * Z_high

    return Z_eps
