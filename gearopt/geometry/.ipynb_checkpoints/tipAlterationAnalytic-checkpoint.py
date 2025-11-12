import numpy as np

def tipAlterationAnalytic(rb1, rb2, a, alpha_t, m_n):
    """
    Compute smooth analytical approximation of tip shortening (k_tip).

    Parameters
    ----------
    rb1, rb2 : float
        Base radii of sun and planet gears [mm].
    a : float
        Center distance [mm].
    alpha_t : float
        Transverse pressure angle [rad].
    m_n : float
        Normal module [mm].

    Returns
    -------
    k_tip : float
        Tip alteration coefficient (in units of module).
    """
    alpha = float(alpha_t)

    # Damped Newton iteration (3 steps)
    for _ in range(3):
        f = rb1 / np.cos(alpha) + rb2 / np.cos(alpha_t + alpha) - a / np.cos(alpha_t)
        df = (
            rb1 * np.sin(alpha) / (np.cos(alpha) ** 2)
            + rb2 * np.sin(alpha_t + alpha) / (np.cos(alpha_t + alpha) ** 2)
        )
        alpha -= f / (df + 1e-9)  # smooth division guard

    # Smooth soft clamp near alpha_t ± 0.3 rad
    alpha = np.clip(alpha, alpha_t - 0.3, alpha_t + 0.3)

    # Compute tip radius and normalized alteration
    r_a1_lim = rb1 / np.cos(alpha)
    k_tip = (r_a1_lim - rb1 / np.cos(alpha_t)) / m_n

    return k_tip
