import numpy as np

def calc_yx(m_n: float) -> float:
    """
    Smooth size factor Y_X as a function of normal module m_n [mm].
    Differentiable approximation of the piecewise function:

        m_n <= 5     → 1.0
        5 < m_n < 25 → 1.05 - 0.01*m_n
        m_n >= 25    → 0.8

    Parameters
    ----------
    m_n : float
        Normal module [mm].

    Returns
    -------
    Y_X : float
        Smooth size factor.
    """
    # Transition sharpness (larger k = sharper step)
    k = 5.0

    # Piecewise components
    Y1 = 1.0
    Y2 = 1.05 - 0.01 * m_n
    Y3 = 0.8

    # Smooth step transitions
    s1 = 0.5 * (1 + np.tanh(k * (m_n - 5.0)))
    s2 = 0.5 * (1 + np.tanh(k * (m_n - 25.0)))

    # Smooth blending of regions
    Y_X = (1 - s1) * Y1 + (s1 - s2) * Y2 + s2 * Y3

    return float(Y_X)
