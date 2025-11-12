import numpy as np

def calc_ynt(rpm: float) -> float:
    """
    Smooth lifetime factor Y_NT as a function of rotational speed (rpm).
    Mirrors the MATLAB calcYNT function exactly.

    Parameters
    ----------
    rpm : float
        Rotational speed in revolutions per minute.

    Returns
    -------
    Y_NT : float
        Lifetime factor.
    """
    # Total number of stress cycles over lifetime
    n = rpm * 25 * 8766 * 60  # 25 years × hours/year × 60 min/h

    # Exponent from empirical relationship
    m = (np.log10(1.0) - np.log10(0.93)) / (np.log10(3e6) - np.log10(110e6))

    # Constant C so that Y_NT(3e6) = 1
    C = 1.0 * (3e6) ** (-m)

    # Lifetime factor
    Y_NT = C * n ** m

    return float(Y_NT)
