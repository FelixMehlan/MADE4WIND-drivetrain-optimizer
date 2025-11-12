import numpy as np

def calc_zbd():
    """
    Calculate ISO 6336 contact stress geometry factors Z_B and Z_D.

    Returns
    -------
    Z_B : float
        Contact geometry factor for helix angle correction.
    Z_D : float
        Diameter-related geometry factor (identical to Z_B for standard gears).
    """
    f_ZCa = 1.2  # helical gears without flank modification
    Z_B = np.sqrt(f_ZCa)
    Z_D = Z_B
    return float(Z_B), float(Z_D)
