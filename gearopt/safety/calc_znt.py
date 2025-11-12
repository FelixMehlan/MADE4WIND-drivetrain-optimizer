import numpy as np

def calc_znt(rpm):
    """
    Calculate ISO 6336-2 contact fatigue life factor Z_NT.

    Parameters
    ----------
    rpm : float or ndarray
        Rotational speed [revolutions per minute].

    Returns
    -------
    Z_NT : float or ndarray
        Contact fatigue life factor (dimensionless).
    """
    # Total load cycles (25 years, 8766 h/year, 60 min/hour)
    n = rpm * 25 * 8766 * 60

    # Slope of the log–log S–N curve (derived from reference points)
    m = (np.log10(0.948) - np.log10(0.913)) / (np.log10(281.6e6) - np.log10(964.6e6))

    # Constant from the first point
    C = 0.948 * (281.6e6) ** (-m)

    # Life factor
    Z_NT = C * n ** m

    return Z_NT
