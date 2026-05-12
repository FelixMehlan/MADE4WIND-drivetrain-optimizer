import numpy as np

def stage_ratio_band_from_bounds(lbI, ubI):
    """
    Compute approximate feasible stage ratio range from integer bounds.
    
    Parameters
    ----------
    lbI : array-like
        Lower integer bounds [z_s, z_p, z_r, N_p]
    ubI : array-like
        Upper integer bounds [z_s, z_p, z_r, N_p]
    
    Returns
    -------
    (rmin, rmax) : tuple of floats
        Minimum and maximum feasible stage ratios.
    """
    zs_min, zs_max = lbI[0], ubI[0]
    zp_min, zp_max = lbI[1], ubI[1]
    zr_max = ubI[2]

    # Effective max planet teeth (limited by ring geometry)
    zp_max_eff = min(zp_max, np.floor((zr_max - zs_min) / 2.0))

    rmin = 2 + 2 * (zp_min / zs_max)
    rmax = 2 + 2 * (zp_max_eff / zs_min)
    return rmin, rmax