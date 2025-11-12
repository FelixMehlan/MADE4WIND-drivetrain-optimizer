import numpy as np

def stage_feasibility(lb, ub, i_st):
    """
    Generate feasible planetary integer combinations for a target stage ratio.

    Parameters
    ----------
    lb : array-like
        Lower bounds [m_n, z_s, z_p, z_r, N_p]
    ub : array-like
        Upper bounds [m_n, z_s, z_p, z_r, N_p]
    i_st : float
        Target stage ratio (≈ (i_gb)^(1/N_st))

    Returns
    -------
    candidates : ndarray
        Feasible integer combinations (rows = [m_n, z_s, z_p, z_r, N_p])
    """
    tol_ratio = 0.02
    h_a = 1.0

    m_n_range = np.arange(lb[0], ub[0] + 1)
    z_s_range = np.arange(lb[1], ub[1] + 1)
    Np_range = np.arange(lb[4], ub[4] + 1)
    dz_range = np.arange(-2, 3)

    candidates = []

    for m_n in m_n_range:
        for z_s in z_s_range:
            for N_p in Np_range:
                z_r_nom = np.floor((i_st - 1) * z_s)
                for dz_r in dz_range:
                    z_r_adj = z_r_nom - ((z_r_nom + z_s) % N_p) + dz_r * N_p
                    if not (lb[3] <= z_r_adj <= ub[3]):
                        continue

                    z_p_nom = (z_r_adj - z_s) / 2
                    for dz_p in dz_range:
                        z_p_adj = np.floor(z_p_nom) + dz_p
                        if not (lb[2] <= z_p_adj <= ub[2]):
                            continue

                        i_12a = 1 + z_r_adj / z_s
                        if abs(i_12a - i_st) / i_st > 0.1:
                            continue

                        a = 0.5 * (z_s + z_p_adj)
                        d_ap = z_p_adj + 2 * h_a
                        cond_adj = (d_ap - 2 * a * np.sin(np.pi / N_p)) / d_ap
                        if cond_adj >= -0.02:
                            continue

                        candidates.append([m_n, z_s, z_p_adj, z_r_adj, N_p])

    return np.array(candidates)
