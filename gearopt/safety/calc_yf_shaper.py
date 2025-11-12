import numpy as np
from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_acos import safe_acos
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.smooth_feps import smooth_feps


def calc_yf_shaper(d, d_Na, z, x, eps_alpha, m_n, alpha_n, h_fPf, rho_fPf, beta, eps_beta):
    """
    Gradient-friendly root geometry factors for SHAPER cutter
    (internal gear). Fully safe-div version, cleaned of redundant tiny terms.

    Returns
    -------
    Y_F : float
        Root form factor
    Y_S : float
        Stress correction factor
    """
    tiny = 1e-12  # only used for np.maximum or convergence checks

    # --- Fundamental geometry ---
    cos_an = np.cos(alpha_n)
    sin_an = np.sin(alpha_n)
    cos_b = np.cos(beta)
    beta_b = safe_asin(np.sin(beta) * cos_an)
    cos_bb = np.cos(beta_b)

    z_n = safe_div(z, cos_bb**2 * cos_b)
    eps_alpha_n = safe_div(eps_alpha, cos_bb**2)
    d_n = m_n * z_n
    d_bn = d_n * cos_an
    d_an = d_n + d_Na - d

    denom_z = abs_smooth(z)
    inner = safe_sqrt(
        safe_div(d_an**2, 4) - safe_div(d_bn**2, 4)
    ) - np.pi * d * np.cos(beta) * cos_an * safe_div(eps_alpha_n - 1, denom_z)
    d_en = safe_div(2 * z, denom_z) * safe_sqrt(inner**2 + safe_div(d_bn**2, 4))
    alpha_en = safe_acos(safe_div(d_bn, d_en))

    gamma_e = (
        safe_div(0.5 * np.pi + 2 * x * np.tan(alpha_n), z_n)
        + np.tan(alpha_n)
        - alpha_n
        - np.tan(alpha_en)
        + alpha_en
    )
    alpha_Fen = alpha_en - gamma_e

    # --- Smooth helix/contact factor ---
    f_eps = smooth_feps(eps_beta, eps_alpha_n)

    # --- Shaper geometry constants ---
    z_0 = 43.0
    theta = np.pi / 3.0
    x_0 = 0.0
    rho_a0 = 0.38 * m_n
    h_aP0 = 1.0 * m_n
    z_0v = safe_div(z_0, np.cos(beta) * cos_bb**2)

    xi = safe_div(2 * (x_0 + x), z_0v + z_n) * np.tan(alpha_n) + np.tan(alpha_n) - alpha_n

    # --- Approximate alpha_w0 (Newton refinement) ---
    alpha_w0 = np.abs(3 * xi) ** (1 / 3)
    alpha_w0 = alpha_w0 + safe_div(xi - np.tan(alpha_w0) + alpha_w0, np.tan(alpha_w0) ** 2)

    a_0 = safe_div(m_n * (z_0v + z_n), 2) * safe_div(cos_an, np.cos(alpha_w0))
    u_0 = safe_div(z_0v, z_n)
    r_w = safe_div(a_0, 1 + u_0)
    r_w0 = r_w * u_0
    r_b0 = 0.5 * m_n * z_0v * cos_an
    r_M = m_n * (safe_div(z_0v, 2) + safe_div(h_aP0, m_n) + x_0 - safe_div(rho_a0, m_n))
    alpha_M = safe_acos(safe_div(r_b0, r_M))

    delta_alpha = (
        safe_div(0.5 * np.pi + 2 * x_0 * np.tan(alpha_n), z_0v)
        - safe_div(rho_a0, r_b0)
        + np.tan(alpha_n)
        - alpha_n
        - np.tan(alpha_M)
        + alpha_M
    )

    # --- Iterative ψ solver ---
    psi0 = safe_div(np.pi, z_n) + theta
    psi = psi0

    for _ in range(8):
        lambda_ = safe_div(r_w0, r_M) * np.cos(psi)
        y = psi - safe_acos(lambda_) + delta_alpha + safe_div(psi - psi0, u_0)
        yprime = 1 + safe_div(1, u_0) - safe_div(r_w0 * np.sin(psi), r_M * safe_sqrt(1 - lambda_**2))
        step = safe_div(-y, yprime)
        psi = psi + step

    # --- Geometry ---
    delta = safe_acos(safe_div(r_w0 * np.cos(psi), r_M))
    omega_0 = delta - psi - delta_alpha
    delta_h_prime = r_M - r_w0 * np.cos(omega_0)
    delta_h = delta_h_prime * safe_div(np.sin(psi), np.sin(psi + omega_0))
    K = safe_div(delta_h, np.sin(psi))

    X = r_w * np.sin(psi - theta) - (K + rho_a0) * np.cos(theta)
    Y = r_w * np.cos(psi - theta) - (K + rho_a0) * np.sin(theta)
    s_Fn = 2 * X

    denom = safe_div(r_w0 * r_w * np.sin(psi), r_w0 + r_w) + K
    rho_F = safe_div(K**2, denom) + rho_a0
    h_Fe = safe_div(
        (np.cos(gamma_e) - np.sin(gamma_e) * np.tan(alpha_Fen)) * d_en, 2
    ) - Y

    # --- Factors ---
    Y_F = safe_div(
        6 * h_Fe * np.cos(alpha_Fen) * f_eps,
        (safe_div(s_Fn, m_n)) ** 2 * np.cos(alpha_n) * m_n,
    )

    L = safe_div(s_Fn, h_Fe)
    q_s = abs_smooth(safe_div(s_Fn, 2 * rho_F))
    Y_S = (1.2 + 0.13 * L) * q_s ** safe_div(1.0, (1.21 + safe_div(2.3, L)))

    return float(Y_F), float(Y_S)
