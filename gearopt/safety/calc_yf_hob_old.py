import numpy as np
from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_acos import safe_acos
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.smooth_feps import smooth_feps


def calc_yf_hob(
    d, d_Na, z, x, eps_alpha, m_n, alpha_n,
    h_fPf, rho_fPf, beta, eps_beta
):
    """
    Gradient-friendly root geometry factors for a hobbed gear tooth.
    WITH PRINT STATEMENTS FOR DEBUGGING.
    """

    print("\n==== calc_yf_hob DEBUG ====")

    tiny = 1e-12

    cos_an = np.cos(alpha_n)
    sin_an = np.sin(alpha_n)
    cos_b  = np.cos(beta)

    beta_b = safe_asin(np.sin(beta) * cos_an)
    cos_bb = np.cos(beta_b)

    z_n = safe_div(z, cos_bb**2 * cos_b)
    eps_alpha_n = safe_div(eps_alpha, cos_bb**2)

    print("z_n =", z_n)
    print("eps_alpha_n =", eps_alpha_n)

    d_n  = m_n * z_n
    d_bn = d_n * cos_an
    d_an = d_n + d_Na - d

    denom_z = abs_smooth(z)

    inner = (
        safe_sqrt(d_an**2 / 4 - d_bn**2 / 4)
        - np.pi * d * np.cos(beta) * cos_an * (eps_alpha_n - 1) / denom_z
    )

    d_en = safe_div(2 * z, denom_z) * safe_sqrt(inner**2 + d_bn**2 / 4)
    alpha_en = safe_acos(safe_div(d_bn, d_en))

    print("d_en =", d_en)
    print("alpha_en =", alpha_en)

    gamma_e = (
        safe_div(0.5 * np.pi + 2 * x * np.tan(alpha_n), z_n)
        + np.tan(alpha_n)
        - alpha_n
        - np.tan(alpha_en)
        + alpha_en
    )

    alpha_Fen = alpha_en - gamma_e
    print("gamma_e =", gamma_e)
    print("alpha_Fen =", alpha_Fen)

    # Smooth facewidth factor
    f_eps = smooth_feps(eps_beta, eps_alpha_n)

    # Hob geometry
    h_fP = h_fPf * m_n
    rho_fP = rho_fPf * m_n

    E = np.pi / 4 * m_n - h_fP * np.tan(alpha_n) - (1 - sin_an) * rho_fP / cos_an
    G = rho_fP / m_n - h_fP / m_n + x

    T = np.pi / 3
    H = safe_div(2, z_n) * (np.pi / 2 - E / m_n) - T

    print("E =", E, " G =", G, " H =", H)

    # --- Newton iteration for θ ---
    theta = np.pi / 6  # initial guess
    print("\nNewton Solver for theta:")
    for k in range(12):
        tanth = np.tan(theta)
        f = safe_div(2 * G, z_n) * tanth - H - theta
        df = safe_div(2 * G, z_n) / np.cos(theta)**2 - 1

        step = safe_div(-f, df)
        theta_new = theta + step
        theta_clipped = softclip(
            theta_new,
            5 * np.pi / 180,
            85 * np.pi / 180,
            20,
        )

        print(f" iter {k:2d}: theta = {theta:.6f}, step = {step:.6f}, clipped = {theta_clipped:.6f}")

        if abs_smooth(theta_clipped - theta) < 1e-6:
            print(" Converged.")
            theta = theta_clipped
            break

        theta = theta_clipped

    print("Final theta =", theta)

    # Tooth geometry
    s_Fn = m_n * (
        z_n * np.sin(np.pi / 3 - theta)
        + np.sqrt(3) * (safe_div(G, np.cos(theta)) - rho_fP / m_n)
    )

    rho_F = rho_fP + m_n * safe_div(
        2 * G**2 / np.cos(theta),
        z_n * np.cos(theta)**2 - 2 * G
    )

    h_Fe = (
        m_n / 2
        * (
            (np.cos(gamma_e) - np.sin(gamma_e) * np.tan(alpha_Fen))
            * safe_div(d_en, m_n)
            - z_n * np.cos(np.pi / 3 - theta)
            - (safe_div(G, np.cos(theta)) - rho_fP / m_n)
        )
    )

    print("s_Fn =", s_Fn)
    print("rho_F =", rho_F)
    print("h_Fe =", h_Fe)

    # Factors
    Y_F = (
        safe_div(6 * h_Fe, m_n)
        * np.cos(alpha_Fen)
        / (safe_div(s_Fn, m_n)) ** 2
        / np.cos(alpha_n)
        * f_eps
    )

    L = safe_div(s_Fn, h_Fe)
    q_s = safe_div(s_Fn, 2 * rho_F)
    Y_S = (1.2 + 0.13 * L) * q_s ** safe_div(1.0, (1.21 + 2.3 / np.maximum(L, 1e-6)))

    print("Y_F =", Y_F)
    print("Y_S =", Y_S)
    print("==== END ====\n")

    return float(Y_F), float(Y_S)
