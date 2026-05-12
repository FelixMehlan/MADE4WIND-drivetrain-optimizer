import jax.numpy as jnp
from jax import lax

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_acos import safe_acos
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.smooth_feps import smooth_feps


def calc_yf_hob(
    d, d_Na, z, x, eps_alpha, m_n, alpha_n,
    h_fPf, rho_fPf, beta, eps_beta,
    debug_mode=False
):
    """
    Stable JAX version of Y_F, Y_S with deterministic Newton solver.
    """
    # ---------------------------
    # Precompute geometry terms
    # ---------------------------
    cos_an = jnp.cos(alpha_n)
    sin_an = jnp.sin(alpha_n)
    cos_b  = jnp.cos(beta)

    beta_b = safe_asin(jnp.sin(beta) * cos_an)
    cos_bb = jnp.cos(beta_b)

    z_n = safe_div(z, cos_bb**2 * cos_b)
    eps_alpha_n = safe_div(eps_alpha, cos_bb**2)

    d_n  = m_n * z_n
    d_bn = d_n * cos_an
    d_an = d_n + d_Na - d

    denom_z = abs_smooth(z)

    inner = (
        safe_sqrt(d_an**2 * 0.25 - d_bn**2 * 0.25)
        - jnp.pi * d * cos_b * cos_an * (eps_alpha_n - 1.0) / denom_z
    )

    d_en = safe_div(2.0 * z, denom_z) * safe_sqrt(inner**2 + d_bn**2 * 0.25)
    alpha_en = safe_acos(safe_div(d_bn, d_en))

    gamma_e = (
        safe_div(0.5 * jnp.pi + 2.0 * x * jnp.tan(alpha_n), z_n)
        + jnp.tan(alpha_n)
        - alpha_n
        - jnp.tan(alpha_en)
        + alpha_en
    )
    alpha_Fen = alpha_en - gamma_e
    
    f_eps = smooth_feps(eps_beta, eps_alpha_n)

    # Hob parameters
    h_fP = h_fPf * m_n
    rho_fP = rho_fPf * m_n

    E = jnp.pi/4 * m_n - h_fP * jnp.tan(alpha_n) - (1.0 - sin_an) * rho_fP / cos_an
    G = rho_fP / m_n - h_fP / m_n + x

    T = jnp.pi / 3
    H = safe_div(2.0, z_n) * (jnp.pi/2 - E/m_n) - T

    # ==============================================
    # Stable Newton solver (matches NumPy behavior)
    # ==============================================
    theta = jnp.pi / 6 # theta0
    
    # Run the first iteration
    f = safe_div(2.0 * G, z_n) * jnp.tan(theta) - H - theta
    df = safe_div(2.0 * G, z_n) / jnp.cos(theta)**2 - 1.0
    step = safe_div(-f, df)  
    theta_new = theta + step
    theta = softclip(theta_new, 5.0 * jnp.pi/180.0, 85.0 * jnp.pi/180.0, 20.0)
    
    # Run a second iteration
    f = safe_div(2.0 * G, z_n) * jnp.tan(theta) - H - theta
    df = safe_div(2.0 * G, z_n) / jnp.cos(theta)**2 - 1.0
    step = safe_div(-f, df)  
    theta_new = theta + step
    theta = softclip(theta_new, 5.0 * jnp.pi/180.0, 85.0 * jnp.pi/180.0, 20.0)
    
    # Run a third iteration
    f = safe_div(2.0 * G, z_n) * jnp.tan(theta) - H - theta
    df = safe_div(2.0 * G, z_n) / jnp.cos(theta)**2 - 1.0
    step = safe_div(-f, df)  
    theta_new = theta + step
    theta = softclip(theta_new, 5.0 * jnp.pi/180.0, 85.0 * jnp.pi/180.0, 20.0)

    # ---------------------------------------------
    # Tooth geometry
    # ---------------------------------------------
    s_Fn = m_n * (
        z_n * jnp.sin(jnp.pi/3 - theta)
        + jnp.sqrt(3.0) * (safe_div(G, jnp.cos(theta)) - rho_fP/m_n)
    )

    den = z_n * jnp.cos(theta)**2 - 2.0 * G
    den = jnp.where(jnp.abs(den) < 1e-6, jnp.sign(den) * 1e-6, den)

    rho_F = rho_fP + m_n * safe_div(
        2.0 * G**2 / jnp.cos(theta),
        den
    )

    h_Fe = (
        m_n/2.0 * (
            (jnp.cos(gamma_e) - jnp.sin(gamma_e) * jnp.tan(alpha_Fen)) * safe_div(d_en, m_n)
            - z_n * jnp.cos(jnp.pi/3 - theta)
            - (safe_div(G, jnp.cos(theta)) - rho_fP/m_n)
        )
    )

    # ---------------------------------------------
    # Final ISO 6336 factors
    # ---------------------------------------------
    Y_F = (
        safe_div(6.0 * h_Fe, m_n)
        * jnp.cos(alpha_Fen)
        / (safe_div(s_Fn, m_n))**2
        / jnp.cos(alpha_n)
        * f_eps
    )

    L = safe_div(s_Fn, h_Fe)
    q_s = safe_div(s_Fn, 2.0 * rho_F)

    Y_S = (1.2 + 0.13 * L) * q_s ** safe_div(
        1.0,
        (1.21 + 2.3 / jnp.maximum(L, 1e-6))
    )

    print(dict(
            Y_F=Y_F, Y_S=Y_S,
            z_n=z_n,
            eps_alpha_n=eps_alpha_n,
            d_en=d_en,
            alpha_en=alpha_en,
            gamma_e=gamma_e,
            alpha_Fen=alpha_Fen,
            E=E, G=G, H=H,
            theta=theta,
            s_Fn=s_Fn,
            rho_F=rho_F,
            h_Fe=h_Fe,
            #x=X,
            #y=Y,
        ))
    # ---------------------------------------------
    # Debug mode returns ALL intermediate values
    # ---------------------------------------------
    if debug_mode:
        return dict(
            Y_F=Y_F, Y_S=Y_S,
            z_n=z_n,
            eps_alpha_n=eps_alpha_n,
            d_en=d_en,
            alpha_en=alpha_en,
            gamma_e=gamma_e,
            alpha_Fen=alpha_Fen,
            E=E, G=G, H=H,
            theta=theta,
            s_Fn=s_Fn,
            rho_F=rho_F,
            h_Fe=h_Fe,
            #x=X,
            #y=Y,
        )

    return Y_F, Y_S
