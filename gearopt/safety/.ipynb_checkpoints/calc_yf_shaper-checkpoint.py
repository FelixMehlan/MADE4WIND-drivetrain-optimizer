import jax.numpy as jnp
import jax.lax as lax

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_acos import safe_acos
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.smooth_feps import smooth_feps


def calc_yf_shaper(d, d_Na, z, x, eps_alpha, m_n,
                   alpha_n, h_fPf, rho_fPf, beta, eps_beta, debug_mode=False):
    """
    JAX-compatible version of ISO 6336 shaper-cut internal gear
    root geometry factors Y_F and Y_S.
    """

    cos_an = jnp.cos(alpha_n)
    sin_an = jnp.sin(alpha_n)
    cos_b  = jnp.cos(beta)

    beta_b = safe_asin(jnp.sin(beta) * cos_an)
    cos_bb = jnp.cos(beta_b)

    # Normalised tooth count
    z_n = safe_div(z, cos_bb**2 * cos_b)
    eps_alpha_n = safe_div(eps_alpha, cos_bb**2)

    d_n  = m_n * z_n
    d_bn = d_n * cos_an
    d_an = d_n + d_Na - d

    denom_z = abs_smooth(z)

    inner = (
        safe_sqrt(safe_div(d_an**2, 4.0) - safe_div(d_bn**2, 4.0))
        - jnp.pi * d * cos_b * cos_an * safe_div(eps_alpha_n - 1.0, denom_z)
    )

    d_en = safe_div(2.0 * z, denom_z) * safe_sqrt(inner**2 + safe_div(d_bn**2, 4.0))
    alpha_en = safe_acos(safe_div(d_bn, d_en))

    gamma_e = (
        safe_div(0.5 * jnp.pi + 2 * x * jnp.tan(alpha_n), z_n)
        + jnp.tan(alpha_n)
        - alpha_n
        - jnp.tan(alpha_en)
        + alpha_en
    )

    alpha_Fen = alpha_en - gamma_e

    # Smooth helix/contact factor
    f_eps = smooth_feps(eps_beta, eps_alpha_n)

    # ---------------------------------------------------------
    # Shaper cutter fundamental geometry
    # ---------------------------------------------------------
    z_0 = 36.0
    theta = jnp.pi / 3.0
    
    x_0 = 0.0
    rho_a0 = 0.38 * m_n
    h_aP0 = 1.0 * m_n

    z_0v = safe_div(z_0, (jnp.cos(beta) * cos_bb**2))

    xi = (
        safe_div(2 * (x_0 + x), z_0v + z_n) * jnp.tan(alpha_n)
        + jnp.tan(alpha_n)
        - alpha_n
    )

    # Initial α_w0 estimate + Newton refinement (1 step)
    alpha_w0 = jnp.abs(3.0 * xi) ** (1.0 / 3.0)
    alpha_w0 = alpha_w0 + safe_div(
        xi - jnp.tan(alpha_w0) + alpha_w0,
        jnp.tan(alpha_w0)**2
    )

    a_0 = safe_div(m_n * (z_0v + z_n), 2.0) * safe_div(cos_an, jnp.cos(alpha_w0))
    u_0 = safe_div(z_0v, z_n)

    r_w  = safe_div(a_0, 1.0 + u_0)

    r_w0 = r_w * u_0
    r_b0 = 0.5 * m_n * z_0v * cos_an

    r_M = m_n * (
        safe_div(z_0v, 2.0)
        + safe_div(h_aP0, m_n)
        + x_0
        - safe_div(rho_a0, m_n)
    )
    
    alpha_M = safe_acos(safe_div(r_b0, r_M))

    delta_alpha = (
        safe_div(0.5 * jnp.pi + 2 * x_0 * jnp.tan(alpha_n), z_0v)
        - safe_div(rho_a0, r_b0)
        + jnp.tan(alpha_n)
        - alpha_n
        - jnp.tan(alpha_M)
        + alpha_M
    )

    # ---------------------------------------------------------
    # Newton iteration for ψ (8 iterations) via jax.lax.fori_loop
    # ---------------------------------------------------------
    
    psi0 = safe_div(jnp.pi, z_n) + theta
    
    def psi_step(i, psi):
        lam = safe_div(r_w0, r_M) * jnp.cos(psi)
    
        y = (
            psi
            - safe_acos(lam)
            + delta_alpha
            + safe_div((psi - psi0), u_0)
        )
    
        yprime = (
            1.0
            + safe_div(1.0, u_0)
            - safe_div(
                r_w0 * jnp.sin(psi),
                r_M * safe_sqrt(1.0 - lam**2)
            )
        )
    
        psi_next = psi + safe_div(-y, yprime)

        return psi_next   # <-- MUST return only the updated psi


    psi = lax.fori_loop(0, 8, psi_step, psi0)
   

    # ---------------------------------------------------------
    # Internal gear geometry results
    # ---------------------------------------------------------
    delta = safe_acos(safe_div(r_w0 * jnp.cos(psi), r_M))
    omega_0 = delta - psi - delta_alpha

    delta_h_prime = r_M - r_w0 * jnp.cos(omega_0)
    delta_h = delta_h_prime * safe_div(jnp.sin(psi), jnp.sin(psi + omega_0))
    K = safe_div(delta_h, jnp.sin(psi))
    X = r_w * jnp.sin(psi - theta) - (K + rho_a0) * jnp.cos(theta)
    Y = r_w * jnp.cos(psi - theta) - (K + rho_a0) * jnp.sin(theta)


    s_Fn = 2.0 * X

    denom = safe_div(r_w0 * r_w * jnp.sin(psi), r_w0 + r_w) + K
    rho_F = safe_div(K**2, denom) + rho_a0
    
    h_Fe = safe_div(
        (jnp.cos(gamma_e) - jnp.sin(gamma_e) * jnp.tan(alpha_Fen)) * d_en,
        2.0
    ) - Y

    # ---------------------------------------------------------
    # Root form factor Y_F
    # ---------------------------------------------------------
    Y_F = safe_div(
        6.0 * h_Fe * jnp.cos(alpha_Fen) * f_eps,
        (safe_div(s_Fn, m_n))**2 * jnp.cos(alpha_n) * m_n,
    )

    # ---------------------------------------------------------
    # Stress correction factor Y_S
    # ---------------------------------------------------------
    L  = safe_div(s_Fn, h_Fe)
    q_s = abs_smooth(safe_div(s_Fn, 2.0 * rho_F))

    Y_S = (1.2 + 0.13 * L) * q_s ** safe_div(
        1.0,
        (1.21 + safe_div(2.3, L))
    )
    if debug_mode:
        return dict(
            Y_F=Y_F, Y_S=Y_S,
            z_n=z_n,
            eps_alpha_n=eps_alpha_n,
            d_en=d_en,
            alpha_en=alpha_en,
            gamma_e=gamma_e,
            alpha_Fen=alpha_Fen,
            s_Fn=s_Fn,
            rho_F=rho_F,
            h_Fe=h_Fe,
            f_eps=f_eps,
            x=X,
            y=Y,
            psi=psi,
            delta=delta,
            omega_0=omega_0,
            delta_alpha=delta_alpha,
            r_w=r_w,
            r_w0=r_w0,
            r_M=r_M,
            K=K,
            theta=theta,
        )
    
    return Y_F, Y_S
