import jax.numpy as jnp
from jax import jit

# These must eventually be rewritten in JAX form too:
from gearopt.util.safe_div import safe_div
from gearopt.util.safe_asin import safe_asin
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth
from gearopt.util.softclip import softclip
from gearopt.util.interp_linear import interp_linear


@jit
def calc_geometry(x, par):
    """
    JAX-compatible, differentiable geometry computation.
    """

    # --- Parameters ---
    alpha_n = jnp.deg2rad(par["alpha_n_deg"])
    s_R_s = par["s_R_s"]
    s_R_p = par["s_R_p"]
    s_R_r = par["s_R_r"]

    # --- Standard module table ---
    m_n_table = jnp.array([8., 10., 12., 16., 20., 25., 32., 40., 50.])
    idxs = jnp.arange(1, len(m_n_table) + 1)

    # --- Design variables ---
    z_s, z_p, z_r, Np_raw, dm_n, db, x_s, x_p, eps_beta = x

    # --- Smooth module interpolation ---
    m_n = interp_linear(dm_n, idxs, m_n_table)

    # --- Facewidth & planet count ---
    b = db * m_n
    N_p = jnp.clip(Np_raw, 3.0, 5.0)

    # --- Helix angle ---
    arg_raw = safe_div(eps_beta * jnp.pi * m_n, b)
    beta = safe_asin(arg_raw)

    cos_beta = jnp.cos(beta)
    tan_alpha_n = jnp.tan(alpha_n)
    cos_alpha_n = jnp.cos(alpha_n)

    m_t = m_n / cos_beta
    alpha_t = jnp.arctan(tan_alpha_n / cos_beta)
    beta_b = safe_asin(jnp.sin(beta) * cos_alpha_n)
    cos_alpha_t = jnp.cos(alpha_t)

    # --- Pitch & base diameters ---
    d_p_s = z_s * m_t
    d_p_p = z_p * m_t
    d_p_r = z_r * m_t
    d_b_s = d_p_s * cos_alpha_t
    d_b_p = d_p_p * cos_alpha_t
    d_b_r = d_p_r * cos_alpha_t

    # --- Working pressure angle (sun–planet) ---
    inv_alpha_wt_sp = (
        2 * tan_alpha_n * (x_s + x_p) / jnp.maximum(z_s + z_p, 1e-9)
        + jnp.tan(alpha_t) - alpha_t
    )

    alpha_wt_sp = solve_alpha_wt_sp_newton(alpha_t, inv_alpha_wt_sp)
    cos_alpha_wt_sp = jnp.cos(alpha_wt_sp)

    # --- Working pitch diameters ---
    d_pw_s = d_b_s / jnp.maximum(cos_alpha_wt_sp, 1e-9)
    d_pw_p = d_b_p / jnp.maximum(cos_alpha_wt_sp, 1e-9)

    # --- Center distance ---
    y_sp = 0.5 * (z_s + z_p) * (cos_alpha_t / jnp.maximum(cos_alpha_wt_sp, 1e-9) - 1)
    a = ((z_s + z_p) * 0.5 + y_sp) * m_t

    # --- Ring–planet mesh ---
    den_rp = jnp.maximum(z_r - z_p, 1e-9)
    d_pw_r = 2 * a * z_r / den_rp
    d_pw_p_rp = 2 * a * z_p / den_rp
    arg_rp = d_b_r / jnp.maximum(d_pw_r, 1e-9)
    alpha_wt_rp = safe_asin(safe_sqrt(1 - arg_rp**2))

    # --- Profile shift (ring) ---
    x_rp = (
        -(jnp.tan(alpha_wt_rp) - alpha_wt_rp - jnp.tan(alpha_t) + alpha_t)
        * 0.5
        / jnp.maximum(tan_alpha_n, 1e-9)
        * (z_r - z_p)
    )
    x_r = x_rp - x_p

    # ------------------------------------------------------------
    # DIN 3967 generating profile shift xE,i (series cd, tol 25)
    # ------------------------------------------------------------
    Asn_i_s = Asn_i_cd_T25_from_d(d_p_s)
    Asn_i_p = Asn_i_cd_T25_from_d(d_p_p)
    Asn_i_r = Asn_i_cd_T25_from_d(d_p_r)

    denom_xE = jnp.maximum(2.0 * m_n * tan_alpha_n, 1e-9)

    xE_i_s = x_s + Asn_i_s / denom_xE
    xE_i_p = x_p + Asn_i_p / denom_xE
    xE_i_r = x_r + Asn_i_r / denom_xE


    # --- Diameters ---
    k_tip_sp = tipAlterationAnalytic(
        d_b_s / 2, d_b_p / 2,
        a, alpha_t, m_n,
        d_p_s, d_p_p,
        x_s, x_p,
    )

    d_a_s = d_p_s + 2 * (1 + x_s + k_tip_sp) * m_n
    d_a_p = d_p_p + 2 * (1 + x_p + k_tip_sp) * m_n
    d_a_r = d_p_r - 2 * (1 + x_r) * m_n
    d_f_s = d_a_s - 2 * (2.25 + k_tip_sp) * m_n
    d_f_p = d_a_p - 2 * (2.25 + k_tip_sp) * m_n
    d_f_r = d_a_r + 2 * 2.25 * m_n
    d_i_s = d_f_s - 2 * s_R_s * m_n
    d_i_p = d_f_p - 2 * s_R_p * m_n
    d_o_r = d_f_r + 2 * s_R_r * m_n

    # --- Contact ratios ---
    den_CR = jnp.maximum(2 * jnp.pi * m_t * cos_alpha_t, 1e-9)
    term_s = safe_sqrt(d_a_s**2 - d_b_s**2)
    term_p = safe_sqrt(d_a_p**2 - d_b_p**2)
    term_r = safe_sqrt(d_a_r**2 - d_b_r**2)

    eps_alpha_t_sp = (term_s + term_p - 2 * a * jnp.sin(alpha_wt_sp)) / den_CR
    eps_alpha_t_rp_raw = (term_p - term_r + 2 * a * jnp.sin(alpha_wt_rp)) / den_CR
    eps_alpha_t_rp = abs_smooth(eps_alpha_t_rp_raw)

    eps_1_sp = (term_s - d_pw_s * jnp.sin(alpha_wt_sp)) / den_CR
    eps_2_sp = (term_p - d_pw_p * jnp.sin(alpha_wt_sp)) / den_CR

    eps_1_rp_raw = (term_p - d_pw_p_rp * jnp.sin(alpha_wt_rp)) / den_CR
    eps_1_rp = softclip(eps_1_rp_raw, 0, eps_alpha_t_rp, 20)

    eps_2_rp_raw = (-term_r + d_pw_r * jnp.sin(alpha_wt_rp)) / den_CR
    eps_2_rp = softclip(eps_2_rp_raw, 0, 5, 20)

    return {
        "m_t": m_t,
        "alpha_t": alpha_t,
        "beta_b": beta_b,
        "alpha_wt_sp": alpha_wt_sp,
        "alpha_wt_rp": alpha_wt_rp,
        "d_p_s": d_p_s,
        "d_p_p": d_p_p,
        "d_p_r": d_p_r,
        "d_b_s": d_b_s,
        "d_b_p": d_b_p,
        "d_b_r": d_b_r,
        "d_pw_s": d_pw_s,
        "d_pw_p": d_pw_p,
        "d_pw_r": d_pw_r,
        "d_a_s": d_a_s,
        "d_a_p": d_a_p,
        "d_a_r": d_a_r,
        "d_f_s": d_f_s,
        "d_f_p": d_f_p,
        "d_f_r": d_f_r,
        "d_i_s": d_i_s,
        "d_i_p": d_i_p,
        "d_o_r": d_o_r,
        "y_sp": y_sp,
        "a": a,
        "x_r": x_r,
        "eps_alpha_t_sp": eps_alpha_t_sp,
        "eps_alpha_t_rp": eps_alpha_t_rp,
        "eps_beta": eps_beta,
        "eps_1_sp": eps_1_sp,
        "eps_2_sp": eps_2_sp,
        "eps_1_rp": eps_1_rp,
        "eps_2_rp": eps_2_rp,
        "xE_i_s": xE_i_s,
        "xE_i_p": xE_i_p,
        "xE_i_r": xE_i_r,
        "Asn_i_s_mm": Asn_i_s,
        "Asn_i_p_mm": Asn_i_p,
        "Asn_i_r_mm": Asn_i_r,

    }
    
import jax.numpy as jnp
import jax.lax as lax

def solve_alpha_wt_sp_newton(alpha_t, inv_alpha_wt_sp):
    """
    JAX-compatible Newton solver for working pressure angle.
    Performs exactly 4 Newton iterations (fixed, differentiable).
    """

    def newton_step(i, alpha_guess):
        f  = jnp.tan(alpha_guess) - alpha_guess - inv_alpha_wt_sp
        df = 1.0 / jnp.maximum(jnp.cos(alpha_guess)**2, 1e-9) - 1.0
        alpha_new = alpha_guess - f / jnp.maximum(df, 1e-9)
        return alpha_new  # SINGLE CARRY, not a tuple

    # initial guess
    alpha0 = alpha_t + 0.1

    # run fixed iterations
    alpha_final = lax.fori_loop(0, 4, newton_step, alpha0)

    return alpha_final

import jax.numpy as jnp
import jax.lax as lax

from gearopt.util.safe_div import safe_div
from gearopt.util.safe_sqrt import safe_sqrt
from gearopt.util.abs_smooth import abs_smooth


def tipAlterationAnalytic(rb1, rb2, a, alpha_t, m_n, d_p1, d_p2, x1, x2):
    """
    Tip shortening coefficient k_tip (in units of module), shared for gear1 and gear2.

    This returns a (typically small) NEGATIVE number if shortening is required,
    computed relative to nominal addendum radii:
      ra_nom = rp + (1 + x) * m_n

    Parameters
    ----------
    rb1, rb2 : float
        Base radii of gear1 and gear2 [mm].
    a : float
        Center distance [mm].
    alpha_t : float
        Transverse reference pressure angle [rad].
    m_n : float
        Normal module [mm].
    d_p1, d_p2 : float
        Pitch diameters of gear1 and gear2 [mm].
    x1, x2 : float
        Profile shifts of gear1 and gear2 [-].

    Returns
    -------
    k_tip : float
        Shared tip alteration coefficient (<= 0), in units of module.
    """

    # Nominal pitch radii
    rp1 = 0.5 * d_p1
    rp2 = 0.5 * d_p2

    # Nominal addendum radii (standard addendum 1*m_n plus profile shift)
    ra1_nom = rp1 + (1.0 + x1) * m_n
    ra2_nom = rp2 + (1.0 + x2) * m_n

    # ----- Limiting condition for max possible addendum radius -----
    # We solve for a pressure angle phi at a hypothetical limiting contact where
    # the distance along line-of-action corresponds to the center distance projection.
    #
    # A simple consistent constraint is:
    #   r1(phi) + r2(phi) = a / cos(alpha_t)
    # where r(phi) = rb / cos(phi), and phi is the transverse pressure angle at contact.
    #
    # This gives a geometric "envelope" contact point on the line of action.

    target = safe_div(a, jnp.cos(alpha_t))

    def f(phi):
        return safe_div(rb1, jnp.cos(phi)) + safe_div(rb2, jnp.cos(phi)) - target

    def df(phi):
        # derivative of rb/cos(phi) is rb*sin(phi)/cos(phi)^2
        c = jnp.cos(phi)
        s = jnp.sin(phi)
        return rb1 * s / (c * c) + rb2 * s / (c * c)

    # Initial guess: start near alpha_t (safe)
    phi0 = alpha_t

    def newton_step(_, phi):
        phi_next = phi - safe_div(f(phi), df(phi) + 1e-9)
        return phi_next

    # Few fixed iterations (differentiable)
    phi = lax.fori_loop(0, 5, newton_step, phi0)

    # Clamp to avoid invalid cos(phi) near pi/2
    phi = jnp.clip(phi, 1e-4, 1.2)  # ~69° upper bound; adjust if needed

    # Limiting radii from this phi
    r1_lim = safe_div(rb1, jnp.cos(phi))
    r2_lim = safe_div(rb2, jnp.cos(phi))

    # Convert to limiting addendum radii in a conservative way:
    # The addendum circle radius must be at least the radius at limiting contact.
    ra1_lim = r1_lim
    ra2_lim = r2_lim

    # Compute the required shortening (negative or zero) for each gear:
    k1 = safe_div(ra1_lim - ra1_nom, m_n)   # <=0 if ra1_lim < ra1_nom
    k2 = safe_div(ra2_lim - ra2_nom, m_n)

    # Shared k_tip must satisfy both => take the most negative (most restrictive)
    k_tip = jnp.minimum(k1, k2)

    # Do not allow lengthening via this mechanism
    k_tip = jnp.minimum(k_tip, 0.0)

    return k_tip


def din3967_band_value(d_mm, edges, values, k=20.0):
    """
    Smooth piecewise-constant lookup:
      - edges: monotonically increasing band upper edges [mm], length N
      - values: value for each band, length N
    Behavior:
      returns approx values[i] when d is in band i.
      Smooth transitions around edges using tanh.
    """
    # Clamp softly to range to avoid extrapolation artifacts
    d0 = edges[0]
    d1 = edges[-1]
    d = softclip(d_mm, d0, d1, 20.0)

    # weights for each band:
    # band i active when edges[i-1] < d <= edges[i]
    # implement as w_i = H(d - lower) - H(d - upper)
    lowers = jnp.concatenate([jnp.array([0.0], dtype=edges.dtype), edges[:-1]])
    uppers = edges

    H_low = 0.5 * (1.0 + jnp.tanh(k * (d - lowers)))
    H_up = 0.5 * (1.0 + jnp.tanh(k * (d - uppers)))
    w = H_low - H_up  # ~1 inside band, ~0 outside

    # normalize (numerical safety)
    w = w / jnp.maximum(jnp.sum(w), 1e-9)

    return jnp.sum(w * values)

def Asn_i_cd_T25_from_d(d_mm: jnp.ndarray, k=20.0) -> jnp.ndarray:
    """
    DIN 3967:
      Asn_i = Asn_e - T_sn
    with:
      Table 1: series 'cd' for Asn_e (µm)
      Table 2: toleranzreihe 25 for T_sn (µm)
    Stepwise by diameter band, smoothed transitions.
    Returns mm.
    """

    edges = jnp.array(
        [10., 50., 125., 280., 560., 1000., 1600., 2500., 4000., 6300., 10000.],
        dtype=jnp.float32
    )

    # Table 1 (series cd), µm
    Asn_e_cd_um = jnp.array(
        [-40., -54., -70., -95., -130., -175., -240., -320., -430., -580., -780.],
        dtype=jnp.float32
    )

    # Table 2 (toleranzreihe 25), µm
    Tsn_25_um = jnp.array(
        [20., 30., 40., 50., 60., 80., 100., 130., 160., 200., 250.],
        dtype=jnp.float32
    )

    Asn_e_um = din3967_band_value(d_mm, edges, Asn_e_cd_um, k=k)
    Tsn_um   = din3967_band_value(d_mm, edges, Tsn_25_um, k=k)

    Asn_i_um = Asn_e_um - Tsn_um
    return Asn_i_um * 1e-3  # µm -> mm
