import jax.numpy as jnp

def calc_yx(m_n):
    """
    JAX-compatible smooth size factor Y_X(m_n).

        m_n <= 5     → 1.0
        5 < m_n < 25 → 1.05 - 0.01*m_n
        m_n >= 25    → 0.8

    Smooth transitions via tanh blending.
    """

    k = 5.0  # transition sharpness

    # Base regions
    Y1 = 1.0
    Y2 = 1.05 - 0.01 * m_n
    Y3 = 0.8

    # Smooth step functions
    s1 = 0.5 * (1.0 + jnp.tanh(k * (m_n - 5.0)))
    s2 = 0.5 * (1.0 + jnp.tanh(k * (m_n - 25.0)))

    # Smooth blending of segments
    return (1 - s1) * Y1 + (s1 - s2) * Y2 + s2 * Y3
