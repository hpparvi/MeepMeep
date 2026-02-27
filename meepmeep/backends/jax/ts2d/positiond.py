#  MeepMeep: fast orbit calculations for exoplanet modelling
#  Copyright (C) 2022 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import jax
import jax.numpy as jnp

from ..ea import eccentric_anomaly

TWO_PI = 2.0 * jnp.pi
HALF_PI = 0.5 * jnp.pi


def mean_anomaly_at_transit(ecc, w):
    """Compute the Mean Anomaly at primary transit.

    Parameters
    ----------
    ecc : float
        Orbital eccentricity.
    w : float
        Argument of periastron [rad].

    Returns
    -------
    float
        Mean Anomaly at transit center [rad].
    """
    e_off = jnp.arctan2(jnp.sqrt(1.0 - ecc**2) * jnp.cos(w), ecc + jnp.sin(w))
    return e_off - ecc * jnp.sin(e_off)


def solve_xy_p5(phase, p, a, i, e, w):
    """Calculate 5th-order Taylor expansion coefficients for (x, y) position.

    Parameters
    ----------
    phase : float
        Phase angle (time) for the expansion [days].
    p : float
        Orbital period [days].
    a : float
        Semi-major axis [R_star].
    i : float
        Inclination [rad].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [rad].

    Returns
    -------
    jnp.ndarray (2, 5)
        Pre-scaled Taylor coefficients (divided by 1, 1, 2, 6, 24).
    """
    n = TWO_PI / p
    mu = n**2 * a**3

    sqe2 = jnp.sqrt(1.0 - e**2)
    ci = jnp.cos(i)
    cw = jnp.cos(w)
    sw = jnp.sin(w)

    # Mean anomaly
    offset = mean_anomaly_at_transit(e, w)
    ma = (TWO_PI * phase / p + offset) % TWO_PI

    # Eccentric anomaly (custom_jvp-enabled)
    ea = eccentric_anomaly(ma, e)
    sea = jnp.sin(ea)
    cea = jnp.cos(ea)

    # Orbital plane position
    r_val = a * (1.0 - e * cea)
    xi = a * (cea - e)
    eta = a * sqe2 * sea

    # E_dot = n * a / r
    ea_dot = n * a / r_val

    # Velocity in orbital plane
    v_xi = -a * sea * ea_dot
    v_eta = a * sqe2 * cea * ea_dot

    # Higher-order derivatives
    r2 = r_val**2
    v2 = v_xi**2 + v_eta**2
    rv = xi * v_xi + eta * v_eta

    inv_r3 = 1.0 / (r2 * r_val)
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2

    u = -mu * inv_r3
    u_dot = 3.0 * mu * rv * inv_r5
    u_ddot = 3.0 * mu * (v2 * inv_r5 - 5.0 * rv**2 * inv_r7) - 3.0 * u**2

    # Acceleration
    a_xi = u * xi
    a_eta = u * eta

    # Jerk
    j_xi = u_dot * xi + u * v_xi
    j_eta = u_dot * eta + u * v_eta

    # Snap
    s_coeff = u_ddot + u**2
    s_xi = s_coeff * xi + 2.0 * u_dot * v_xi
    s_eta = s_coeff * eta + 2.0 * u_dot * v_eta

    # Rotation to sky plane
    m00 = -cw
    m01 = sw
    m10 = -sw * ci
    m11 = -cw * ci

    cf = jnp.array([
        [m00 * xi + m01 * eta,
         m00 * v_xi + m01 * v_eta,
         (m00 * a_xi + m01 * a_eta) * 0.5,
         (m00 * j_xi + m01 * j_eta) / 6.0,
         (m00 * s_xi + m01 * s_eta) / 24.0],
        [m10 * xi + m11 * eta,
         m10 * v_xi + m11 * v_eta,
         (m10 * a_xi + m11 * a_eta) * 0.5,
         (m10 * j_xi + m11 * j_eta) / 6.0,
         (m10 * s_xi + m11 * s_eta) / 24.0],
    ])

    return cf


def solve_xy_p5_d(phase, p, a, i, e, w):
    """Calculate Taylor coefficients and their parameter derivatives.

    Uses JAX forward-mode autodiff instead of manual chain-rule propagation.

    Parameters
    ----------
    phase, p, a, i, e, w : float
        Orbital parameters (see solve_xy_p5).

    Returns
    -------
    cf : jnp.ndarray (2, 5)
        Taylor coefficients.
    dcf : jnp.ndarray (6, 2, 5)
        Parameter derivatives: dcf[k] = d(cf)/d(theta_k)
        for theta = (phase, p, a, i, e, w).
    """
    cf = solve_xy_p5(phase, p, a, i, e, w)
    dcf = jax.jacfwd(solve_xy_p5, argnums=(0, 1, 2, 3, 4, 5))(phase, p, a, i, e, w)
    dcf = jnp.stack(dcf, axis=0)  # (6, 2, 5)
    return cf, dcf


def xy_t15_d(tc, t0, p, c, dc):
    """Evaluate Taylor polynomial for (x, y) position with parameter derivatives.

    Parameters
    ----------
    tc : float
        Current time.
    t0 : float
        Taylor expansion time.
    p : float
        Orbital period.
    c : jnp.ndarray (2, 5)
        Taylor coefficients from solve_xy_p5.
    dc : jnp.ndarray (6, 2, 5)
        Parameter derivative coefficients from solve_xy_p5_d.

    Returns
    -------
    px : float
        Sky-plane x position.
    py : float
        Sky-plane y position.
    dpx : jnp.ndarray (6,)
        Derivatives of px w.r.t. (phase, p, a, i, e, w).
    dpy : jnp.ndarray (6,)
        Derivatives of py w.r.t. (phase, p, a, i, e, w).
    """
    epoch = jnp.floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)

    # Horner evaluation for position
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))

    # Horner evaluation for derivatives (vectorized over the 6 parameters)
    dpx = dc[:, 0, 0] + t * (dc[:, 0, 1] + t * (dc[:, 0, 2] + t * (dc[:, 0, 3] + t * dc[:, 0, 4])))
    dpy = dc[:, 1, 0] + t * (dc[:, 1, 1] + t * (dc[:, 1, 2] + t * (dc[:, 1, 3] + t * dc[:, 1, 4])))

    return px, py, dpx, dpy


def pd_t15_d(tc, t0, p, c, dc):
    """Calculate projected planet-star distance and parameter derivatives.

    Parameters
    ----------
    tc : float
        Current time.
    t0 : float
        Taylor expansion time.
    p : float
        Orbital period.
    c : jnp.ndarray (2, 5)
        Taylor coefficients from solve_xy_p5.
    dc : jnp.ndarray (6, 2, 5)
        Parameter derivative coefficients from solve_xy_p5_d.

    Returns
    -------
    d : float
        Projected planet-star distance.
    dd : jnp.ndarray (6,)
        Derivatives of d w.r.t. (phase, p, a, i, e, w).
    """
    px, py, dpx, dpy = xy_t15_d(tc, t0, p, c, dc)
    d = jnp.sqrt(px**2 + py**2)
    dd = (px * dpx + py * dpy) / d
    return d, dd
