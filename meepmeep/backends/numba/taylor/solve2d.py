#  MeepMeep: fast orbit calculations for exoplanet modelling
#  Copyright (C) 2022-2026 Hannu Parviainen
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

from numba import njit
from numpy import ndarray, sqrt, cos, sin, zeros, linspace, pi

from ..newton.newton import ea_from_ma
from ..utils import mean_anomaly_at_transit, TWO_PI


@njit(fastmath=True)
def solve2d(phase: float, p: float, a: float, i: float, e: float, w: float) -> ndarray:
    """ Calculate the Taylor expansion for the (x, y) position around a given phase angle.

    Parameters
    ----------
    phase : float
        Phase angle (time) for the Taylor series expansion [days].
    p : float
        Orbital period [days].
    a : float
        Semi-major axis of the orbit [R_star].
    i : float
        Inclination of the orbit [rad].
    e : float
        Eccentricity of the orbit.
    w : float
        Argument of periastron [rad].

    Returns
    -------
    ndarray
        A 2x5 coefficient matrix where each element is a pre-scaled coefficient for Taylor series expansion.
        Pre-scaling means that the coefficients are divided by 1, 1, 2, 6, and 24 to improve numerical speed.
    """
    # Analytic differentiation of Keplerian motion
    # --------------------------------------------

    # Constants
    n = TWO_PI / p
    mu = n**2 * a**3  # Standard gravitational parameter [R_star^3 / day^2]

    sqe2 = sqrt(1.0 - e**2)
    ci = cos(i)
    cw = cos(w)
    sw = sin(w)

    # 1. Calculate Mean Anomaly and Eccentric Anomaly
    # -----------------------------------------------
    # Matches the phase definition in utils.mean_anomaly for t0=0
    offset = mean_anomaly_at_transit(e, w)
    ma = (TWO_PI * (phase - (-offset * p / TWO_PI)) / p) % TWO_PI

    ea = ea_from_ma(ma, e)
    sea = sin(ea)
    cea = cos(ea)

    # 2. Orbital Plane Position & Velocity
    # ------------------------------------
    # r vector components in orbital plane (xi points to periastron)
    r_val = a * (1.0 - e * cea)
    xi = a * (cea - e)
    eta = a * sqe2 * sea

    # Derivatives of E w.r.t time: E_dot = n * a / r
    ea_dot = n * a / r_val

    # Velocity components in orbital plane
    v_xi = -a * sea * ea_dot
    v_eta = a * sqe2 * cea * ea_dot

    # 3. Higher Order Derivatives (Acceleration, Jerk, Snap)
    # ------------------------------------------------------
    # Based on recursive differentiation of a = -mu * r / |r|^3

    r2 = r_val**2
    v2 = v_xi**2 + v_eta**2
    rv = xi * v_xi + eta * v_eta  # Dot product r . v

    # u = -mu / r^3
    inv_r3 = 1.0 / (r2 * r_val)
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2

    u = -mu * inv_r3
    u_dot = 3.0 * mu * rv * inv_r5
    u_ddot = 3.0 * mu * (v2 * inv_r5 - 5.0 * rv**2 * inv_r7) - 3.0 * u**2

    # Acceleration components
    a_xi = u * xi
    a_eta = u * eta

    # Jerk components
    j_xi = u_dot * xi + u * v_xi
    j_eta = u_dot * eta + u * v_eta

    # Snap components
    s_coeff = u_ddot + u**2
    s_xi = s_coeff * xi + 2.0 * u_dot * v_xi
    s_eta = s_coeff * eta + 2.0 * u_dot * v_eta

    # 4. Rotation to Sky Plane and Coefficient Filling
    # ------------------------------------------------
    # X = -xi * cw + eta * sw
    # Y = (-xi * sw - eta * cw) * ci

    # Rotation matrix elements
    m00 = -cw
    m01 = sw
    m10 = -sw * ci
    m11 = -cw * ci

    cf = zeros((2, 5))

    # Position (0th derivative)
    cf[0, 0] = m00 * xi + m01 * eta
    cf[1, 0] = m10 * xi + m11 * eta

    # Velocity (1st derivative)
    cf[0, 1] = m00 * v_xi + m01 * v_eta
    cf[1, 1] = m10 * v_xi + m11 * v_eta

    # Acceleration / 2
    cf[0, 2] = (m00 * a_xi + m01 * a_eta) * 0.5
    cf[1, 2] = (m10 * a_xi + m11 * a_eta) * 0.5

    # Jerk / 6
    cf[0, 3] = (m00 * j_xi + m01 * j_eta) / 6.0
    cf[1, 3] = (m10 * j_xi + m11 * j_eta) / 6.0

    # Snap / 24
    cf[0, 4] = (m00 * s_xi + m01 * s_eta) / 24.0
    cf[1, 4] = (m10 * s_xi + m11 * s_eta) / 24.0

    return cf

