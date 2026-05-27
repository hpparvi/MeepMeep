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
from numpy import ndarray, sqrt, cos, sin, zeros, pi
from numpy.typing import NDArray

from ..newton.newton import ea_from_ma
from ..utils import mean_anomaly_at_transit, TWO_PI


@njit(fastmath=True)
def solve3d(t: float, p: float, a: float, i: float, e: float, w: float, lan: float = 0.0) -> NDArray:
    """ Calculate the Taylor expansion for the (x, y, z) position around a given time relative to the transit center.

    Parameters
    ----------
    t : float
        Time for the Taylor series expansion [days], t=0 for the transit center.
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
    lan : float, optional
        Longitude of the ascending node [rad]. A constant counterclockwise rotation
        of the sky-plane (x, y) coordinates about the line of sight; the line-of-sight
        (z) coordinate is unaffected. Defaults to 0.0.

    Returns
    -------
    ndarray
        A 3x5 coefficient matrix where each element is a pre-scaled coefficient for Taylor series expansion.
        Pre-scaling means that the coefficients are divided by 1, 1, 2, 6, and 24 to improve numerical speed.
    """
    # Analytic differentiation of Keplerian motion
    # --------------------------------------------

    # Constants
    n = TWO_PI / p
    mu = n**2 * a**3  # Standard gravitational parameter [R_star^3 / day^2]

    sqe2 = sqrt(1.0 - e**2)
    ci = cos(i)
    si = sin(i)
    cw = cos(w)
    sw = sin(w)

    # 1. Calculate Mean Anomaly and Eccentric Anomaly
    # -----------------------------------------------
    offset = mean_anomaly_at_transit(e, w)
    ma = (TWO_PI * (t - (-offset * p / TWO_PI)) / p) % TWO_PI

    ea = ea_from_ma(ma, e)
    sea = sin(ea)
    cea = cos(ea)

    # 2. Orbital Plane Position & Velocity
    # ------------------------------------
    r_val = a * (1.0 - e * cea)
    xi = a * (cea - e)
    eta = a * sqe2 * sea

    ea_dot = n * a / r_val

    v_xi = -a * sea * ea_dot
    v_eta = a * sqe2 * cea * ea_dot

    # 3. Higher Order Derivatives (Acceleration, Jerk, Snap)
    # ------------------------------------------------------
    r2 = r_val**2
    v2 = v_xi**2 + v_eta**2
    rv = xi * v_xi + eta * v_eta

    inv_r3 = 1.0 / (r2 * r_val)
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2

    u = -mu * inv_r3
    u_dot = 3.0 * mu * rv * inv_r5
    u_ddot = 3.0 * mu * (v2 * inv_r5 - 5.0 * rv**2 * inv_r7) - 3.0 * u**2

    a_xi = u * xi
    a_eta = u * eta

    j_xi = u_dot * xi + u * v_xi
    j_eta = u_dot * eta + u * v_eta

    s_coeff = u_ddot + u**2
    s_xi = s_coeff * xi + 2.0 * u_dot * v_xi
    s_eta = s_coeff * eta + 2.0 * u_dot * v_eta

    # 4. Rotation to Sky Frame and Coefficient Filling
    # ------------------------------------------------
    # X = -xi * cw + eta * sw          (toward observer)
    # Y = (-xi * sw - eta * cw) * ci   (sky plane)
    # Z = (xi * sw + eta * cw) * si    (above sky plane)

    m00 = -cw
    m01 = sw
    m10 = -sw * ci
    m11 = -cw * ci
    m20 = sw * si
    m21 = cw * si

    cf = zeros((3, 5))

    # Position (0th derivative)
    cf[0, 0] = m00 * xi + m01 * eta
    cf[1, 0] = m10 * xi + m11 * eta
    cf[2, 0] = m20 * xi + m21 * eta

    # Velocity (1st derivative)
    cf[0, 1] = m00 * v_xi + m01 * v_eta
    cf[1, 1] = m10 * v_xi + m11 * v_eta
    cf[2, 1] = m20 * v_xi + m21 * v_eta

    # Acceleration / 2
    cf[0, 2] = (m00 * a_xi + m01 * a_eta) * 0.5
    cf[1, 2] = (m10 * a_xi + m11 * a_eta) * 0.5
    cf[2, 2] = (m20 * a_xi + m21 * a_eta) * 0.5

    # Jerk / 6
    cf[0, 3] = (m00 * j_xi + m01 * j_eta) / 6.0
    cf[1, 3] = (m10 * j_xi + m11 * j_eta) / 6.0
    cf[2, 3] = (m20 * j_xi + m21 * j_eta) / 6.0

    # Snap / 24
    cf[0, 4] = (m00 * s_xi + m01 * s_eta) / 24.0
    cf[1, 4] = (m10 * s_xi + m11 * s_eta) / 24.0
    cf[2, 4] = (m20 * s_xi + m21 * s_eta) / 24.0

    # 5. Longitude of the ascending node
    # ----------------------------------
    # Rotate the sky-plane (x, y) coordinates about the line of sight by `lan`.
    # The rotation is constant in time and leaves the line-of-sight (z) coordinate
    # unchanged, so it applies uniformly to the x and y rows of every Taylor column.
    cO = cos(lan)
    sO = sin(lan)
    for col in range(5):
        x0 = cf[0, col]
        y0 = cf[1, col]
        cf[0, col] = cO * x0 - sO * y0
        cf[1, col] = sO * x0 + cO * y0

    return cf
