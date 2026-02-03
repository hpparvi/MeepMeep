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

from numba import njit, types
from numpy import cos, sin, floor, sqrt, zeros, linspace, array, ndarray, pi

from ..newton.newton import ea_from_ma
from ..utils import mean_anomaly_at_transit

TWO_PI = 2.0 * pi

@njit(fastmath=True)
def solve_xy_p5(phase: float, p: float, a: float, i: float, e: float, w: float) -> ndarray:
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


@njit
def solve_xy_o5(p: float, a: float, i: float, e: float, w: float, npt: int):
    """Calculate the 2D Taylor series expansion for a Keplerian orbit in npt points along the orbit.

    Parameters
    ----------
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
    npt : int
        Number of points.

    Returns
    -------
    dt : float
        Time interval between points.
    points : ndarray
        Array of points in the range [0, p].
    coeffs : ndarray
        Array of coefficients calculated for each point.
    """
    points = linspace(0.0, p, npt)
    dt = points[1] - points[0]
    coeffs = zeros((npt, 2, 5))
    for ix in range(npt-1):
        coeffs[ix, :, :] = solve_xy_p5(points[ix], p, a, i, e, w)
    coeffs[-1] = coeffs[0]
    return dt, points, coeffs


@njit(fastmath=True)
def xy_t15(tc, t0: float, p: float, c: ndarray):
    """Calculate planet's (x, y) position using Taylor series expansion.

    Automatically works with both scalar and array time inputs through broadcasting.

    Parameters
    ----------
    tc : float or ndarray
        The current time(s).
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    tuple[float, float] or tuple[ndarray, ndarray]
        The (x, y) position(s). Returns scalars for scalar input, arrays for array input.
    """
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    px = c[0,0] + t*(c[0,1] + t*(c[0,2] + t*(c[0, 3] + t*c[0,4])))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2] + t*(c[1, 3] + t*c[1,4])))
    return px, py


@njit(fastmath=True)
def xy_t15c(t: float, c: ndarray) -> tuple[float, float]:
    """Calculate planet's (x,y) position using Taylor series expansion for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float)
        The (x, y) position.
    """
    px = c[0,0] + t*(c[0,1] + t*(c[0,2] + t*(c[0, 3] + t*c[0,4])))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2] + t*(c[1, 3] + t*c[1,4])))
    return px, py


@njit(fastmath=True)
def xyd_t15c(t: float, c: ndarray) -> tuple[float, float, float]:
    """Calculate planet's (x,y) position and the projected distance for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float, float)
        The (x, y) position and the projected star-planet distance.
    """
    px = c[0,0] + t*(c[0,1] + t*(c[0,2] + t*(c[0, 3] + t*c[0,4])))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2] + t*(c[1, 3] + t*c[1,4])))
    return px, py, sqrt(px**2 + py**2)


@njit(fastmath=True)
def pd_t15(tc, t0, p, c):
    """Calculate the (p)rojected planet-star center (d)istance near (t)ransit."""
    px, py = xy_t15(tc, t0, p, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def pd_t15c(tc, c):
    """Calculate the (p)rojected planet-star center (d)istance near (t)ransit."""
    px, py = xy_t15c(tc, c)
    return sqrt(px ** 2 + py ** 2)


@njit
def find_contact_point(k: float, point: int, c: ndarray):
    """Find the contact point time for a planet.

    Parameters
    ----------
    k
        Radius ratio.
    point
        Contact point, can be 1, 2, 3, or 4.
    c
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    float
        The calculated contact point time.

    """
    if point == 1 or point == 2 or point == 12:
        s = -1.0
    else:
        s = 1.0

    if point == 1 or point == 4:
        zt = 1.0 + k
    elif point == 2 or point == 3:
        zt = 1.0 - k
    else:
        zt = 1.0

    vx = c[0, 1]

    t0 = 0.0
    t2 = s*2.0/vx
    t1 = 0.5*t2

    z0 = pd_t15c(t0, c) - zt
    z1 = pd_t15c(t1, c) - zt

    i = 0
    while abs(t2 - t0) > 1e-6 and i < 100:
        if z0*z1 < 0.0:
            t1, t2 = 0.5*(t0 + t1), t1
            z1, z2 = pd_t15c(t1, c) - zt, z1
        else:
            t0, t1 = t1, 0.5*(t1 + t2)
            z0, z1 = z1, pd_t15c(t1, c) - zt
        i += 1
    return t1


@njit
def bounding_box(k: float, coeffs: ndarray):
    """Calculate the bounding box for a transit.


    Parameters
    ----------
    k
        Radius ratio.
    coeffs
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.


    Returns
    -------
    tuple
        A tuple containing the T1 and T4 times.
    """
    t1 = find_contact_point(k, 1, coeffs)
    t4 = find_contact_point(k, 4, coeffs)
    return t1, t4