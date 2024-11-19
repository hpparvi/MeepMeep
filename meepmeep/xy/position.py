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
from numpy import cos, sin, floor, sqrt, zeros, linspace, array, ndarray

from ..newton import ta_newton_s


@njit(fastmath=True)
def solve_xy_p5s(phase: float, p: float, a: float, i: float, e: float, w: float) -> ndarray:
    """ Calculate the Taylor expansion for the (x, y) position around a given phase angle.

    Parameters
    ----------
    phase : float
        Phase angle for the Taylor series expansion [rad].
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
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.
    """

    # Time step for central finite difference
    # ---------------------------------------
    # I've tried to choose a value that is small enough to
    # work with ultra-short-period orbits and large enough
    # not to cause floating point problems with the fourth
    # derivative (anything much smaller starts hitting the
    # double precision limit.)
    dt = 2e-2

    ae = a*(1. - e**2)
    ci = cos(i)

    # Calculation of X and Y positions
    # --------------------------------
    # These could just as well be calculated with a single
    # loop with X and Y as arrays, but I've decided to
    # manually unroll it because it seems to give a small
    # speed advantage with numba.

    f0 = ta_newton_s(phase-3 * dt, 0.0, p, e, w)
    f1 = ta_newton_s(phase-2 * dt, 0.0, p, e, w)
    f2 = ta_newton_s(phase-dt, 0.0, p, e, w)
    f3 = ta_newton_s(phase, 0.0, p, e, w)
    f4 = ta_newton_s(phase+dt, 0.0, p, e, w)
    f5 = ta_newton_s(phase+2 * dt, 0.0, p, e, w)
    f6 = ta_newton_s(phase+3 * dt, 0.0, p, e, w)

    r0 = ae/(1. + e*cos(f0))
    r1 = ae/(1. + e*cos(f1))
    r2 = ae/(1. + e*cos(f2))
    r3 = ae/(1. + e*cos(f3))
    r4 = ae/(1. + e*cos(f4))
    r5 = ae/(1. + e*cos(f5))
    r6 = ae/(1. + e*cos(f6))

    x0 = -r0*cos(w + f0)
    x1 = -r1*cos(w + f1)
    x2 = -r2*cos(w + f2)
    x3 = -r3*cos(w + f3)
    x4 = -r4*cos(w + f4)
    x5 = -r5*cos(w + f5)
    x6 = -r6*cos(w + f6)

    y0 = -r0*sin(w + f0)*ci
    y1 = -r1*sin(w + f1)*ci
    y2 = -r2*sin(w + f2)*ci
    y3 = -r3*sin(w + f3)*ci
    y4 = -r4*sin(w + f4)*ci
    y5 = -r5*sin(w + f5)*ci
    y6 = -r6*sin(w + f6)*ci

    cf = zeros((2, 5))

    cf[0, 0] = x3
    cf[1, 0] = y3

    # First time derivative of position: velocity
    # -------------------------------------------
    a, b, c = 1/60, 9/60, 45/60
    cf[0, 1] = (a*(x6 - x0) + b*(x1 - x5) + c*(x4 - x2))/dt  # vx
    cf[1, 1] = (a*(y6 - y0) + b*(y1 - y5) + c*(y4 - y2))/dt  # vy

    # Second time derivative of position: acceleration
    # ------------------------------------------------
    a, b, c, d = 1/90, 3/20, 3/2, 49/18
    cf[0, 2] = (a*(x0 + x6) - b*(x1 + x5) + c*(x2 + x4) - d*x3)/dt**2  # ax
    cf[1, 2] = (a*(y0 + y6) - b*(y1 + y5) + c*(y2 + y4) - d*y3)/dt**2  # ay

    # Third time derivative of position: jerk
    # ---------------------------------------
    a, b, c = 1/8, 1, 13/8
    cf[0, 3] = (a*(x0 - x6) + b*(x5 - x1) + c*(x2 - x4))/dt**3
    cf[1, 3] = (a*(y0 - y6) + b*(y5 - y1) + c*(y2 - y4))/dt**3

    # Fourth time derivative of position: snap
    # ----------------------------------------
    a, b, c, d = 1/6, 2, 13/2, 28/3
    cf[0, 4] = (-a*(x0 + x6) + b*(x1 + x5) - c*(x2 + x4) + d*x3)/dt**4
    cf[1, 4] = (-a*(y0 + y6) + b*(y1 + y5) - c*(y2 + y4) + d*y3)/dt**4

    return cf


@njit
def solve_xy_t25(dt, p, a, i, e, w) -> tuple[ndarray, ndarray]:
    """Calculate the Taylor series expansion at two points around the transit center.

    Parameters
    ----------
    dt : float
        Time difference between the two points.
    p : float
        Orbital parameter.
    a : float
        Semi-major axis of the orbit.
    i : float
        Orbital inclination.
    e : float
        Orbital eccentricity.
    w : float
        Argument of periapsis.

    Returns
    -------
    (ndarray, ndarray)
        Two Taylor series coefficient arrays.
    """
    c1 = array(solve_xy_p5s(-0.5*dt, p, a, i, e, w))
    c2 = array(solve_xy_p5s(0.5*dt, p, a, i, e, w))
    return c1, c2


@njit
def solve_xy_o5s(p: float, a: float, i: float, e: float, w: float, npt: int):
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
    coeffs = zeros((npt, 10))
    for ix in range(npt-1):
        coeffs[ix] = solve_xy_p5s(points[ix], p, a, i, e, w)
    coeffs[-1] = coeffs[0]
    return dt, points, coeffs


@njit(fastmath=True)
def xy_t15s(tc: float, t0: float, p: float, c: ndarray) -> tuple[float, float]:
    """Calculate planet's (x, y) position using Taylor series expansion.

    Parameters
    ----------
    tc : float
        The current time.
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : numpy.ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float)
        The (x, y) position.
    """
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    px = c[0,0] + t*(c[0,1] + t*(c[0,2]/2.0 + t*(c[0, 3]/6.0 + t*c[0,4]/24.0)))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2]/2.0 + t*(c[1, 3]/6.0 + t*c[1,4]/24.0)))
    return px, py


@njit(fastmath=True)
def xy_t15sc(t: float, c: ndarray) -> tuple[float, float]:
    """Calculate planet's (x,y) position using Taylor series expansion for t centered on the expansion time.

    Parameters
    ----------
    tc : float
        Time.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float)
        The (x, y) position.
    """
    px = c[0,0] + t*(c[0,1] + t*(c[0,2]/2.0 + t*(c[0, 3]/6.0 + t*c[0,4]/24.0)))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2]/2.0 + t*(c[1, 3]/6.0 + t*c[1,4]/24.0)))
    return px, py

#TODO: Fix the naming inconsistency with xy_t15s and xy_t15sc
@njit(fastmath=True)
def xyd_t15s(t: float, c: ndarray) -> tuple[float, float, float]:
    """Calculate planet's (x,y) position and the projected distance for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float, float)
        The (x, y) position and the projected star-planet distance.
    """
    px = c[0,0] + t*(c[0,1] + t*(c[0,2]/2.0 + t*(c[0, 3]/6.0 + t*c[0,4]/24.0)))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2]/2.0 + t*(c[1, 3]/6.0 + t*c[1,4]/24.0)))
    return px, py, sqrt(px**2 + py**2)


@njit
def xy_t15v(tc, t0, p, c):
    npt = tc.size
    xs, ys = zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i] = xy_t15s(tc[i], t0, p, c)
    return xs, ys


def xy_t15(tc, t0, p, c):
    if isinstance(tc, types.Float):
        return xy_t15s(tc, t0, p, c)
    else:
        return xy_t15v(tc, t0, p, c)


@njit(fastmath=True)
def pd_t15(tc, t0, p, c):
    """Calculate the (p)rojected planet-star center (d)istance near (t)ransit."""
    px, py = xy_t15(tc, t0, p, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def pd_t15sc(tc, c):
    """Calculate the (p)rojected planet-star center (d)istance near (t)ransit."""
    px, py = xy_t15sc(tc, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def pd_t25s(t, t0, p, dt, c1, c2):
    """Slower but more accurate (p)rojected planet-star center (d)istance near (t)ransit for scalar time.

    A more accurate version of planet-star center distance calculation that interpolates between two Taylor
    series expansions around the transit center. Much slower than `pd_t15s` and you're unlikely really going
    to need the added precision. Use `solve_xy_t25d` to compute the coefficient arrays.
    """
    epoch = floor((t - t0 + 0.5 * p) / p)
    tg = t - (t0 + epoch * p)
    dt = 0.5*dt

    if tg < dt:
        t1 = tg + dt
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1
        px1 = c1[0] + c1[2] * t1 + 0.5 * c1[4] * t2 + c1[6] * t3 / 6.0 + c1[8] * t4 / 24.
        py1 = c1[1] + c1[3] * t1 + 0.5 * c1[5] * t2 + c1[7] * t3 / 6.0 + c1[9] * t4 / 24.
    else:
        px1, py1 = 0.0, 0.0

    if tg > -dt:
        t1 = tg - dt
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t3 * t1
        px2 = c2[0] + c2[2] * t1 + 0.5 * c2[4] * t2 + c2[6] * t3 / 6.0 + c2[8] * t4 / 24.
        py2 = c2[1] + c2[3] * t1 + 0.5 * c2[5] * t2 + c2[7] * t3 / 6.0 + c2[9] * t4 / 24.
    else:
        px2, py2 = 0.0, 0.0

    if tg < -dt:
        return sqrt(px1 ** 2 + py1 ** 2)
    elif tg > dt:
        return sqrt(px2 ** 2 + py2 ** 2)
    else:
        a = (tg + dt) / (2 * dt)
        px = (1 - a) * px1 + a * px2
        py = (1 - a) * py1 + a * py2
        return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def pd_t25v(times, t0, p, dt, c1, c2):
    """Slower but more accurate (p)rojected planet-star center (d)istance near (t)ransit for a time array.

    A more accurate version of planet-star center distance calculation that interpolates between two Taylor
    series expansions around the transit center. Much slower than `pd_t15s` and you're unlikely really going
    to need the added precision. Use `solve_xy_t25d` to compute the coefficient arrays.
    """
    z = zeros(times.size)
    for i in range(times.size):
        z[i] = pd_t25s(times[i], t0, p, dt, c1, c2)
    return z
