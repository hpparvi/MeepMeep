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

from numba import njit
from numpy import floor, sqrt, zeros, pi

TWO_PI = 2.0 * pi


@njit(fastmath=True)
def xyz_t15c_d(t, c, dc):
    """Calculate planet's (x, y, z) position and parameter derivatives using Taylor series.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray (3, 5)
        Position Taylor coefficients from solve_xyz_p5.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients from solve_xyz_p5_d.

    Returns
    -------
    px, py, pz : float
        Sky-frame (x, y, z) position.
    dpx, dpy, dpz : ndarray (6,)
        Derivatives of (px, py, pz) w.r.t. (phase, p, a, i, e, w).
    """
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    pz = c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))

    dpx = zeros(6)
    dpy = zeros(6)
    dpz = zeros(6)
    for k in range(6):
        dpx[k] = dc[k, 0, 0] + t * (dc[k, 0, 1] + t * (dc[k, 0, 2] + t * (dc[k, 0, 3] + t * dc[k, 0, 4])))
        dpy[k] = dc[k, 1, 0] + t * (dc[k, 1, 1] + t * (dc[k, 1, 2] + t * (dc[k, 1, 3] + t * dc[k, 1, 4])))
        dpz[k] = dc[k, 2, 0] + t * (dc[k, 2, 1] + t * (dc[k, 2, 2] + t * (dc[k, 2, 3] + t * dc[k, 2, 4])))

    return px, py, pz, dpx, dpy, dpz


@njit(fastmath=True)
def xyz_t15_d(t, t0, p, c, dc):
    """Calculate planet's (x, y, z) position and parameter derivatives using Taylor series.

    Parameters
    ----------
    t : float
        The current time.
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : ndarray (3, 5)
        Position Taylor coefficients from solve_xyz_p5.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients from solve_xyz_p5_d.

    Returns
    -------
    px, py, pz : float
        Sky-frame (x, y, z) position.
    dpx, dpy, dpz : ndarray (6,)
        Derivatives of (px, py, pz) w.r.t. (phase, p, a, i, e, w).
    """
    epoch = floor((t - t0 + 0.5 * p) / p)
    return xyz_t15c_d(t - (t0 + epoch * p), c, dc)


@njit(fastmath=True)
def pd_t15c_d(t, c, dc):
    """Calculate projected planet-star distance and its parameter derivatives.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray (3, 5)
        Position Taylor coefficients from solve_xyz_p5.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients from solve_xyz_p5_d.

    Returns
    -------
    d : float
        Projected planet-star distance.
    dd : ndarray (6,)
        Derivatives of d w.r.t. (phase, p, a, i, e, w).
    """
    px, py, pz, dpx, dpy, dpz = xyz_t15c_d(t, c, dc)
    d = sqrt(px**2 + py**2)
    dd = zeros(6)
    for k in range(6):
        dd[k] = (px * dpx[k] + py * dpy[k]) / d
    return d, dd


@njit(fastmath=True)
def pd_t15_d(tc, t0, p, c, dc):
    """Calculate projected planet-star distance and its parameter derivatives.

    Parameters
    ----------
    tc : float
        The current time.
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : ndarray (3, 5)
        Position Taylor coefficients from solve_xyz_p5.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients from solve_xyz_p5_d.

    Returns
    -------
    d : float
        Projected planet-star distance.
    dd : ndarray (6,)
        Derivatives of d w.r.t. (phase, p, a, i, e, w).
    """
    epoch = floor((tc - t0 + 0.5 * p) / p)
    return pd_t15c_d(tc - (t0 + epoch * p), c, dc)
