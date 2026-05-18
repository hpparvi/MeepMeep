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
from numpy import floor, sqrt, sin, cos, pi, zeros


@njit(fastmath=True)
def v3dc_d(t, c, dc):
    """Calculate planet's (vx, vy, vz) velocity and parameter derivatives.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray (3, 5)
        Position Taylor coefficients.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients.

    Returns
    -------
    vx, vy, vz : float
        Velocity components.
    dvx, dvy, dvz : ndarray (6,)
        Derivatives of (vx, vy, vz) w.r.t. (phase, p, a, i, e, w).
    """
    vx = c[0, 1] + t * (2.0 * c[0, 2] + t * (3.0 * c[0, 3] + t * 4.0 * c[0, 4]))
    vy = c[1, 1] + t * (2.0 * c[1, 2] + t * (3.0 * c[1, 3] + t * 4.0 * c[1, 4]))
    vz = c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))

    dvx = zeros(6)
    dvy = zeros(6)
    dvz = zeros(6)
    for k in range(6):
        dvx[k] = dc[k, 0, 1] + t * (2.0 * dc[k, 0, 2] + t * (3.0 * dc[k, 0, 3] + t * 4.0 * dc[k, 0, 4]))
        dvy[k] = dc[k, 1, 1] + t * (2.0 * dc[k, 1, 2] + t * (3.0 * dc[k, 1, 3] + t * 4.0 * dc[k, 1, 4]))
        dvz[k] = dc[k, 2, 1] + t * (2.0 * dc[k, 2, 2] + t * (3.0 * dc[k, 2, 3] + t * 4.0 * dc[k, 2, 4]))

    return vx, vy, vz, dvx, dvy, dvz


@njit(fastmath=True)
def vzc_d(t, c, dc):
    """Calculate planet's z-velocity and parameter derivatives.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray (3, 5)
        Position Taylor coefficients.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients.

    Returns
    -------
    vz : float
        Z-velocity.
    dvz : ndarray (6,)
        Derivatives of vz w.r.t. (phase, p, a, i, e, w).
    """
    vz = c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))
    dvz = zeros(6)
    for k in range(6):
        dvz[k] = dc[k, 2, 1] + t * (2.0 * dc[k, 2, 2] + t * (3.0 * dc[k, 2, 3] + t * 4.0 * dc[k, 2, 4]))
    return vz, dvz


@njit(fastmath=True)
def vz_d(t, t0, p, c, dc):
    """Calculate planet's z-velocity and parameter derivatives.

    Parameters
    ----------
    t : float
        The current time.
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : ndarray (3, 5)
        Position Taylor coefficients.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients.

    Returns
    -------
    vz : float
        Z-velocity.
    dvz : ndarray (6,)
        Derivatives of vz w.r.t. (phase, p, a, i, e, w).
    """
    epoch = floor((t - t0 + 0.5 * p) / p)
    return vzc_d(t - (t0 + epoch * p), c, dc)


@njit(fastmath=True)
def rvc_d(t, k, p, a, i, e, c, dc):
    """Calculate radial velocity and parameter derivatives.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    k : float
        RV semi-amplitude.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    c : ndarray (3, 5)
        Position Taylor coefficients.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients.

    Returns
    -------
    rv : float
        Radial velocity.
    drv : ndarray (6,)
        Derivatives of rv w.r.t. (phase, p, a, i, e, w).
    """
    n = 2.0 * pi / p * (a * sin(i)) / sqrt(1.0 - e ** 2)
    s = k / n

    vz, dvz = vzc_d(t, c, dc)
    rv_val = s * vz

    # ds/dθ for each parameter: phase, p, a, i, e, w
    drv = zeros(6)
    ds = zeros(6)
    ds[1] = s / p       # ds/dp
    ds[2] = -s / a      # ds/da
    ds[3] = -s * cos(i) / sin(i)  # ds/di
    ds[4] = -s * e / (1.0 - e ** 2)  # ds/de

    for j in range(6):
        drv[j] = s * dvz[j] + vz * ds[j]

    return rv_val, drv


@njit(fastmath=True)
def rv_d(t, k, t0, p, a, i, e, c, dc):
    """Calculate radial velocity and parameter derivatives.

    Parameters
    ----------
    t : float
        The current time.
    k : float
        RV semi-amplitude.
    t0 : float
        The Taylor series expansion time.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    c : ndarray (3, 5)
        Position Taylor coefficients.
    dc : ndarray (6, 3, 5)
        Parameter derivative coefficients.

    Returns
    -------
    rv : float
        Radial velocity.
    drv : ndarray (6,)
        Derivatives of rv w.r.t. (phase, p, a, i, e, w).
    """
    epoch = floor((t - t0 + 0.5 * p) / p)
    return rvc_d(t - (t0 + epoch * p), k, p, a, i, e, c, dc)
