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
from numpy import floor, sqrt, ndarray, pi

TWO_PI = 2.0 * pi


# Position evaluators
# -------------------

@njit(fastmath=True)
def xyz_t15(tc, t0: float, p: float, c: ndarray):
    """Calculate planet's (x, y, z) position using Taylor series expansion.

    Parameters
    ----------
    tc : float
        The current time.
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : ndarray
        A 3x5 coefficient matrix.

    Returns
    -------
    (float, float, float)
        The (x, y, z) position.
    """
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    pz = c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))
    return px, py, pz


@njit(fastmath=True)
def xyz_t15c(t: float, c: ndarray) -> tuple[float, float, float]:
    """Calculate planet's (x, y, z) position for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 3x5 coefficient matrix.

    Returns
    -------
    (float, float, float)
        The (x, y, z) position.
    """
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    pz = c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))
    return px, py, pz


@njit(fastmath=True)
def xyzd_t15c(t: float, c: ndarray) -> tuple[float, float, float, float]:
    """Calculate planet's (x, y, z) position and the projected distance for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 3x5 coefficient matrix.

    Returns
    -------
    (float, float, float, float)
        The (x, y, z) position and the projected star-planet distance.
    """
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    pz = c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))
    return px, py, pz, sqrt(px**2 + py**2)


# Projected distance evaluators
# ------------------------------

@njit(fastmath=True)
def pd_t15(tc, t0, p, c):
    """Calculate the projected planet-star center distance."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    return sqrt(px**2 + py**2)


@njit(fastmath=True)
def pd_t15c(t, c):
    """Calculate the projected planet-star center distance for t centered on the expansion time."""
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    return sqrt(px**2 + py**2)


# Z-position evaluators
# ----------------------

@njit(fastmath=True)
def z_t15(tc, t0, p, c):
    """Calculate planet's z position."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    return c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))


@njit(fastmath=True)
def z_t15c(t, c):
    """Calculate planet's z position for t centered on the expansion time."""
    return c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))


# Velocity evaluators
# --------------------

@njit(fastmath=True)
def vxyz_t15c(t: float, c: ndarray) -> tuple[float, float, float]:
    """Calculate planet's (vx, vy, vz) velocity for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 3x5 coefficient matrix.

    Returns
    -------
    (float, float, float)
        The (vx, vy, vz) velocity.
    """
    vx = c[0, 1] + t * (2.0 * c[0, 2] + t * (3.0 * c[0, 3] + t * 4.0 * c[0, 4]))
    vy = c[1, 1] + t * (2.0 * c[1, 2] + t * (3.0 * c[1, 3] + t * 4.0 * c[1, 4]))
    vz = c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))
    return vx, vy, vz


@njit(fastmath=True)
def vz_t15(tc, t0, p, c):
    """Calculate planet's z-velocity."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    return c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))


@njit(fastmath=True)
def vz_t15c(t, c):
    """Calculate planet's z-velocity for t centered on the expansion time."""
    return c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))


# Contact point and bounding box
# --------------------------------

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
        A 3x5 coefficient matrix.

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
    t2 = s * 2.0 / vx
    t1 = 0.5 * t2

    z0 = pd_t15c(t0, c) - zt
    z1 = pd_t15c(t1, c) - zt

    j = 0
    while abs(t2 - t0) > 1e-6 and j < 100:
        if z0 * z1 < 0.0:
            t1, t2 = 0.5 * (t0 + t1), t1
            z1, z2 = pd_t15c(t1, c) - zt, z1
        else:
            t0, t1 = t1, 0.5 * (t1 + t2)
            z0, z1 = z1, pd_t15c(t1, c) - zt
        j += 1
    return t1


@njit
def bounding_box(k: float, coeffs: ndarray):
    """Calculate the bounding box for a transit.

    Parameters
    ----------
    k
        Radius ratio.
    coeffs
        A 3x5 coefficient matrix.

    Returns
    -------
    tuple
        A tuple containing the T1 and T4 times.
    """
    t1 = find_contact_point(k, 1, coeffs)
    t4 = find_contact_point(k, 4, coeffs)
    return t1, t4
