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
from numpy import floor, sqrt, ndarray


@njit(fastmath=True)
def p3dc(t: float, c: ndarray) -> tuple[float, float, float]:
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
def p3d(tc, t0: float, p: float, c: ndarray):
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
    return p3dc(tc - (t0 + epoch * p), c)


@njit(fastmath=True)
def d3dc(t, c):
    """Calculate the projected planet-star center distance for t centered on the expansion time."""
    px = c[0, 0] + t * (c[0, 1] + t * (c[0, 2] + t * (c[0, 3] + t * c[0, 4])))
    py = c[1, 0] + t * (c[1, 1] + t * (c[1, 2] + t * (c[1, 3] + t * c[1, 4])))
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def d3d(tc, t0, p, c):
    """Calculate the projected planet-star center distance."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    return d3dc(tc - (t0 + epoch * p), c)


@njit(fastmath=True)
def pd3dc(t: float, c: ndarray) -> tuple[float, float, float, float]:
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
    return px, py, pz, sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def pd3d(tc, t0, p, c):
    """Calculate planet's (x, y, z) position and the projected distance."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    return pd3dc(tc - (t0 + epoch * p), c)


@njit(fastmath=True)
def z3dc(t, c):
    """Calculate planet's z position for t centered on the expansion time."""
    return c[2, 0] + t * (c[2, 1] + t * (c[2, 2] + t * (c[2, 3] + t * c[2, 4])))


@njit(fastmath=True)
def z3d(tc, t0, p, c):
    """Calculate planet's z position."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    return z3dc(tc - (t0 + epoch * p), c)
