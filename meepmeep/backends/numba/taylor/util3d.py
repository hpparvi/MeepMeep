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
from numpy import ndarray

from .position3d import d3dc


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

    z0 = d3dc(t0, c) - zt
    z1 = d3dc(t1, c) - zt

    j = 0
    while abs(t2 - t0) > 1e-6 and j < 100:
        if z0 * z1 < 0.0:
            t1, t2 = 0.5 * (t0 + t1), t1
            z1, z2 = d3dc(t1, c) - zt, z1
        else:
            t0, t1 = t1, 0.5 * (t1 + t2)
            z0, z1 = z1, d3dc(t1, c) - zt
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
