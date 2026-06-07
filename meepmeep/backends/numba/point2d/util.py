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
from numpy.typing import NDArray

from .separation import sep_c


@njit
def find_contact_point(k: float, point: int, c: NDArray):
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

    z0 = sep_c(t0, c) - zt
    z1 = sep_c(t1, c) - zt

    i = 0
    while abs(t2 - t0) > 1e-6 and i < 100:
        if z0*z1 < 0.0:
            t1, t2 = 0.5*(t0 + t1), t1
            z1, z2 = sep_c(t1, c) - zt, z1
        else:
            t0, t1 = t1, 0.5*(t1 + t2)
            z0, z1 = z1, sep_c(t1, c) - zt
        i += 1
    return t1


@njit
def bounding_box(k: float, coeffs: NDArray):
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


@njit
def t14(k: float, c: NDArray) -> float:
    """Total transit duration T14 (first to fourth contact).

    Parameters
    ----------
    k : float
        Radius ratio.
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    float
        Duration between first and fourth contact.
    """
    t1 = find_contact_point(k, 1, c)
    t4 = find_contact_point(k, 4, c)
    return t4 - t1


@njit
def t23(k: float, c: NDArray) -> float:
    """Full-transit duration T23 (second to third contact).

    Parameters
    ----------
    k : float
        Radius ratio.
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    float
        Duration between second and third contact.
    """
    t2 = find_contact_point(k, 2, c)
    t3 = find_contact_point(k, 3, c)
    return t3 - t2


@njit
def t12(k: float, c: NDArray) -> float:
    """Ingress duration T12 (first to second contact).

    Parameters
    ----------
    k : float
        Radius ratio.
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    float
        Duration between first and second contact.
    """
    t1 = find_contact_point(k, 1, c)
    t2 = find_contact_point(k, 2, c)
    return t2 - t1


@njit
def t34(k: float, c: NDArray) -> float:
    """Egress duration T34 (third to fourth contact).

    Parameters
    ----------
    k : float
        Radius ratio.
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    float
        Duration between third and fourth contact.
    """
    t3 = find_contact_point(k, 3, c)
    t4 = find_contact_point(k, 4, c)
    return t4 - t3


@njit
def t1(k: float, c: NDArray) -> float:
    """First contact time.

    Parameters
    ----------
    k : float
        Radius ratio.
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    float
        Time of first contact.
    """
    return find_contact_point(k, 1, c)


@njit
def t4(k: float, c: NDArray) -> float:
    """Fourth contact time.

    Parameters
    ----------
    k : float
        Radius ratio.
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    float
        Time of fourth contact.
    """
    return find_contact_point(k, 4, c)


@njit
def find_z_min(tc: float, c: NDArray):
    """Locate the local minimum of the projected planet-star distance.

    Uses golden-section search in a tight window around an initial guess.
    Operates in the centered coordinate system of `c` (times are offsets
    from the expansion point).

    Parameters
    ----------
    tc : float
        Initial guess for the minimum (offset from the expansion point).
    c : NDArray
        A (2, 5) Taylor coefficient matrix.

    Returns
    -------
    t_min : float
        Time of minimum projected distance.
    z_min : float
        Projected distance at the minimum.
    """
    r = 0.61803399
    cc = 1.0 - r
    x0, x3 = tc - 0.01, tc + 0.01
    x1 = tc
    x2 = tc + cc * (x3 - tc)

    f1 = sep_c(x1, c)
    f2 = sep_c(x2, c)

    j = 0
    while abs(x3 - x0) > 1e-7 and j < 100:
        if f2 < f1:
            x0, x1, x2 = x1, x2, r * x2 + cc * x3
            f1, f2 = f2, sep_c(x2, c)
        else:
            x3, x2, x1 = x2, x1, r * x1 + cc * x0
            f2, f1 = f1, sep_c(x1, c)
        j += 1

    if f1 < f2:
        return x1, f1
    else:
        return x2, f2
