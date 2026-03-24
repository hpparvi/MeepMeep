from numba import njit
from numpy import ndarray

from .position2d import d2dc


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

    z0 = d2dc(t0, c) - zt
    z1 = d2dc(t1, c) - zt

    i = 0
    while abs(t2 - t0) > 1e-6 and i < 100:
        if z0*z1 < 0.0:
            t1, t2 = 0.5*(t0 + t1), t1
            z1, z2 = d2dc(t1, c) - zt, z1
        else:
            t0, t1 = t1, 0.5*(t1 + t2)
            z0, z1 = z1, d2dc(t1, c) - zt
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
