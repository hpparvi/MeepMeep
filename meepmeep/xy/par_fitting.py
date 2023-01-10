from numba import njit
from numpy import zeros, arctan2, array

from ..utils import as_from_rhop, i_from_baew
from .position import solve_xy_p5s
from .derivatives import xy_derivative_coeffs


@njit
def diffs(p, rho, b, secw, sesw, dd: float = 1e-4):
    v = zeros((6, 6))
    a0 = as_from_rhop(rho, p)
    e0 = secw ** 2 + sesw ** 2
    w0 = arctan2(sesw, secw)
    i0 = i_from_baew(b, a0, e0, w0)

    v[:, :] = array([0.0, p, a0, i0, e0, w0])
    v[0, 0] -= dd

    # Period: p, a, and i
    a1 = as_from_rhop(rho, p + dd)
    i1 = i_from_baew(b, a1, e0, w0)
    v[1, 1] += dd
    v[1, 2] = a1
    v[1, 3] = i1

    # Stellar density: a and i
    a1 = as_from_rhop(rho + dd, p)
    i1 = i_from_baew(b, a1, e0, w0)
    v[2, 2] = a1
    v[2, 3] = i1

    # Impact parameter: i
    i1 = i_from_baew(b + dd, a0, e0, w0)
    v[3, 3] = i1

    # sqrt e cos w: i, e, and w
    e1 = (secw + dd) ** 2 + sesw ** 2
    w1 = arctan2(sesw, secw + dd)
    i1 = i_from_baew(b, a0, e1, w1)
    v[4, 3] = i1
    v[4, 4] = e1
    v[4, 5] = w1

    # sqrt e sin w: i, e, and w
    e2 = secw ** 2 + (sesw + dd) ** 2
    w2 = arctan2(sesw + dd, secw)
    i2 = i_from_baew(b, a0, e2, w2)
    v[5, 3] = i2
    v[5, 4] = e2
    v[5, 5] = w2
    return v


@njit
def coeffs(phase, p, rho, b, secw, sesw):
    a = as_from_rhop(rho, p)
    e = secw ** 2 + sesw ** 2
    w = arctan2(sesw, secw)
    i = i_from_baew(b, a, e, w)

    coeffs = solve_xy_p5s(phase, p, a, i, e, w)
    dcoeffs = xy_derivative_coeffs(diffs(p, rho, b, secw, sesw, 1e-4), 1e-4, coeffs)
    return coeffs, dcoeffs
