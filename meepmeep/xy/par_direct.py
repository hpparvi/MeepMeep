from numba import njit
from numpy import zeros, array, atleast_1d, asarray

from .position import solve_xy_p5s
from .derivatives import xy_derivative_coeffs, loc_and_der_coeffs


@njit
def partial_derivatives():
    v = zeros((6, 6))
    v[0, 0] = -1.0
    for i in range(1, 6):
        v[i, i] = 1.0
    return v


@njit
def coeffs_old(phase, p, a, i, e, w):
    coeffs = solve_xy_p5s(phase, p, a, i, e, w)
    pds =  partial_derivatives()
    dcoeffs = xy_derivative_coeffs(0.0, p, a, i, e, w, pds, coeffs, 1e-4)
    return coeffs, dcoeffs

@njit
def coeffs(phase, p, a, i, e, w, with_derivatives):
    p = atleast_1d(asarray(p))
    a = atleast_1d(asarray(a))
    i = atleast_1d(asarray(i))
    e = atleast_1d(asarray(e))
    w = atleast_1d(asarray(w))
    nor = p.size

    if with_derivatives:
        diffs = zeros((nor, 6, 6))
        for j in range(nor):
            diffs[j] = partial_derivatives()
    else:
        diffs = None

    return loc_and_der_coeffs(phase, p, a, i, e, w, diffs)