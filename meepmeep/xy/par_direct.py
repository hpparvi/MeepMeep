from numba import njit
from numpy import zeros, array

from .position import solve_xy_p5s
from .derivatives import xy_derivative_coeffs


@njit
def diffs(p, a, i, e, w, dd: float = 1e-4):
    v = zeros((6, 6))
    v[:, :] = array([0.0, p, a, i, e, w])
    v[0, 0] -= dd
    for i in range(1, 6):
        v[i, i] += dd
    return v


@njit
def coeffs(phase, p, a, i, e, w):
    coeffs = solve_xy_p5s(phase, p, a, i, e, w)
    dcoeffs = xy_derivative_coeffs(diffs(p, a, i, e, w, 1e-4), 1e-4, coeffs)
    return coeffs, dcoeffs
