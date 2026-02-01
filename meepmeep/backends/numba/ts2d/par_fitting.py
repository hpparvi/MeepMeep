from math import sin, sqrt, pi, cos

from numba import njit
from numpy import zeros, arctan2, array, atleast_1d, asarray
from scipy.constants import G

from ..utils import as_from_rhop, i_from_baew
from .position import solve_xy_p5s
from .derivatives import xy_derivative_coeffs, loc_and_der_coeffs


@njit
def pds_diff(p, rho, b, secw, sesw, dd):
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
def partial_derivatives(p, rho, b, secw, sesw):
    v = zeros((6, 6))
    a = as_from_rhop(rho, p)
    e = secw ** 2 + sesw ** 2
    w = arctan2(sesw, secw)
    i = i_from_baew(b, a, e, w)

    v[0, 0] = -1

    # Period: p, a, and i
    da = da_dp(rho, p)
    di = di_da(b, a, e, w)
    v[1, 1] = 1
    v[1, 2] = da
    v[1, 3] = da * di

    # Stellar density: a and i
    da = da_drho(rho, p)
    di = di_da(b, a, e, w)
    v[2, 2] = da
    v[2, 3] = da * di

    # Impact parameter: i
    v[3, 3] = di_db(b, a, e, w)

    # secw and sesw
    # -------------
    die = di_de(b, a, e, w)
    diw = di_dw(b, a, e, w)

    # sqrt e cos w: i, e, and w
    de = de_dsecw(secw)
    dw = dw_dsecw(secw, sesw)
    v[4, 3] = de * die + dw * diw
    v[4, 4] = de
    v[4, 5] = dw

    # sqrt e sin w: i, e, and w
    de = de_dsesw(sesw)
    dw = dw_dsesw(secw, sesw)
    v[5, 3] = de * die + dw * diw
    v[5, 4] = de
    v[5, 5] = dw
    return v


@njit
def coeffs_old(phase, p, rho, b, secw, sesw):
    a = as_from_rhop(rho, p)
    e = secw ** 2 + sesw ** 2
    w = arctan2(sesw, secw)
    i = i_from_baew(b, a, e, w)

    coeffs = solve_xy_p5s(phase, p, a, i, e, w)
    pds = partial_derivatives(p, rho, b, secw, sesw)
    dcoeffs = xy_derivative_coeffs(0.0, p, a, i, e, w, pds, coeffs, 1e-4)
    return coeffs, dcoeffs

@njit
def coeffs(phase, p, rho, b, secw, sesw, with_derivatives):
    p = atleast_1d(asarray(p))
    rho = atleast_1d(asarray(rho))
    b = atleast_1d(asarray(b))
    secw = atleast_1d(asarray(secw))
    sesw = atleast_1d(asarray(sesw))
    nor = p.size

    a = as_from_rhop(rho, p)
    e = secw ** 2 + sesw ** 2
    w = arctan2(sesw, secw)
    i = i_from_baew(b, a, e, w)

    if with_derivatives:
        diffs = zeros((nor, 6, 6))
        for j in range(nor):
            diffs[j] = partial_derivatives(p[j], rho[j], b[j], secw[j], sesw[j])
    else:
        diffs = None

    return loc_and_der_coeffs(phase, p, a, i, e, w, diffs)

# Partial derivatives
# -------------------

@njit
def da_dp(rho, p):
    return (2 / 3) * (G * (1e3 * rho * 86400 ** 2) / (3 * pi * p)) ** (1 / 3)


@njit
def da_drho(rho, p):
    return (1 / 3) * ((G * 86400 ** 2 * p ** 2 * 1e3) / (3 * pi)) ** (1 / 3) * rho ** (-2 / 3)


@njit
def di_da(b, a, e, w):
    ea = (1.0 - e ** 2) / (1.0 + e * sin(w))
    return (b / (a ** 2 * ea)) / sqrt(1.0 - (b / (a * ea)) ** 2)


@njit
def di_db(b, a, e, w):
    ea = (1.0 - e ** 2) / (1.0 + e * sin(w))
    return -1.0 / (a * ea * sqrt(1.0 - (b / (a * ea)) ** 2))


@njit
def di_de(b, a, e, w):
    l = b / a
    m = (e ** 2 - 1) ** 2
    return -l * (e ** 2 * sin(w) + 2 * e + sin(w)) / (sqrt(-(l ** 2 * (e * sin(w) + 1) ** 2 - m) / m) * m)


@njit
def di_dw(b, a, e, w):
    l = b / a
    m = (e ** 2 - 1) ** 2
    return l * e * cos(w) / (sqrt((-l ** 2 * (e * sin(w) + 1) ** 2 + m) / m) * (e ** 2 - 1))


@njit
def de_dsecw(secw):
    return 2 * secw


@njit
def de_dsesw(sesw):
    return 2 * sesw


@njit
def dw_dsecw(secw, sesw):
    return -sesw / (secw ** 2 + sesw ** 2)


@njit
def dw_dsesw(secw, sesw):
    return secw / (secw ** 2 + sesw ** 2)