from numba import njit
from numpy import cos, sin, zeros

from .utils import mean_anomaly, ta_from_ea_s, ta_from_ea_v, z_from_ta_s, z_from_ta_v


@njit
def ea_newton_s(t, t0, p, e, w):
    ma = mean_anomaly(t, t0, p, e, w)
    ea = ma
    err = 0.05
    k = 0
    while abs(err) > 1e-8 and k < 1000:
        err = ea - e*sin(ea) - ma
        ea = ea - err/(1.0-e*cos(ea))
        k += 1
    return ea


@njit
def ea_newton_v(t, t0, p, e, w):
    ea = zeros(t.size)
    for i in range(len(t)):
        ea[i] = ea_newton_s(t[i], t0, p, e, w)
    return ea


@njit
def ta_newton_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_newton_s(t, t0, p, e, w), e)


@njit
def ta_newton_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_newton_v(t, t0, p, e, w), e)


@njit(fastmath=True)
def xy_newton_v(time, t0, p, a, i, e, w):
    """Planet velocity and acceleration at mid-transit in [R_star / day]"""
    f = ta_newton_v(time, t0, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    return x, y


@njit(fastmath=True)
def xyz_newton_v(time, t0, p, a, i, e, w):
    """Planet velocity and acceleration at mid-transit in [R_star / day]"""
    f = ta_newton_v(time, t0, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    z =  r * sin(w + f) * sin(i)
    return x, y, z


@njit
def z_newton_s(time, t0, p, a, i, e, w):
    """Normalized projected distance for scalar time.
    """
    ta = ta_newton_s(time, t0, p, e, w)
    return z_from_ta_s(ta, a, i, e, w)


@njit
def z_newton_v(time, t0, p, a, i, e, w):
    """Normalized projected distance for an array of times.
    """
    ta = ta_newton_v(time, t0, p, e, w)
    return z_from_ta_v(ta, a, i, e, w)


@njit
def rv_newton_v(times, k, t0, p, e, w):
    ta_n = ta_newton_v(times, t0, p, e, w)
    return k * (cos(w + ta_n) + e * cos(w))
