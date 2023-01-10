from numba import njit
from numpy import zeros, sqrt

from .xy5 import solve_xy_p5s, xy_t15s


@njit
def xy_derivative_coeffs(p, a, i, e, w, c0):
    dd = 0.001
    cs = zeros((6, 2, 5))
    cs[0] = solve_xy_p5s(dd, p, a, i, e, w)
    cs[1] = solve_xy_p5s(0.0, p + dd, a, i, e, w)
    cs[2] = solve_xy_p5s(0.0, p, a + dd, i, e, w)
    cs[3] = solve_xy_p5s(0.0, p, a, i - dd, e, w)
    cs[4] = solve_xy_p5s(0.0, p, a, i, e + dd, w)
    cs[5] = solve_xy_p5s(0.0, p, a, i, e, w + dd)
    return (cs - c0)/dd


# Position derivatives
# --------------------

@njit
def dxy_dtc(t, t0, p, dcs):
    return xy_t15s(t, t0, p, dcs[0])


@njit
def dxy_dp(t, t0, p, dcs):
    return xy_t15s(t, t0, p, dcs[1])


@njit
def dxy_da(t, t0, p, dcs):
    return xy_t15s(t, t0, p, dcs[2])


@njit
def dxy_di(t, t0, p, dcs):
    return xy_t15s(t, t0, p, dcs[3])


@njit
def dxy_de(t, t0, p, dcs):
    return xy_t15s(t, t0, p, dcs[4])


@njit
def dxy_dw(t, t0, p, dcs):
    return xy_t15s(t, t0, p, dcs[5])


# Projected distance derivatives
# ------------------------------
@njit(fastmath=True)
def dpd(t, t0, p, x, y, dcs):
    dx, dy = xy_t15s(t, t0, p, dcs)
    return (0.5/sqrt(x**2 + y**2))*(2*x*dx + 2*y*dy)


@njit(fastmath=True)
def pd_derivatives_s(t, t0, p, x, y, dcf, res):
    res[0] = dpd(t, t0, p, x, y, dcf[0])  # 0: Zero epoch
    res[1] = dpd(t, t0, p, x, y, dcf[1])  # 1: Period
    res[2] = dpd(t, t0, p, x, y, dcf[2])  # 2: Semi-major axis
    res[3] = dpd(t, t0, p, x, y, dcf[3])  # 3: Inclination
    res[4] = dpd(t, t0, p, x, y, dcf[4])  # 4: Eccentricity
    res[5] = dpd(t, t0, p, x, y, dcf[5])  # 5: Argument of periastron
    return res


@njit(fastmath=True)
def pd_with_derivatives_s(t, t0, p, cf, dcf, res):
    x, y = xy_t15s(t, t0, p, cf)
    res[0] = sqrt(x**2 + y**2)            # 0: Projected distance [R_Sun]
    res[1] = dpd(t, t0, p, x, y, dcf[0])  # 1: Zero epoch
    res[2] = dpd(t, t0, p, x, y, dcf[1])  # 2: Period
    res[3] = dpd(t, t0, p, x, y, dcf[2])  # 3: Semi-major axis
    res[4] = dpd(t, t0, p, x, y, dcf[3])  # 4: Inclination
    res[5] = dpd(t, t0, p, x, y, dcf[4])  # 5: Eccentricity
    res[6] = dpd(t, t0, p, x, y, dcf[5])  # 6: Argument of periastron
    return res


@njit
def dpd_dtc(t, t0, p, x, y, dcs):
    return dpd(t, t0, p, x, y, dcs[0])


@njit
def dpd_dp(t, p, xy, dcs):
    return dpd(p, xy, t, dcs[1])


@njit
def dpd_da(t, p, xy, dcs):
    return dpd(p, xy, t, dcs[2])


@njit
def dpd_di(t, p, xy, dcs):
    return dpd(p, xy, t, dcs[3])


@njit
def dpd_de(t, p, xy, dcs):
    return dpd(p, xy, t, dcs[4])


@njit
def dpd_dw(t, p, xy, dcs):
    return dpd(p, xy, t, dcs[5])
