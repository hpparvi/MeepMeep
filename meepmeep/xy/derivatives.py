from numba import njit
from numpy import zeros, sqrt

from .position import solve_xy_p5s, xy_t15s, xy_t15sc


@njit
def xy_derivative_coeffs(v, eps, c0):
    cs = zeros((6, 2, 5))
    for i in range(6):
        cs[i] = solve_xy_p5s(v[i, 0], v[i, 1], v[i, 2], v[i, 3], v[i, 4], v[i, 5])
    return (cs - c0) / eps


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
def dpd(t, x, y, dcs):
    dx, dy = xy_t15sc(t, dcs)
    return (0.5/sqrt(x**2 + y**2))*(2*x*dx + 2*y*dy)


@njit(fastmath=True)
def pd_derivatives_s(t, x, y, dcf, res):
    res[0] = dpd(t, x, y, dcf[0])  # 0: Zero epoch
    res[1] = dpd(t, x, y, dcf[1])  # 1: Period
    res[2] = dpd(t, x, y, dcf[2])  # 2: Semi-major axis
    res[3] = dpd(t, x, y, dcf[3])  # 3: Inclination
    res[4] = dpd(t, x, y, dcf[4])  # 4: Eccentricity
    res[5] = dpd(t, x, y, dcf[5])  # 5: Argument of periastron
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
def pd_with_derivatives_v(t, t0, p, cf, dcf):
    npt = t.size
    res = zeros((7, npt))
    for i in range(npt):
        pd_with_derivatives_s(t[i], t0, p, cf, dcf, res[:, i])
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
