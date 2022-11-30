from numba import njit
from numpy import zeros, array, sqrt, floor

from .xy5 import solve_xy_p5s


@njit(fastmath=True)
def xy_t15as(tc, t0, p, cs):
    """Calculate planet's (x,y) position near transit."""
    x0, y0, vx, vy, ax, ay, jx, jy, sx, sy = cs
    xy = zeros((2, tc.size))
    epoch = floor((tc - t0 + 0.5*p)/p)
    t = tc - (t0 + epoch*p)
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    xy[0] = x0 + vx*t + 0.5*ax*t2 + jx*t3/6.0 + sx*t4/24.
    xy[1] = y0 + vy*t + 0.5*ay*t2 + jy*t3/6.0 + sy*t4/24.
    return xy


@njit
def xy_derivative_coeffs(p, a, i, e, w, c0):
    dd = 0.001
    cs = zeros((6, 10))
    cs[0] = array(solve_xy_p5s(dd, p, a, i, e, w))
    cs[1] = array(solve_xy_p5s(0.0, p + dd, a, i, e, w))
    cs[2] = array(solve_xy_p5s(0.0, p, a + dd, i, e, w))
    cs[3] = array(solve_xy_p5s(0.0, p, a, i - dd, e, w))
    cs[4] = array(solve_xy_p5s(0.0, p, a, i, e + dd, w))
    cs[5] = array(solve_xy_p5s(0.0, p, a, i, e, w + dd))
    return (cs - c0)/dd


# Position derivatives
# --------------------

@njit
def dxy_dtc(t, t0, p, dcs):
    return xy_t15as(t, t0, p, dcs[0])


@njit
def dxy_dp(t, t0, p, dcs):
    return xy_t15as(t, t0, p, dcs[1])


@njit
def dxy_da(t, t0, p, dcs):
    return xy_t15as(t, t0, p, dcs[2])


@njit
def dxy_di(t, t0, p, dcs):
    return xy_t15as(t, t0, p, dcs[3])


@njit
def dxy_de(t, t0, p, dcs):
    return xy_t15as(t, t0, p, dcs[4])


@njit
def dxy_dw(t, t0, p, dcs):
    return xy_t15as(t, t0, p, dcs[5])


# Projected distance derivatives
# ------------------------------
@njit
def dpd(xy, times, dcs):
    x, y = xy
    dx, dy = xy_t15as(times, 0.0, p, dcs)
    return (0.5/sqrt(x**2 + y**2))*(2*x*dx + 2*y*dy)


@njit
def pd_derivatives(times, p, a, i, e, w):
    c0 = array(solve_xy_p5s(0.0, p, a, i, e, w))
    xy = xy_t15as(times, 0.0, p, c0)
    dcs = xy_derivative_coeffs(p, a, i, e, w, c0)
    derivatives = zeros((6, times.size))
    derivatives[0] = dpd(xy, times, dcs[0])
    derivatives[1] = dpd(xy, times, dcs[1])
    derivatives[2] = dpd(xy, times, dcs[2])
    derivatives[3] = dpd(xy, times, dcs[3])
    derivatives[4] = dpd(xy, times, dcs[4])
    derivatives[5] = dpd(xy, times, dcs[5])
    return derivatives


@njit
def dpd_dtc(t, xy, dcs):
    return dpd(xy, t, dcs[0])


@njit
def dpd_dp(t, xy, dcs):
    return dpd(xy, t, dcs[1])


@njit
def dpd_da(t, xy, dcs):
    return dpd(xy, t, dcs[2])


@njit
def dpd_di(t, xy, dcs):
    return dpd(xy, t, dcs[3])


@njit
def dpd_de(t, xy, dcs):
    return dpd(xy, t, dcs[4])


@njit
def dpd_dw(t, xy, dcs):
    return dpd(xy, t, dcs[5])
