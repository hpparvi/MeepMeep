from math import isfinite

from numba import njit
from numpy import zeros, sqrt, nan

from .position import solve_xy_p5s, xy_t15s, xy_t15sc


@njit
def xy_derivative_coeffs(phase, p, a, i, e, w, pds, c0, eps):
    cs = zeros((6, 2, 5))
    for j in range(6):
        v = pds[j]*eps
        cs[j, :, :] = solve_xy_p5s(phase+v[0], p+v[1], a+v[2], i+v[3], e+v[4], w+v[5])
    return (cs - c0) / eps


@njit
def loc_and_der_coeffs(phase, p, a, i, e, w, diffs):
    """Expands a planet's (x,y) location into a Taylor series.

    Expands a planet's (x,y) location into a Taylor series and optionally also expands the
    location's partial derivatives.
    """
    nor = p.size

    # ---------------------------------------------------------------------
    # Solve the Taylor series coefficients for the planet's (x, y) location
    # and its derivatives if they're requested.
    # ---------------------------------------------------------------------
    cfs = zeros((nor, 2, 5))
    for j in range(nor):
        if a[j] > 1.0 and e[j] < 0.99:
            cfs[j, :, :] = solve_xy_p5s(phase, p[j], a[j], i[j], e[j], w[j])
        else:
            cfs[j, :, :] = nan

    if diffs is not None:
        dcfs = zeros((nor, 6, 2, 5))
        for j in range(nor):
            if isfinite(cfs[j, 0, 0]):
                dcfs[j, :, :, :] = xy_derivative_coeffs(phase, p[j], a[j], i[j], e[j], w[j],
                                                        diffs[j], cfs[j], 1e-4)
            else:
                dcfs[j, :, :, :] = nan
    else:
        dcfs = zeros((0, 0, 0, 0))
    return cfs, dcfs

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
def pd_with_derivatives_s(t, cf, dcf, res):
    x, y = xy_t15sc(t, cf)
    res[0] = sqrt(x**2 + y**2)        # 0: Projected distance [R_Sun]
    res[1] = dpd(t, x, y, dcf[0])     # 1: Zero epoch
    res[2] = dpd(t, x, y, dcf[1])     # 2: Period
    res[3] = dpd(t, x, y, dcf[2])     # 3: Semi-major axis
    res[4] = dpd(t, x, y, dcf[3])     # 4: Inclination
    res[5] = dpd(t, x, y, dcf[4])     # 5: Eccentricity
    res[6] = dpd(t, x, y, dcf[5])  # 6: Argument of periastron
    return res


@njit
def pd_with_derivatives_v(t, cf, dcf):
    npt = t.size
    res = zeros((7, npt))
    for i in range(npt):
        pd_with_derivatives_s(t[i], cf, dcf, res[:, i])
    return res


@njit
def dpd_dtc(t, x, y, dcs):
    return dpd(t, x, y, dcs[0])


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
