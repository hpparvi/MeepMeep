from numba import njit
from numpy import zeros, pi, floor

from .solve3d import solve3d
from ..utils import mean_anomaly_at_transit


@njit
def solve3d_orbit(knot_times, p, a, i, e, w, npt):
    coeffs = zeros((npt, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt-1):
        coeffs[ix, :, :] = solve3d(p*knot_times[ix] - to, p, a, i, e, w)
    coeffs[-1, : ,:] = coeffs[0]
    return coeffs


@njit(fastmath=True)
def knot_ix(t, t0, p, dt, pktable) -> int:
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    return pktable[int(floor(tc / (dt*p)))]
