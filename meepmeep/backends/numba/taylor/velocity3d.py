from numba import njit
from numpy import ndarray, floor, pi, sqrt, sin, zeros
from numpy.typing import NDArray

from meepmeep.backends.numba.taylor.orbit3d import knot_ix


@njit(fastmath=True)
def v3dc(t: float | NDArray, c: NDArray) -> tuple[float, float, float] | tuple[NDArray, NDArray, NDArray]:
    """Calculate planet's (vx, vy, vz) velocity for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 3x5 coefficient matrix.

    Returns
    -------
    (float, float, float)
        The (vx, vy, vz) velocity.
    """
    vx = c[0, 1] + t * (2.0 * c[0, 2] + t * (3.0 * c[0, 3] + t * 4.0 * c[0, 4]))
    vy = c[1, 1] + t * (2.0 * c[1, 2] + t * (3.0 * c[1, 3] + t * 4.0 * c[1, 4]))
    vz = c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))
    return vx, vy, vz


@njit(fastmath=True)
def vzc(t, c):
    """Calculate planet's z-velocity for t centered on the expansion time."""
    return c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))


@njit(fastmath=True)
def vz(t, t0, p, c):
    """Calculate planet's z-velocity."""
    epoch = floor((t - t0 + 0.5 * p) / p)
    return vzc(t - (t0 + epoch * p), c)


@njit
def rvc(t, k, p, a, i, e, c):
    """Calculate radial velocity induced by the planet."""
    n = 2 * pi / p * (a * sin(i)) / sqrt(1 - e ** 2)  # Perryman (2018) Eq. 2.23
    return vzc(t, c) / n * k


@njit
def rv(t, k, t0, p, a, i, e, c):
    """Calculate radial velocity induced by the planet."""
    n = 2 * pi / p * (a * sin(i)) / sqrt(1 - e ** 2)
    return vz(t, t0, p, c) / n * k


@njit
def rvo(times, k, t0, p, a, i, e, dt, pktable, points, coeffs):
    """Calculate radial velocity induced by the planet."""
    npt = times.size
    rvs = zeros(npt)
    n = 2*pi/p * (a*sin(i))/sqrt(1-e**2)
    for i in range(npt):
        ix = knot_ix(times[i], t0, p, dt, pktable)
        t0 -= points[ix] * p
        rvs[i] = vz(times[i], t0, p, coeffs[ix]) / n * k
    return rvs


