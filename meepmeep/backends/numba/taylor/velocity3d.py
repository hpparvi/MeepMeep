from numba import njit
from numpy import ndarray, floor


@njit(fastmath=True)
def v3dc(t: float, c: ndarray) -> tuple[float, float, float]:
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
def vz3dc(t, c):
    """Calculate planet's z-velocity for t centered on the expansion time."""
    return c[2, 1] + t * (2.0 * c[2, 2] + t * (3.0 * c[2, 3] + t * 4.0 * c[2, 4]))


@njit(fastmath=True)
def vz3d(tc, t0, p, c):
    """Calculate planet's z-velocity."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    return vz3dc(tc - (t0 + epoch * p), c)
