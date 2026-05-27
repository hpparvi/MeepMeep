#  MeepMeep: fast orbit calculations for exoplanet modelling
#  Copyright (C) 2022-2026 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for the multi-knot orbit3d evaluators.

Holds the per-quantity-agnostic infrastructure used across the orbit3d
package: the Numba array-type predicate used by the ``*_o`` overloads,
the per-orbit coefficient builder, and the time-to-knot lookup.
"""

from numba import njit, types
from numpy import zeros, pi, floor

from ..solve3d import solve3d
from ...utils import mean_anomaly_at_transit


def _is_1d_array(typ):
    """True for a 1-D Numba array type (any layout)."""
    return isinstance(typ, types.Array) and typ.ndim == 1


@njit
def solve3d_orbit(knot_times, p, a, i, e, w, npt, lan=0.0):
    """Pre-compute Taylor coefficients at every knot of one orbit.

    For each interior knot this calls :func:`~meepmeep.backends.numba.taylor.solve3d.solve3d`
    once and stacks the resulting ``(3, 5)`` matrices into a single
    ``(npt, 3, 5)`` array. The last slot is the periodic image of the
    first and is copied rather than recomputed.

    Parameters
    ----------
    knot_times : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]``, with ``knot_times[-1]``
        equal to ``knot_times[0] + 1``. Built by
        :func:`~meepmeep.backends.numba.knots.create_knots`.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity, :math:`0 \\le e < 1`.
    w : float
        Argument of periastron [radians].
    npt : int
        Number of knots, including the periodic-image slot.
    lan : float, optional
        Longitude of the ascending node [radians]. Constant rotation of the
        sky-plane (x, y) coordinates about the line of sight. Defaults to 0.0.

    Returns
    -------
    coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every knot. Each ``coeffs[ix]`` is a
        ``(3, 5)`` matrix in the layout produced by ``solve3d`` (rows: x, y,
        z; columns: position, velocity, acceleration, jerk, snap; pre-scaled
        by factorial of the Taylor order).

    Notes
    -----
    If you hand-roll ``knot_times`` you must enforce the periodic-image
    contract yourself; ``knots.create_knots`` does this automatically.
    """
    coeffs = zeros((npt, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        coeffs[ix, :, :] = solve3d(p * knot_times[ix] - to, p, a, i, e, w, lan)
    coeffs[-1, :, :] = coeffs[0]
    return coeffs


@njit(fastmath=True, inline="always")
def knot_ix(t, tpa, p, dt, pktable) -> int:
    """Return the knot index for a single time.

    Epoch-folds ``t`` into one period and dispatches it to the appropriate
    knot via ``pktable``.

    Parameters
    ----------
    t : float
        Time at which to look up the knot.
    tpa : float
        Periastron time anchoring the knot grid. Note the convention
        difference: the high-level :class:`~meepmeep.orbit.Orbit` API
        takes the transit-center time as ``t0`` and converts it to
        periastron time before calling functions in this module (see
        ``Orbit.__init__``).
    p : float
        Orbital period [days].
    dt : float
        Width of one ``pktable`` bucket in fraction of the period.
    pktable : ndarray of int
        Time-to-knot lookup table built by
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    ix : int
        Index into ``coeffs`` / ``points`` for the relevant knot.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    return pktable[int(floor(tc / (dt * p)))]
