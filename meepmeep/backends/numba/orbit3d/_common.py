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

"""Shared helpers for the multi-expansion-point orbit3d evaluators.

Holds the per-quantity-agnostic infrastructure used across the orbit3d
package: the Numba array-type predicate used by the ``*_o`` overloads,
the per-orbit coefficient builder, and the time-to-expansion-point lookup.
"""

from numba import njit, types
from numpy import zeros, pi, floor

from ..point3d.solve import solve3d
from ..utils import mean_anomaly_at_transit


def _is_1d_array(typ):
    """True for a 1-D Numba array type (any layout)."""
    return isinstance(typ, types.Array) and typ.ndim == 1


@njit
def solve3d_orbit(ep_times, p, a, i, e, w, lan=0.0, npt=15):
    """Pre-compute Taylor coefficients at every expansion point of one orbit.

    For each interior expansion point this calls :func:`~meepmeep.backends.numba.point3d.solve.solve3d`
    once and stacks the resulting ``(3, 5)`` matrices into a single
    ``(npt, 3, 5)`` array. The last slot is the periodic image of the
    first and is copied rather than recomputed.

    Parameters
    ----------
    ep_times : ndarray, shape (npt,)
        Normalised expansion-point phases in ``[0, 1]``, with ``ep_times[-1]``
        equal to ``ep_times[0] + 1``. Built by
        :func:`~meepmeep.backends.numba.expansion_points.create_expansion_points`.
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
    lan : float, optional
        Longitude of the ascending node [radians]. Constant rotation of the
        sky-plane (x, y) coordinates about the line of sight. Defaults to 0.0.
    npt : int, optional
        Number of expansion points, including the periodic-image slot. Defaults to 15.

    Returns
    -------
    coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every expansion point. Each ``coeffs[ix]`` is a
        ``(3, 5)`` matrix in the layout produced by ``solve3d`` (rows: x, y,
        z; columns: position, velocity, acceleration, jerk, snap; pre-scaled
        by factorial of the Taylor order).

    Notes
    -----
    If you hand-roll ``ep_times`` you must enforce the periodic-image
    contract yourself; ``expansion points.create_expansion_points`` does this automatically.
    """
    coeffs = zeros((npt, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        coeffs[ix, :, :] = solve3d(p * ep_times[ix] - to, p, a, i, e, w, lan)
    coeffs[-1, :, :] = coeffs[0]
    return coeffs


@njit(fastmath=True, inline="always")
def ep_ix(t, tpa, p, dt, ep_table) -> int:
    """Return the expansion-point index for a single time.

    Epoch-folds ``t`` into one period and dispatches it to the appropriate
    expansion point via ``ep_table``.

    Parameters
    ----------
    t : float
        Time at which to look up the expansion point.
    tpa : float
        Periastron time anchoring the expansion-point grid. Note the convention
        difference: the high-level :class:`~meepmeep.orbit.Orbit` API
        takes the transit-center time as ``tc`` and converts it to
        periastron time before calling functions in this module (see
        ``Orbit.__init__``).
    p : float
        Orbital period [days].
    dt : float
        Width of one ``ep_table`` bucket in fraction of the period.
    ep_table : ndarray of int
        Time-to-expansion-point lookup table built by
        :func:`~meepmeep.backends.numba.expansion_points.create_expansion_points`.

    Returns
    -------
    ix : int
        Index into ``coeffs`` / ``ep_times`` for the relevant expansion point.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    return ep_table[int(floor(tc / (dt * p)))]
