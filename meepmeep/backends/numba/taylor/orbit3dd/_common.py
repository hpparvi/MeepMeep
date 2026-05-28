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

"""Shared helpers for the multi-knot orbit3dd gradient evaluators.

Holds the per-quantity-agnostic infrastructure used across the orbit3dd
package: the Numba array-type predicate used by the ``*_od`` overloads and
the per-orbit Taylor/derivative coefficient builder. The ``*_osd`` kernels
fold the epoch inline, so there is no ``knot_ix`` here.
"""

from numba import njit, types
from numpy import zeros, pi

from ..solve3dd import solve3d_d
from ...utils import mean_anomaly_at_transit


def _is_1d_array(typ):
    """True for a 1-D Numba array type (any layout)."""
    return isinstance(typ, types.Array) and typ.ndim == 1


@njit
def solve3d_orbit_d(knot_times, p, a, i, e, w, lan=0.0, npt=15):
    """Pre-compute Taylor and derivative coefficients at every knot of one orbit.

    Derivative-returning counterpart of
    :func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit`. Calls
    :func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d` once per
    interior knot and stacks the resulting ``(3, 5)`` and ``(6, 3, 5)``
    matrices into per-orbit arrays. The last slot is the periodic image of
    the first and is copied rather than recomputed.

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
    lan : float, optional
        Longitude of the ascending node [radians]. Constant rotation of the
        sky-plane (x, y) coordinates about the line of sight. Defaults to 0.0.
    npt : int, optional
        Number of knots, including the periodic-image slot. Defaults to 15.

    Returns
    -------
    coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every knot (same layout as in
        :func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit`).
    dcoeffs : ndarray, shape (npt, 7, 3, 5)
        Parameter-derivative tensors at every knot. The second axis is
        ordered ``(t0, p, a, i, e, w, lan)``.

    Notes
    -----
    If you hand-roll ``knot_times`` you must enforce the periodic-image
    contract yourself; ``knots.create_knots`` does this automatically.
    """
    coeffs = zeros((npt, 3, 5))
    dcoeffs = zeros((npt, 7, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        cf, dcf = solve3d_d(p * knot_times[ix] - to, p, a, i, e, w, lan)
        coeffs[ix, :, :] = cf
        dcoeffs[ix, :, :, :] = dcf
    coeffs[-1, :, :] = coeffs[0]
    dcoeffs[-1, :, :, :] = dcoeffs[0]
    return coeffs, dcoeffs
