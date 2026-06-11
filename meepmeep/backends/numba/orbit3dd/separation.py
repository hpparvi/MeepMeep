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

"""Multi-knot sky-projected separation evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.separation import _sep_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _sep_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dd):
    """Write-into orbit kernel: epoch fold, knot lookup, and evaluation.

    Writes the seven-parameter gradient into the caller-provided ``(7,)``
    buffer ``dd`` and returns the separation; see
    :func:`~meepmeep.backends.numba.orbit3dd.position._pos_ow`.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return _sep_cd_w(tc - points[ix] * p, coeffs[ix], dcoeffs[ix], dd)


@njit(fastmath=True)
def _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`sep_od`. See that function for documentation."""
    dd = zeros(7)
    d = _sep_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dd)
    return d, dd


@njit(fastmath=True)
def _sep_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`sep_od`. See that function for documentation."""
    n = times.size
    ds = zeros(n)
    dds = zeros((n, 7))
    for j in range(n):
        ds[j] = _sep_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dds[j])
    return ds, dds


def sep_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected planet-star separation and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_sep_osd`) or vector (:func:`_sep_ovd`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Returns :math:`\\sqrt{x^2 + y^2}` together with its gradient w.r.t.
    the seven orbital parameters. The chain rule
    :math:`\\partial d/\\partial \\theta = (p_x \\partial p_x / \\partial \\theta + p_y \\partial p_y / \\partial \\theta)/d`
    is applied inside :func:`~meepmeep.backends.numba.point3dd.separation.sep_cd`;
    this dispatcher just locates the knot.

    Parameters
    ----------
    t : float or ndarray
        Time at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`~meepmeep.backends.numba.orbit3dd.position.pos_od`.

    Returns
    -------
    d : float or ndarray
        Sky-projected separation [stellar radii]. Arrays of shape (N,)
        for an array ``t``.
    dd : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a
        scalar ``t``, (N, 7) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _sep_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(sep_od, jit_options={'fastmath': True})
def _sep_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _sep_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
