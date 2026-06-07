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

from ..point3dd.separation import sep_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected planet-star separation and orbital-parameter derivatives at scalar time.

    Returns :math:`\\sqrt{x^2 + y^2}` together with its gradient w.r.t.
    the seven orbital parameters. The chain rule
    :math:`\\partial d/\\partial \\theta = (p_x \\partial p_x / \\partial \\theta + p_y \\partial p_y / \\partial \\theta)/d`
    is applied inside :func:`~meepmeep.backends.numba.taylor.point3dd.separation.sep_cd`;
    this dispatcher just locates the knot.

    Parameters
    ----------
    t : float
        Time at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    d : float
        Sky-projected separation [stellar radii].
    dd : ndarray, shape (7,)
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return sep_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _sep_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected planet-star separation and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    ds : ndarray, shape (N,)
        Sky-projected separations per time.
    dds : ndarray, shape (N, 7)
        Gradients w.r.t. ``(tc, p, a, i, e, w, lan)`` per time.
    """
    n = times.size
    ds = zeros(n)
    dds = zeros((n, 7))
    for j in range(n):
        d, dd = _sep_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        ds[j] = d
        for k in range(7):
            dds[j, k] = dd[k]
    return ds, dds


def sep_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected separation with gradients. See :func:`_sep_osd` / :func:`_sep_ovd`."""
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
