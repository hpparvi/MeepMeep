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

"""Multi-knot planet z-position (line-of-sight) evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..position3dd import zpos_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives at scalar time.

    Cheaper than :func:`_pos_osd` when only the line-of-sight coordinate
    and its gradient are needed.

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-coordinate and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    pz : float
        Line-of-sight planet coordinate [stellar radii].
    dpz : ndarray, shape (7,)
        Gradient w.r.t. ``(t0, p, a, i, e, w, lan)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zpos_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _zpos_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-coordinate and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    zs : ndarray, shape (N,)
        Line-of-sight coordinates per time.
    dzs : ndarray, shape (N, 7)
        Gradients w.r.t. ``(t0, p, a, i, e, w, lan)`` per time.
    """
    n = times.size
    zs = zeros(n)
    dzs = zeros((n, 7))
    for j in range(n):
        z, dz = _zpos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        zs[j] = z
        for k in range(7):
            dzs[j, k] = dz[k]
    return zs, dzs


def zpos_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position with gradients. See :func:`_zpos_osd` / :func:`_zpos_ovd`."""
    if isinstance(t, ndarray):
        return _zpos_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(zpos_od, jit_options={'fastmath': True})
def _zpos_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zpos_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
