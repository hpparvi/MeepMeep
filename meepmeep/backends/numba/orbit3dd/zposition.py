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

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.zposition import _zpos_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _zpos_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dz):
    """Write-into orbit kernel: epoch fold, knot lookup, and evaluation.

    Writes the seven-parameter gradient into the caller-provided ``(7,)``
    buffer ``dz`` and returns the z position; see
    :func:`~meepmeep.backends.numba.orbit3dd.position._pos_ow`.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return _zpos_cd_w(tc - points[ix] * p, coeffs[ix], dcoeffs[ix], dz)


@njit(fastmath=True)
def _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`zpos_od`. See that function for documentation."""
    dz = zeros(7)
    z = _zpos_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dz)
    return z, dz


@njit(fastmath=True)
def _zpos_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`zpos_od`. See that function for documentation."""
    n = times.size
    zs = zeros(n)
    dzs = zeros((n, 7))
    for j in range(n):
        zs[j] = _zpos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dzs[j])
    return zs, dzs


@njit(fastmath=True, parallel=True)
def _zpos_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`_zpos_ovd`."""
    n = times.size
    zs = zeros(n)
    dzs = zeros((n, 7))
    for j in prange(n):
        zs[j] = _zpos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dzs[j])
    return zs, dzs


def zpos_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_zpos_osd`) or vector (:func:`_zpos_ovd`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Cheaper than :func:`~meepmeep.backends.numba.orbit3dd.position.pos_od` when
    only the line-of-sight coordinate and its gradient are needed.

    Parameters
    ----------
    t : float or ndarray
        Time at which to evaluate the z-coordinate and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`~meepmeep.backends.numba.orbit3dd.position.pos_od`.

    Returns
    -------
    pz : float or ndarray
        Line-of-sight planet coordinate [stellar radii]. Arrays of shape
        (N,) for an array ``t``.
    dpz : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a
        scalar ``t``, (N, 7) for an array ``t``.
    """
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
