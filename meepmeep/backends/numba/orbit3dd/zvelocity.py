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

"""Multi-knot planet z-velocity (line-of-sight) evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.zvelocity import zvel_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`zvel_od`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _zvel_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`zvel_od`. See that function for documentation."""
    n = times.size
    vzs = zeros(n)
    dvzs = zeros((n, 7))
    for j in range(n):
        vz, dvz = _zvel_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        vzs[j] = vz
        for k in range(7):
            dvzs[j, k] = dvz[k]
    return vzs, dvzs


def zvel_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_zvel_osd`) or vector (:func:`_zvel_ovd`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Cheaper than :func:`~meepmeep.backends.numba.orbit3dd.velocity.vel_od` when
    only the line-of-sight component is needed (e.g. for radial-velocity
    gradients).

    Parameters
    ----------
    t : float or ndarray
        Time at which to evaluate the z-velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`~meepmeep.backends.numba.orbit3dd.position.pos_od`.

    Returns
    -------
    vz : float or ndarray
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`]. Arrays of
        shape (N,) for an array ``t``.
    dvz : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a
        scalar ``t``, (N, 7) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _zvel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(zvel_od, jit_options={'fastmath': True})
def _zvel_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zvel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
