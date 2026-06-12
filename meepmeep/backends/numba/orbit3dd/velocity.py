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

"""Multi-knot planet (vx, vy, vz) velocity evaluators with parameter derivatives."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.velocity import _vel_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _vel_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dvx, dvy, dvz):
    """Write-into orbit kernel: epoch fold, knot lookup, and evaluation.

    Writes the seven-parameter gradients into the caller-provided ``(7,)``
    buffers ``dvx``, ``dvy``, and ``dvz`` and returns the velocity values;
    see :func:`~meepmeep.backends.numba.orbit3dd.position._pos_ow`.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return _vel_cd_w(tc - points[ix] * p, coeffs[ix], dcoeffs[ix], dvx, dvy, dvz)


@njit(fastmath=True)
def _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`vel_od`. See that function for documentation."""
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    vx, vy, vz = _vel_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dvx, dvy, dvz)
    return vx, vy, vz, dvx, dvy, dvz


@njit(fastmath=True)
def _vel_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`vel_od`. See that function for documentation."""
    n = times.size
    vxs = zeros(n)
    vys = zeros(n)
    vzs = zeros(n)
    dvxs = zeros((n, 7))
    dvys = zeros((n, 7))
    dvzs = zeros((n, 7))
    for j in range(n):
        vxs[j], vys[j], vzs[j] = _vel_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                                         dvxs[j], dvys[j], dvzs[j])
    return vxs, vys, vzs, dvxs, dvys, dvzs


@njit(fastmath=True, parallel=True)
def _vel_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`_vel_ovd`."""
    n = times.size
    vxs, vys, vzs = zeros(n), zeros(n), zeros(n)
    dvxs, dvys, dvzs = zeros((n, 7)), zeros((n, 7)), zeros((n, 7))
    for j in prange(n):
        vxs[j], vys[j], vzs[j] = _vel_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                                         dvxs[j], dvys[j], dvzs[j])
    return vxs, vys, vzs, dvxs, dvys, dvzs


def vel_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_vel_osd`) or vector (:func:`_vel_ovd`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time at which to evaluate the velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`~meepmeep.backends.numba.orbit3dd.position.pos_od`.

    Returns
    -------
    vx, vy, vz : float or ndarray
        Velocity components in :math:`R_\\star/\\mathrm{day}`. Arrays of
        shape (N,) for an array ``t``.
    dvx, dvy, dvz : ndarray
        Gradients w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a
        scalar ``t``, (N, 7) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _vel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(vel_od, jit_options={'fastmath': True})
def _vel_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _vel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
