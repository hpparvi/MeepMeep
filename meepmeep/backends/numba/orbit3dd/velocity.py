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

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.velocity import vel_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives at scalar time.

    Parameters
    ----------
    t : float
        Time at which to evaluate the velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vx, vy, vz : float
        Velocity components in :math:`R_\\star/\\mathrm{day}`.
    dvx, dvy, dvz : ndarray, shape (7,)
        Gradients w.r.t. ``(tc, p, a, i, e, w, lan)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return vel_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _vel_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vxs, vys, vzs : ndarray, shape (N,)
        Velocity components per time.
    dvxs, dvys, dvzs : ndarray, shape (N, 7)
        Gradients w.r.t. ``(tc, p, a, i, e, w, lan)`` per time.
    """
    n = times.size
    vxs = zeros(n)
    vys = zeros(n)
    vzs = zeros(n)
    dvxs = zeros((n, 7))
    dvys = zeros((n, 7))
    dvzs = zeros((n, 7))
    for j in range(n):
        vx, vy, vz, dvx, dvy, dvz = _vel_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        vxs[j] = vx
        vys[j] = vy
        vzs[j] = vz
        for k in range(7):
            dvxs[j, k] = dvx[k]
            dvys[j, k] = dvy[k]
            dvzs[j, k] = dvz[k]
    return vxs, vys, vzs, dvxs, dvys, dvzs


def vel_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet velocity with gradients. See :func:`_vel_osd` / :func:`_vel_ovd`."""
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
