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

from ..velocity3dd import zvel_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives at scalar time.

    Cheaper than :func:`_vel_osd` when only the line-of-sight component is
    needed (e.g. for radial-velocity gradients).

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vz : float
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`].
    dvz : ndarray, shape (7,)
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _zvel_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vzs : ndarray, shape (N,)
        Line-of-sight velocities per time.
    dvzs : ndarray, shape (N, 7)
        Gradients w.r.t. ``(tc, p, a, i, e, w, lan)`` per time.
    """
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
    """Planet z-velocity with gradients. See :func:`_zvel_osd` / :func:`_zvel_ovd`."""
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
