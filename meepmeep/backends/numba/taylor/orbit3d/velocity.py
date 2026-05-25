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

"""Multi-knot planet (vx, vy, vz) velocity evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..velocity3d import vel_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _vel_os(t, tpa, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity at scalar time ``t`` for any orbital phase.

    Parameters
    ----------
    t : float
        Time at which to evaluate the velocity.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    vx, vy, vz : float
        Velocity components in :math:`R_\\star/\\mathrm{day}`. ``vx``,
        ``vy`` are the sky-plane components; ``vz`` is the line-of-sight
        component (positive toward the observer).
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return vel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _vel_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the velocity.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    vxs, vys, vzs : ndarray, shape (N,)
        Velocity component arrays in :math:`R_\\star/\\mathrm{day}`.
    """
    npt = times.size
    vxs, vys, vzs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        vxs[i], vys[i], vzs[i] = _vel_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return vxs, vys, vzs


def vel_o(t, tpa, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity. See :func:`_vel_os` / :func:`_vel_ov`."""
    if isinstance(t, ndarray):
        return _vel_ov(t, tpa, p, dt, pktable, points, coeffs)
    return _vel_os(t, tpa, p, dt, pktable, points, coeffs)


@overload(vel_o, jit_options={'fastmath': True})
def _vel_o_overload(t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _vel_ov(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _vel_os(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
