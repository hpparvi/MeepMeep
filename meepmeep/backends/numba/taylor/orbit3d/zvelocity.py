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

"""Multi-knot planet z-velocity (line-of-sight) evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..velocity3d import zvel_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _zvel_os(t, tpa, p, dt, pktable, points, coeffs):
    """Planet z-velocity at scalar time ``t`` for any orbital phase.

    Cheaper than :func:`_vel_os` when only the line-of-sight component is
    needed (e.g. for radial velocity).

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-velocity.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    vz : float
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`].
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _zvel_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Planet z-velocity at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-velocity.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    vzs : ndarray, shape (N,)
        Line-of-sight velocities [:math:`R_\\star/\\mathrm{day}`].
    """
    npt = times.size
    vzs = zeros(npt)
    for i in range(npt):
        vzs[i] = _zvel_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return vzs


def zvel_o(t, tpa, p, dt, pktable, points, coeffs):
    """Planet z-velocity. See :func:`_zvel_os` / :func:`_zvel_ov`."""
    if isinstance(t, ndarray):
        return _zvel_ov(t, tpa, p, dt, pktable, points, coeffs)
    return _zvel_os(t, tpa, p, dt, pktable, points, coeffs)


@overload(zvel_o, jit_options={'fastmath': True})
def _zvel_o_overload(t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _zvel_ov(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _zvel_os(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
