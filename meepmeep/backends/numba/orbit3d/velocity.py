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

"""Multi-expansion-point planet (vx, vy, vz) velocity evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.velocity import vel_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _vel_os(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`vel_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return vel_c(tc - ep_times[ix] * p, coeffs[ix])


@njit(fastmath=True)
def vel_ov(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`vel_o`. See that function for documentation."""
    npt = times.size
    vxs, vys, vzs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        vxs[i], vys[i], vzs[i] = _vel_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return vxs, vys, vzs


@njit(fastmath=True, parallel=True)
def vel_ovp(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`vel_ov`."""
    n = times.size
    vxs, vys, vzs = zeros(n), zeros(n), zeros(n)
    for i in prange(n):
        vxs[i], vys[i], vzs[i] = _vel_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return vxs, vys, vzs


def vel_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Planet (vx, vy, vz) velocity for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_vel_os`) or vector (:func:`vel_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the velocity.
    tpa, p, dt, ep_table, ep_times, coeffs :
        See :func:`pos_o`.

    Returns
    -------
    vx, vy, vz : float or ndarray
        Velocity components in :math:`R_\\star/\\mathrm{day}`. ``vx``,
        ``vy`` are the sky-plane components; ``vz`` is the line-of-sight
        component (positive toward the observer). Arrays of shape (N,) for
        an array ``t``.
    """
    if isinstance(t, ndarray):
        return vel_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return _vel_os(t, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(vel_o, jit_options={'fastmath': True})
def _vel_o_overload(t, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return vel_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _vel_os(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
