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

"""Multi-expansion-point planet z-position (line-of-sight) evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.zposition import zpos_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _zpos_os(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`zpos_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return zpos_c(tc - ep_times[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _zpos_ov(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`zpos_o`. See that function for documentation."""
    npt = times.size
    zs = zeros(npt)
    for i in range(npt):
        zs[i] = _zpos_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return zs


@njit(fastmath=True, parallel=True)
def _zpos_ovp(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`_zpos_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _zpos_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return out


def zpos_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Planet z-position (line-of-sight coordinate) for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_zpos_os`) or vector (:func:`_zpos_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Cheaper than :func:`pos_o` when only the line-of-sight coordinate is
    needed (e.g. for light travel time and eclipse-side geometry).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the z-coordinate.
    tpa, p, dt, ep_table, ep_times, coeffs :
        See :func:`pos_o`.

    Returns
    -------
    z : float or ndarray
        Line-of-sight planet coordinate [stellar radii], positive toward
        the observer. Arrays of shape (N,) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _zpos_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return _zpos_os(t, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(zpos_o, jit_options={'fastmath': True})
def _zpos_o_overload(t, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _zpos_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _zpos_os(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
