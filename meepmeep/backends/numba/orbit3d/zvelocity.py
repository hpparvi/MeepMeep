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

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.zvelocity import zvel_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _zvel_os(t, tpa, p, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`zvel_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _zvel_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Vector kernel for :func:`zvel_o`. See that function for documentation."""
    npt = times.size
    vzs = zeros(npt)
    for i in range(npt):
        vzs[i] = _zvel_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return vzs


@njit(fastmath=True, parallel=True)
def _zvel_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel (prange) twin of :func:`_zvel_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _zvel_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


def zvel_o(t, tpa, p, dt, pktable, points, coeffs):
    """Planet z-velocity (line-of-sight component) for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_zvel_os`) or vector (:func:`_zvel_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Cheaper than :func:`vel_o` when only the line-of-sight component is
    needed (e.g. for radial velocity).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the z-velocity.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`pos_o`.

    Returns
    -------
    vz : float or ndarray
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`].
        Arrays of shape (N,) for an array ``t``.
    """
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
