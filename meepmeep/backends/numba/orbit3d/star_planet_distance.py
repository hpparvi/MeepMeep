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

"""Multi-knot 3D star-planet distance evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _star_planet_distance_os(t, tpa, p, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`star_planet_distance_o`. See that function for documentation."""
    x, y, z = _pos_os(t, tpa, p, dt, pktable, points, coeffs)
    return sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def _star_planet_distance_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Vector kernel for :func:`star_planet_distance_o`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        out[i] = sqrt(x * x + y * y + z * z)
    return out


@njit(fastmath=True, parallel=True)
def _star_planet_distance_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel (prange) twin of :func:`_star_planet_distance_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _star_planet_distance_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


def star_planet_distance_o(t, tpa, p, dt, pktable, points, coeffs):
    """3D star-planet distance at an array of times.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_star_planet_distance_os`) or vector (:func:`_star_planet_distance_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Returns :math:`\\sqrt{x^2 + y^2 + z^2}`, the full Euclidean separation
    in 3D. Distinct from :func:`_sep_os`, which projects out the
    line-of-sight component.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the separation.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    r : float or ndarray
        3D star-planet separation [stellar radii]. Arrays of shape (N,) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _star_planet_distance_ov(t, tpa, p, dt, pktable, points, coeffs)
    return _star_planet_distance_os(t, tpa, p, dt, pktable, points, coeffs)


@overload(star_planet_distance_o, jit_options={'fastmath': True})
def _star_planet_distance_o_overload(t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _star_planet_distance_ov(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _star_planet_distance_os(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
