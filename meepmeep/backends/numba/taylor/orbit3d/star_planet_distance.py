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

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _star_planet_distance_os(t, tpa, p, dt, pktable, points, coeffs):
    """3D star-planet distance at scalar time.

    Returns :math:`\\sqrt{x^2 + y^2 + z^2}`. Distinct from :func:`_sep_os`,
    which projects out the line-of-sight component.

    Parameters
    ----------
    t : float
        Time at which to evaluate the distance.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    r : float
        3D star-planet distance [stellar radii].
    """
    x, y, z = _pos_os(t, tpa, p, dt, pktable, points, coeffs)
    return sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def _star_planet_distance_ov(times, tpa, p, dt, pktable, points, coeffs):
    """3D star-planet distance at an array of times.

    Returns :math:`\\sqrt{x^2 + y^2 + z^2}`, the full Euclidean separation
    in 3D. Distinct from :func:`_sep_os`, which projects out the
    line-of-sight component.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the separation.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    r : ndarray, shape (N,)
        3D star-planet separation [stellar radii].
    """
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        out[i] = sqrt(x * x + y * y + z * z)
    return out


def star_planet_distance_o(t, tpa, p, dt, pktable, points, coeffs):
    """3D star-planet distance.

    See :func:`_star_planet_distance_os` / :func:`_star_planet_distance_ov`.
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
