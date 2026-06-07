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

"""Multi-knot sky-projected planet-star separation evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.separation import sep_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _sep_os(t, tpa, p, dt, pktable, points, coeffs):
    """Sky-projected planet-star separation at scalar time ``t``.

    Returns :math:`\\sqrt{x^2 + y^2}` in units of the stellar radius —
    the quantity transit light-curve models consume directly.

    Parameters
    ----------
    t : float
        Time at which to evaluate the separation.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    sep : float
        Sky-projected separation [stellar radii], always non-negative.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return sep_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _sep_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Sky-projected planet-star separation at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the separation.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    seps : ndarray, shape (N,)
        Sky-projected separations [stellar radii], always non-negative.
    """
    n = times.size
    out = zeros(n)
    for j in range(n):
        out[j] = _sep_os(times[j], tpa, p, dt, pktable, points, coeffs)
    return out


def sep_o(t, tpa, p, dt, pktable, points, coeffs):
    """Sky-projected separation. See :func:`_sep_os` / :func:`_sep_ov`."""
    if isinstance(t, ndarray):
        return _sep_ov(t, tpa, p, dt, pktable, points, coeffs)
    return _sep_os(t, tpa, p, dt, pktable, points, coeffs)


@overload(sep_o, jit_options={'fastmath': True})
def _sep_o_overload(t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _sep_ov(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _sep_os(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
