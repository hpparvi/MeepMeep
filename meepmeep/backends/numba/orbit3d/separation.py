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

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.separation import sep_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _sep_os(t, tpa, p, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`sep_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return sep_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _sep_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Vector kernel for :func:`sep_o`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    for j in range(n):
        out[j] = _sep_os(times[j], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def _sep_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel (prange) twin of :func:`_sep_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _sep_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


def sep_o(t, tpa, p, dt, pktable, points, coeffs):
    """Sky-projected planet-star separation for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_sep_os`) or vector (:func:`_sep_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Returns :math:`\\sqrt{x^2 + y^2}` in units of the stellar radius -
    the quantity transit light-curve models consume directly.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the separation.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`pos_o`.

    Returns
    -------
    sep : float or ndarray
        Sky-projected separation [stellar radii], always non-negative.
        Arrays of shape (N,) for an array ``t``.
    """
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
