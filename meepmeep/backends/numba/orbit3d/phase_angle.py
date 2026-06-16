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

"""Multi-expansion-point phase-angle (star-planet-observer) cosine evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _cos_alpha_os(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`cos_alpha_o`. See that function for documentation."""
    x, y, z = _pos_os(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return -z / sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def _cos_alpha_ov(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`cos_alpha_o`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
        out[i] = -z / sqrt(x * x + y * y + z * z)
    return out


@njit(fastmath=True, parallel=True)
def _cos_alpha_ovp(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`_cos_alpha_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _cos_alpha_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return out


def cos_alpha_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Cosine of the phase angle for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_cos_alpha_os`) or vector (:func:`_cos_alpha_ov`) kernel at
    compile time (inside ``@njit``) or at call time (pure Python).

    The phase angle :math:`\\alpha` is the star-planet-observer angle.
    With z positive toward the observer, :math:`\\cos\\alpha = -z/r` where
    :math:`r = \\sqrt{x^2 + y^2 + z^2}`. At superior conjunction (full
    phase, planet behind star) :math:`\\cos\\alpha = +1`; at inferior
    conjunction (new phase, planet in front) :math:`\\cos\\alpha = -1`.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the phase angle.
    tpa, p, dt, ep_table, ep_times, coeffs :
        See :func:`pos_o`.

    Returns
    -------
    cos_alpha : float or ndarray
        Cosine of the phase angle, in :math:`[-1, 1]`.
        Arrays of shape (N,) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _cos_alpha_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return _cos_alpha_os(t, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(cos_alpha_o, jit_options={'fastmath': True})
def _cos_alpha_o_overload(t, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _cos_alpha_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _cos_alpha_os(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
