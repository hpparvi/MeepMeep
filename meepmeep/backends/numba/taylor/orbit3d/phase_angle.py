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

"""Multi-knot phase-angle (star-planet-observer) cosine evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _cos_alpha_os(t, tpa, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at scalar time ``t``.

    The phase angle :math:`\\alpha` is the star-planet-observer angle.
    With z positive toward the observer, :math:`\\cos\\alpha = -z/r` where
    :math:`r = \\sqrt{x^2 + y^2 + z^2}`. At superior conjunction (full
    phase, planet behind star) :math:`\\cos\\alpha = +1`; at inferior
    conjunction (new phase, planet in front) :math:`\\cos\\alpha = -1`.

    Parameters
    ----------
    t : float
        Time at which to evaluate the phase angle.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    cos_alpha : float
        Cosine of the phase angle, in :math:`[-1, 1]`.
    """
    x, y, z = _pos_os(t, tpa, p, dt, pktable, points, coeffs)
    return -z / sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def _cos_alpha_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at an array of times.

    See :func:`_cos_alpha_os` for the sign and reference conventions.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the phase angle.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    cos_alpha : ndarray, shape (N,)
        Cosine of the phase angle at each input time.
    """
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        out[i] = -z / sqrt(x * x + y * y + z * z)
    return out


def cos_alpha_o(t, tpa, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle. See :func:`_cos_alpha_os` / :func:`_cos_alpha_ov`."""
    if isinstance(t, ndarray):
        return _cos_alpha_ov(t, tpa, p, dt, pktable, points, coeffs)
    return _cos_alpha_os(t, tpa, p, dt, pktable, points, coeffs)


@overload(cos_alpha_o, jit_options={'fastmath': True})
def _cos_alpha_o_overload(t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _cos_alpha_ov(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _cos_alpha_os(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
