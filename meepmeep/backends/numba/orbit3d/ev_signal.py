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

"""Multi-knot ellipsoidal-variation signal evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, sin, sqrt, ndarray

from .position import _pos_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _ev_signal_os(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`ev_signal_o`. See that function for documentation."""
    sin2_inc = sin(inc) ** 2
    pre = -alpha * mass_ratio * sin2_inc
    x, y, z = _pos_os(t, tpa, p, dt, pktable, points, coeffs)
    d2 = x * x + y * y + z * z
    d = sqrt(d2)
    cz = z / d
    return pre * (2.0 * cz * cz - 1.0) / (d2 * d)


@njit(fastmath=True)
def _ev_signal_ov(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs):
    """Vector kernel for :func:`ev_signal_o`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    sin2_inc = sin(inc) ** 2
    pre = -alpha * mass_ratio * sin2_inc
    for i in range(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        out[i] = pre * (2.0 * cz * cz - 1.0) / (d2 * d)
    return out


@njit(fastmath=True, parallel=True)
def _ev_signal_ovp(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs):
    """Parallel (prange) twin of :func:`_ev_signal_ov`."""
    n = times.size
    out = zeros(n)
    sin2_inc = sin(inc) ** 2
    pre = -alpha * mass_ratio * sin2_inc
    for i in prange(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        out[i] = pre * (2.0 * cz * cz - 1.0) / (d2 * d)
    return out


def ev_signal_o(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs):
    """Ellipsoidal variation signal (Lillo-Box et al. 2014, Eqs. 6-10).

    Returns the relative flux variation induced by the tidally distorted
    primary as a function of the orbital phase. The amplitude scales
    with the mass ratio, the projected-area factor :math:`\\sin^2 i`,
    and the inverse cube of the instantaneous 3D separation.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_ev_signal_os`) or vector (:func:`_ev_signal_ov`) kernel
    at compile time (inside ``@njit``) or at call time (pure Python). Time
    argument is the 4th positional.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians].
    t : float or ndarray
        Time(s) at which to evaluate the signal.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    ev : float or ndarray
        Relative flux variation due to ellipsoidal distortion. Arrays of
        shape (N,) for an array time argument.

    Notes
    -----
    Uses the identity :math:`\\cos(2\\arccos u) = 2u^2 - 1` to skip a
    redundant arccos/cos pair.
    """
    if isinstance(t, ndarray):
        return _ev_signal_ov(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs)
    return _ev_signal_os(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs)


@overload(ev_signal_o, jit_options={'fastmath': True})
def _ev_signal_o_overload(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs):
            return _ev_signal_ov(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs):
            return _ev_signal_os(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
