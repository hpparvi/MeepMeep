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

"""Multi-expansion-point ellipsoidal-variation signal evaluators.

Epoch folding and expansion-point lookup happen here; the signal itself is
delegated to the single-expansion-point
:func:`~meepmeep.backends.numba.point3d.ev_signal.ev_signal_c`.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.ev_signal import _ev_signal_c_s
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _ev_signal_os(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`ev_signal_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _ev_signal_c_s(tc - ep_times[ix] * p, alpha, mass_ratio, inc, coeffs[ix])


@njit(fastmath=True)
def ev_signal_ov(alpha, mass_ratio, inc, times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`ev_signal_o`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    for i in range(n):
        out[i] = _ev_signal_os(alpha, mass_ratio, inc, times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def ev_signal_ovp(alpha, mass_ratio, inc, times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`ev_signal_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _ev_signal_os(alpha, mass_ratio, inc, times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return out


def ev_signal_o(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Ellipsoidal variation signal (Lillo-Box et al. 2014, Eqs. 6-10).

    Returns the relative flux variation induced by the tidally distorted
    primary as a function of the orbital phase. The amplitude scales
    with the mass ratio, the projected-area factor :math:`\\sin^2 i`,
    and the inverse cube of the instantaneous 3D separation.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_ev_signal_os`) or vector (:func:`ev_signal_ov`) kernel
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
    tpa, p, dt, ep_table, ep_times, coeffs :
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
        return ev_signal_ov(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs)
    return _ev_signal_os(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(ev_signal_o, jit_options={'fastmath': True})
def _ev_signal_o_overload(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs):
            return ev_signal_ov(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _ev_signal_os(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
