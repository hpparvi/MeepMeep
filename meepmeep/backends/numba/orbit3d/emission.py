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

"""Multi-expansion-point cosine emission phase-curve evaluators.

Epoch folding and expansion-point lookup happen here; the flux itself is
delegated to the single-expansion-point
:func:`~meepmeep.backends.numba.point3d.emission.emission_phase_curve_c`.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.emission import _emission_phase_curve_c_s
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _emission_phase_curve_os(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`emission_phase_curve_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _emission_phase_curve_c_s(tc - ep_times[ix] * p, k, fratio, offset, coeffs[ix])


@njit(fastmath=True)
def emission_phase_curve_ov(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`emission_phase_curve_o`. See that function for documentation."""
    n = t.size
    res = zeros(n)
    for i in range(n):
        res[i] = _emission_phase_curve_os(t[i], k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs)
    return res


@njit(fastmath=True, parallel=True)
def emission_phase_curve_ovp(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`emission_phase_curve_ov`."""
    n = t.size
    res = zeros(n)
    for i in prange(n):
        res[i] = _emission_phase_curve_os(t[i], k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs)
    return res


def emission_phase_curve_o(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
    """Cosine emission phase-curve flux contribution.

    Evaluates a simple cosine thermal-emission model
    :math:`F(t) = k^2\\,f_\\mathrm{ratio}\\,(1 + \\cos\\delta\\,c_z(t) +
    \\sin\\delta\\,s(t))/2`, where :math:`c_z = -z/d` is the cosine of the
    phase angle, :math:`s` the signed in-plane component from the orbital
    normal, and :math:`\\delta` the hotspot offset. The flux peaks at
    :math:`k^2 f_\\mathrm{ratio}` when the hotspot faces the observer.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_emission_phase_curve_os`) or vector
    (:func:`emission_phase_curve_ov`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the flux contribution.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    fratio : float
        Dayside-to-nightside per-surface-element flux ratio (amplitude
        scaling); the peak-to-peak swing is :math:`k^2 f_\\mathrm{ratio}`.
    offset : float
        Hotspot offset [radians].
    tpa, p, dt, ep_table, ep_times, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    flux : float or ndarray
        Emitted planet-to-star flux ratio. Arrays of shape (N,) for an array
        time argument.
    """
    if isinstance(t, ndarray):
        return emission_phase_curve_ov(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs)
    return _emission_phase_curve_os(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(emission_phase_curve_o, jit_options={'fastmath': True})
def _emission_phase_curve_o_overload(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
            return emission_phase_curve_ov(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
            return _emission_phase_curve_os(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
