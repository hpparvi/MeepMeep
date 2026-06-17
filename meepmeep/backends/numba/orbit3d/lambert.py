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

"""Multi-expansion-point Lambertian phase-curve evaluators.

Holds the Lambertian reflected-light phase curve
(:func:`lambert_phase_curve_o`). Epoch folding and expansion-point lookup
happen here; the flux itself is delegated to the single-expansion-point
:func:`~meepmeep.backends.numba.point3d.lambert.lambert_phase_curve_c`. The
shared phase kernel (:func:`_lambert_kernel`) is re-exported from
``point3d.lambert`` for backward compatibility.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.lambert import _lambert_kernel, _lambert_phase_curve_c_s
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _lambert_phase_curve_os(time, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`lambert_phase_curve_o`. See that function for documentation."""
    epoch = floor((time - tpa) / p)
    tc = time - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _lambert_phase_curve_c_s(tc - ep_times[ix] * p, ag, a, k, coeffs[ix])


@njit(fastmath=True)
def lambert_phase_curve_ov(times, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`lambert_phase_curve_o`. See that function for documentation."""
    n = times.size
    res = zeros(n)
    for i in range(n):
        res[i] = _lambert_phase_curve_os(times[i], ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs)
    return res


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_ovp(times, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`lambert_phase_curve_ov`."""
    n = times.size
    res = zeros(n)
    for i in prange(n):
        res[i] = _lambert_phase_curve_os(times[i], ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs)
    return res


def lambert_phase_curve_o(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
    """Lambertian phase-curve flux contribution.

    Evaluates :math:`F(t) = (k/a)^2\\, A_g\\, f(\\alpha(t))` where
    :math:`f` is the Lambert kernel and :math:`\\alpha(t)` is the
    instantaneous phase angle. The result is the planet-to-star flux
    ratio of reflected light at full phase scaled by the phase function.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_lambert_phase_curve_os`) or vector
    (:func:`lambert_phase_curve_ov`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the flux contribution.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, ep_table, ep_times, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    flux : float or ndarray
        Reflected planet-to-star flux ratio. Arrays of shape (N,) for an
        array time argument.
    """
    if isinstance(t, ndarray):
        return lambert_phase_curve_ov(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs)
    return _lambert_phase_curve_os(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(lambert_phase_curve_o, jit_options={'fastmath': True})
def _lambert_phase_curve_o_overload(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
            return lambert_phase_curve_ov(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs):
            return _lambert_phase_curve_os(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
