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

"""Multi-knot Lambertian phase-curve and emission evaluators.

Holds the two reflected-light quantities that share the Lambertian phase
kernel: the pure phase curve (:func:`lambert_phase_curve_o`) and the
combined reflection-plus-emission model (:func:`lambert_and_emission_o`).
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, pi, sqrt, cos, arccos, ndarray

from .phase_angle import _cos_alpha_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _lambert_kernel(cos_alpha):
    """Lambertian phase function evaluated at a cosine of the phase angle.

    Computes :math:`f(\\alpha) = (\\sin\\alpha + (\\pi - \\alpha)\\cos\\alpha)/\\pi`,
    the disk-integrated reflectance of a Lambertian sphere. The
    implementation substitutes :math:`\\sin\\alpha = \\sqrt{1 - \\cos^2\\alpha}`
    to skip one trig call, and clamps ``cos_alpha`` to ``[-1, 1]`` so a
    Taylor-rounding overshoot cannot produce a NaN from :func:`arccos`.

    Parameters
    ----------
    cos_alpha : float
        Cosine of the phase angle.

    Returns
    -------
    phase : float
        Value of the Lambert kernel, in :math:`[0, 1]`.
    alpha : float
        Phase angle :math:`\\arccos(\\text{cos\\_alpha})` [radians],
        returned as a by-product so callers that also need
        :math:`\\alpha` avoid a second :func:`arccos`.
    """
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
    alpha = arccos(cos_alpha)
    return (sin_alpha + (pi - alpha) * cos_alpha) / pi, alpha


@njit(fastmath=True, inline="always")
def _lambert_phase_curve_os(time, ag, a, k, tpa, p, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`lambert_phase_curve_o`. See that function for documentation."""
    amplitude = k * k * ag / (a * a)
    cos_alpha = _cos_alpha_os(time, tpa, p, dt, pktable, points, coeffs)
    phase, _ = _lambert_kernel(cos_alpha)
    return amplitude * phase


@njit(fastmath=True)
def _lambert_phase_curve_ov(times, ag, a, k, tpa, p, dt, pktable, points, coeffs):
    """Vector kernel for :func:`lambert_phase_curve_o`. See that function for documentation."""
    n = times.size
    res = zeros(n)
    amplitude = k * k * ag / (a * a)
    for i in range(n):
        cos_alpha = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
        phase, _ = _lambert_kernel(cos_alpha)
        res[i] = amplitude * phase
    return res


@njit(fastmath=True, parallel=True)
def _lambert_phase_curve_ovp(times, ag, a, k, tpa, p, dt, pktable, points, coeffs):
    """Parallel (prange) twin of :func:`_lambert_phase_curve_ov`."""
    n = times.size
    res = zeros(n)
    amplitude = k * k * ag / (a * a)
    for i in prange(n):
        cos_alpha = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
        phase, _ = _lambert_kernel(cos_alpha)
        res[i] = amplitude * phase
    return res


@njit(fastmath=True, inline="always")
def _lambert_and_emission_os(t, ag, fr_night, fr_day, emi_offset, a, k,
                             tpa, p, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`lambert_and_emission_o`. See that function for documentation."""
    k2 = k * k
    aref = k2 * ag / (a * a)
    cos_alpha = _cos_alpha_os(t, tpa, p, dt, pktable, points, coeffs)
    phase, alpha = _lambert_kernel(cos_alpha)
    ref = aref * phase
    emi = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


@njit(fastmath=True)
def _lambert_and_emission_ov(times, ag, fr_night, fr_day, emi_offset, a, k,
                            tpa, p, dt, pktable, points, coeffs):
    """Vector kernel for :func:`lambert_and_emission_o`. See that function for documentation."""
    n = times.size
    ref, emi = zeros(n), zeros(n)
    k2 = k * k
    aref = k2 * ag / (a * a)
    for i in range(n):
        cos_alpha = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
        phase, alpha = _lambert_kernel(cos_alpha)
        ref[i] = aref * phase
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


@njit(fastmath=True, parallel=True)
def _lambert_and_emission_ovp(times, ag, fr_night, fr_day, emi_offset, a, k,
                              tpa, p, dt, pktable, points, coeffs):
    """Parallel (prange) twin of :func:`_lambert_and_emission_ov`."""
    n = times.size
    ref, emi = zeros(n), zeros(n)
    k2 = k * k
    aref = k2 * ag / (a * a)
    for i in prange(n):
        cos_alpha = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
        phase, alpha = _lambert_kernel(cos_alpha)
        ref[i] = aref * phase
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


def lambert_phase_curve_o(t, ag, a, k, tpa, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution.

    Evaluates :math:`F(t) = (k/a)^2\\, A_g\\, f(\\alpha(t))` where
    :math:`f` is the Lambert kernel and :math:`\\alpha(t)` is the
    instantaneous phase angle. The result is the planet-to-star flux
    ratio of reflected light at full phase scaled by the phase function.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_lambert_phase_curve_os`) or vector
    (:func:`_lambert_phase_curve_ov`) kernel at compile time (inside
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
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    flux : float or ndarray
        Reflected planet-to-star flux ratio. Arrays of shape (N,) for an
        array time argument.
    """
    if isinstance(t, ndarray):
        return _lambert_phase_curve_ov(t, ag, a, k, tpa, p, dt, pktable, points, coeffs)
    return _lambert_phase_curve_os(t, ag, a, k, tpa, p, dt, pktable, points, coeffs)


@overload(lambert_phase_curve_o, jit_options={'fastmath': True})
def _lambert_phase_curve_o_overload(t, ag, a, k, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, ag, a, k, tpa, p, dt, pktable, points, coeffs):
            return _lambert_phase_curve_ov(t, ag, a, k, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, a, k, tpa, p, dt, pktable, points, coeffs):
            return _lambert_phase_curve_os(t, ag, a, k, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None


def lambert_and_emission_o(t, ag, fr_night, fr_day, emi_offset, a, k,
                           tpa, p, dt, pktable, points, coeffs):
    """Lambertian reflection plus a simple cosine-emission day/night model.

    Returns the reflected and thermal-emission flux ratios separately so
    callers can combine them with their own bolometric weighting. The
    emission model is a smoothly varying interpolation between night-side
    and day-side levels driven by :math:`\\cos(\\alpha + \\delta)`, where
    :math:`\\delta` is an optional offset that captures advection.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_lambert_and_emission_os`) or vector
    (:func:`_lambert_and_emission_ov`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the flux contributions.
    ag : float
        Geometric albedo (reflected component).
    fr_night : float
        Night-side flux ratio (planet/star).
    fr_day : float
        Day-side flux ratio (planet/star).
    emi_offset : float
        Phase-angle offset of the emission peak [radians]. ``0`` puts
        peak emission at superior conjunction; non-zero values shift it
        to model day-to-night advection.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    ref : float or ndarray
        Reflected-light flux ratio. Arrays of shape (N,) for an array time
        argument.
    emi : float or ndarray
        Thermal-emission flux ratio. Arrays of shape (N,) for an array time
        argument.
    """
    if isinstance(t, ndarray):
        return _lambert_and_emission_ov(t, ag, fr_night, fr_day, emi_offset, a, k,
                                        tpa, p, dt, pktable, points, coeffs)
    return _lambert_and_emission_os(t, ag, fr_night, fr_day, emi_offset, a, k,
                                    tpa, p, dt, pktable, points, coeffs)


@overload(lambert_and_emission_o, jit_options={'fastmath': True})
def _lambert_and_emission_o_overload(t, ag, fr_night, fr_day, emi_offset, a, k,
                                     tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, ag, fr_night, fr_day, emi_offset, a, k,
                 tpa, p, dt, pktable, points, coeffs):
            return _lambert_and_emission_ov(t, ag, fr_night, fr_day, emi_offset, a, k,
                                            tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, fr_night, fr_day, emi_offset, a, k,
                 tpa, p, dt, pktable, points, coeffs):
            return _lambert_and_emission_os(t, ag, fr_night, fr_day, emi_offset, a, k,
                                            tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
