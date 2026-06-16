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

"""Multi-expansion-point Lambertian phase-curve evaluators with parameter derivatives.

Holds the Lambertian reflected-light phase curve with gradients
(:func:`lambert_phase_curve_od`) and its shared phase kernel
(:func:`_lambert_kernel_d`).
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, pi, sqrt, arccos, ndarray

from .phase_angle import _cos_alpha_osd, _cos_alpha_ow
from ._common import _is_1d_array


@njit(fastmath=True)
def _lambert_kernel_d(cos_alpha):
    """Lambertian phase function, alpha, and ``dphase/dcos_alpha``.

    The analytic derivative of
    :math:`\\mathrm{phase}(c) = (\\sqrt{1-c^2} + (\\pi - \\arccos c)\\,c)/\\pi`
    simplifies to :math:`(\\pi - \\arccos c)/\\pi` because the contributions
    from :math:`d/dc \\sqrt{1-c^2}` and :math:`c \\cdot d/dc \\arccos c`
    cancel exactly.

    Parameters
    ----------
    cos_alpha : float
        Cosine of the phase angle. Clamped internally to ``[-1, 1]``.

    Returns
    -------
    phase : float
        Value of the Lambert kernel, in :math:`[0, 1]`.
    alpha : float
        Phase angle :math:`\\arccos(\\mathrm{cos\\_alpha})` [radians].
    dphase_dc : float
        Derivative :math:`d\\,\\mathrm{phase}/d\\,\\mathrm{cos\\_alpha}`.
    """
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
    alpha = arccos(cos_alpha)
    phase = (sin_alpha + (pi - alpha) * cos_alpha) / pi
    dphase_dc = (pi - alpha) / pi
    return phase, alpha, dphase_dc


@njit(fastmath=True)
def _lambert_phase_curve_osd(time, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Scalar kernel for :func:`lambert_phase_curve_od`. See that function for documentation."""
    amplitude = k * k * ag / (a * a)
    ca, dca = _cos_alpha_osd(time, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    phase, _, dphase_dc = _lambert_kernel_d(ca)
    flux = amplitude * phase

    dflux = zeros(9)
    # Orbital block — chain through cos_alpha and through amplitude (only `a` matters).
    for kk in range(7):
        dflux[kk] = amplitude * dphase_dc * dca[kk]
    # Add d(amplitude)/da contribution to the `a` slot (index 2):
    # damplitude/da = -2 k^2 ag / a^3.
    dflux[2] += -2.0 * k * k * ag / (a * a * a) * phase
    # Extras: ag (index 7), k (index 8).
    dflux[7] = (k * k / (a * a)) * phase
    dflux[8] = (2.0 * k * ag / (a * a)) * phase
    return flux, dflux


@njit(fastmath=True)
def lambert_phase_curve_ovd(times, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Vector kernel for :func:`lambert_phase_curve_od`. See that function for documentation."""
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    inv_a2 = 1.0 / (a * a)
    amplitude = k * k * ag * inv_a2
    da_amp = -2.0 * k * k * ag / (a * a * a)
    dag_amp = k * k * inv_a2
    dk_amp = 2.0 * k * ag * inv_a2
    dca = zeros(7)
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    for j in range(n):
        ca = _cos_alpha_ow(times[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dca, dx, dy, dz)
        phase, _, dphase_dc = _lambert_kernel_d(ca)
        flux[j] = amplitude * phase
        for kk in range(7):
            dflux[j, kk] = amplitude * dphase_dc * dca[kk]
        dflux[j, 2] += da_amp * phase
        dflux[j, 7] = dag_amp * phase
        dflux[j, 8] = dk_amp * phase
    return flux, dflux


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_ovdp(times, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`lambert_phase_curve_ovd`.

    The phase-angle and position-gradient scratch is hoisted per thread; a
    single shared buffer would be a data race under ``prange``.
    """
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    inv_a2 = 1.0 / (a * a)
    amplitude = k * k * ag * inv_a2
    da_amp = -2.0 * k * k * ag / (a * a * a)
    dag_amp = k * k * inv_a2
    dk_amp = 2.0 * k * ag * inv_a2
    nt = get_num_threads()
    dcas, dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dca = dcas[tid]
        ca = _cos_alpha_ow(times[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                           dca, dxs[tid], dys[tid], dzs[tid])
        phase, _, dphase_dc = _lambert_kernel_d(ca)
        flux[j] = amplitude * phase
        for kk in range(7):
            dflux[j, kk] = amplitude * dphase_dc * dca[kk]
        dflux[j, 2] += da_amp * phase
        dflux[j, 7] = dag_amp * phase
        dflux[j, 8] = dk_amp * phase
    return flux, dflux


def lambert_phase_curve_od(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Lambertian phase-curve flux with gradients.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_lambert_phase_curve_osd`) or vector
    (:func:`lambert_phase_curve_ovd`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Derivative ordering: ``(tc, p, a, i, e, w, lan, ag, k)`` - length 9.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the flux contribution and gradient.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    flux : float or ndarray
        Reflected planet-to-star flux ratio. Arrays of shape (N,) for an array
        time argument.
    dflux : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan, ag, k)``. Shape (9,) for a
        scalar time, (N, 9) for an array time.
    """
    if isinstance(t, ndarray):
        return lambert_phase_curve_ovd(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    return _lambert_phase_curve_osd(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)


@overload(lambert_phase_curve_od, jit_options={'fastmath': True})
def _lambert_phase_curve_od_overload(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return lambert_phase_curve_ovd(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return _lambert_phase_curve_osd(t, ag, a, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    return None
