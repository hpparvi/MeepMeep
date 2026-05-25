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

"""Multi-knot Lambertian phase-curve and emission evaluators with parameter derivatives.

Holds the two reflected-light quantities that share the Lambertian phase
kernel: the pure phase curve (:func:`lambert_phase_curve_od`) and the
combined reflection-plus-emission model (:func:`lambert_and_emission_od`).
"""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, pi, sqrt, sin, cos, arccos, ndarray

from .phase_angle import _cos_alpha_osd
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
def _lambert_phase_curve_osd(time, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux and parameter derivatives at scalar time.

    Derivative ordering: ``(phase, p, a, i, e, w, ag, k)`` — length 8.

    Parameters
    ----------
    time : float
        Time at which to evaluate the flux contribution and gradient.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    flux : float
        Reflected planet-to-star flux ratio.
    dflux : ndarray, shape (8,)
        Gradient w.r.t. ``(phase, p, a, i, e, w, ag, k)``.
    """
    amplitude = k * k * ag / (a * a)
    ca, dca = _cos_alpha_osd(time, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    phase, _, dphase_dc = _lambert_kernel_d(ca)
    flux = amplitude * phase

    dflux = zeros(8)
    # Orbital block — chain through cos_alpha and through amplitude (only `a` matters).
    for kk in range(6):
        dflux[kk] = amplitude * dphase_dc * dca[kk]
    # Add d(amplitude)/da contribution to the `a` slot (index 2):
    # damplitude/da = -2 k^2 ag / a^3.
    dflux[2] += -2.0 * k * k * ag / (a * a * a) * phase
    # Extras: ag (index 6), k (index 7).
    dflux[6] = (k * k / (a * a)) * phase
    dflux[7] = (2.0 * k * ag / (a * a)) * phase
    return flux, dflux


@njit(fastmath=True)
def _lambert_phase_curve_ovd(times, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux and parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the flux contribution and gradient.
    ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_lambert_phase_curve_osd`.

    Returns
    -------
    flux : ndarray, shape (N,)
        Reflected planet-to-star flux ratio per time.
    dflux : ndarray, shape (N, 8)
        Gradient w.r.t. ``(phase, p, a, i, e, w, ag, k)`` per time.
    """
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 8))
    inv_a2 = 1.0 / (a * a)
    amplitude = k * k * ag * inv_a2
    da_amp = -2.0 * k * k * ag / (a * a * a)
    dag_amp = k * k * inv_a2
    dk_amp = 2.0 * k * ag * inv_a2
    for j in range(n):
        ca, dca = _cos_alpha_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        phase, _, dphase_dc = _lambert_kernel_d(ca)
        flux[j] = amplitude * phase
        for kk in range(6):
            dflux[j, kk] = amplitude * dphase_dc * dca[kk]
        dflux[j, 2] += da_amp * phase
        dflux[j, 6] = dag_amp * phase
        dflux[j, 7] = dk_amp * phase
    return flux, dflux


@njit(fastmath=True)
def _lambert_and_emission_osd(t, ag, fr_night, fr_day, emi_offset, a, k,
                              tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus cosine-emission day/night model with derivatives at scalar time.

    Scalar counterpart of :func:`_lambert_and_emission_ovd`. Derivative
    ordering: ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``
    — length 11.

    Parameters
    ----------
    t : float
        Time at which to evaluate the flux contributions and gradients.
    ag, fr_night, fr_day, emi_offset, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_lambert_and_emission_ovd`.

    Returns
    -------
    ref : float
        Reflected (Lambertian) flux contribution.
    emi : float
        Thermal emission contribution.
    dref : ndarray, shape (11,)
        Gradient of ``ref`` w.r.t.
        ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``.
    demi : ndarray, shape (11,)
        Gradient of ``emi`` w.r.t. the same parameter block.
    """
    k2 = k * k
    inv_a2 = 1.0 / (a * a)
    aref = k2 * ag * inv_a2
    daref_da = -2.0 * k2 * ag / (a * a * a)
    daref_dag = k2 * inv_a2
    daref_dk = 2.0 * k * ag * inv_a2

    ca, dca = _cos_alpha_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    phase, alpha, dphase_dc = _lambert_kernel_d(ca)

    dref = zeros(11)
    demi = zeros(11)

    ref = aref * phase
    for kk in range(6):
        dref[kk] = aref * dphase_dc * dca[kk]
    dref[2] += daref_da * phase
    dref[6] = daref_dag * phase
    dref[10] = daref_dk * phase

    cs = cos(alpha + emi_offset)
    sn = sin(alpha + emi_offset)
    bracket = fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cs)
    emi = k2 * bracket

    ca_clamped = ca
    if ca_clamped > 1.0:
        ca_clamped = 1.0
    elif ca_clamped < -1.0:
        ca_clamped = -1.0
    s = sqrt(1.0 - ca_clamped * ca_clamped)
    if s < 1e-12:
        dalpha_dc = 0.0
    else:
        dalpha_dc = -1.0 / s
    demi_dalpha = k2 * (fr_day - fr_night) * 0.5 * sn
    for kk in range(6):
        demi[kk] = demi_dalpha * dalpha_dc * dca[kk]
    demi[7] = k2 * (1.0 - 0.5 * (1.0 - cs))
    demi[8] = k2 * 0.5 * (1.0 - cs)
    demi[9] = k2 * (fr_day - fr_night) * 0.5 * sn
    demi[10] = 2.0 * k * bracket

    return ref, emi, dref, demi


@njit(fastmath=True)
def _lambert_and_emission_ovd(times, ag, fr_night, fr_day, emi_offset, a, k,
                             tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus cosine-emission day/night model with parameter derivatives.

    Derivative ordering: ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``
    — length 11.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the flux contributions and gradients.
    ag : float
        Geometric albedo (reflected component).
    fr_night : float
        Night-side flux ratio (planet/star).
    fr_day : float
        Day-side flux ratio (planet/star).
    emi_offset : float
        Phase-angle offset of the emission peak [radians].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    ref : ndarray, shape (N,)
        Reflected (Lambertian) flux contribution per time.
    emi : ndarray, shape (N,)
        Thermal emission contribution per time.
    dref : ndarray, shape (N, 11)
        Gradient of ``ref`` w.r.t.
        ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``.
    demi : ndarray, shape (N, 11)
        Gradient of ``emi`` w.r.t. the same parameter block.
    """
    n = times.size
    ref = zeros(n)
    emi = zeros(n)
    dref = zeros((n, 11))
    demi = zeros((n, 11))
    k2 = k * k
    inv_a2 = 1.0 / (a * a)
    aref = k2 * ag * inv_a2
    daref_da = -2.0 * k2 * ag / (a * a * a)
    daref_dag = k2 * inv_a2
    daref_dk = 2.0 * k * ag * inv_a2

    for j in range(n):
        ca, dca = _cos_alpha_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        phase, alpha, dphase_dc = _lambert_kernel_d(ca)

        # --- reflected component ---
        ref[j] = aref * phase
        for kk in range(6):
            dref[j, kk] = aref * dphase_dc * dca[kk]
        dref[j, 2] += daref_da * phase
        dref[j, 6] = daref_dag * phase
        # fr_night, fr_day, emi_offset (indices 7..9) are zero for ref.
        dref[j, 10] = daref_dk * phase

        # --- emission component ---
        # emi = k^2 · (fr_night + (fr_day - fr_night) · 0.5 · (1 - cos(alpha + emi_offset)))
        cs = cos(alpha + emi_offset)
        sn = sin(alpha + emi_offset)
        bracket = fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cs)
        emi[j] = k2 * bracket

        # d(alpha)/d(cos_alpha) = -1/sqrt(1 - ca^2). Avoid blow-up at |ca|=1
        # by clamping like _lambert_kernel_d does (interior tests safe).
        ca_clamped = ca
        if ca_clamped > 1.0:
            ca_clamped = 1.0
        elif ca_clamped < -1.0:
            ca_clamped = -1.0
        s = sqrt(1.0 - ca_clamped * ca_clamped)
        if s < 1e-12:
            dalpha_dc = 0.0
        else:
            dalpha_dc = -1.0 / s
        # demi/dorbital via cos_alpha → alpha → bracket
        # demi/dα = k^2 · (fr_day - fr_night) · 0.5 · sin(alpha + emi_offset)
        demi_dalpha = k2 * (fr_day - fr_night) * 0.5 * sn
        for kk in range(6):
            demi[j, kk] = demi_dalpha * dalpha_dc * dca[kk]
        # ag (6) does not enter emi; leave 0.
        # fr_night (7): k^2 · (1 - 0.5·(1-cs)) = k^2 · (0.5 + 0.5·cs)
        demi[j, 7] = k2 * (1.0 - 0.5 * (1.0 - cs))
        # fr_day (8):   k^2 · 0.5 · (1 - cs)
        demi[j, 8] = k2 * 0.5 * (1.0 - cs)
        # emi_offset (9): k^2 · (fr_day - fr_night) · 0.5 · sin(alpha + emi_offset)
        demi[j, 9] = k2 * (fr_day - fr_night) * 0.5 * sn
        # k (10): 2k · bracket
        demi[j, 10] = 2.0 * k * bracket

    return ref, emi, dref, demi


def lambert_phase_curve_od(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux with gradients.

    See :func:`_lambert_phase_curve_osd` / :func:`_lambert_phase_curve_ovd`.
    """
    if isinstance(t, ndarray):
        return _lambert_phase_curve_ovd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _lambert_phase_curve_osd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(lambert_phase_curve_od, jit_options={'fastmath': True})
def _lambert_phase_curve_od_overload(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_phase_curve_ovd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_phase_curve_osd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


def lambert_and_emission_od(t, ag, fr_night, fr_day, emi_offset, a, k,
                            tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus emission with gradients.

    See :func:`_lambert_and_emission_osd` / :func:`_lambert_and_emission_ovd`.
    """
    if isinstance(t, ndarray):
        return _lambert_and_emission_ovd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                         tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _lambert_and_emission_osd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                     tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(lambert_and_emission_od, jit_options={'fastmath': True})
def _lambert_and_emission_od_overload(t, ag, fr_night, fr_day, emi_offset, a, k,
                                      tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, ag, fr_night, fr_day, emi_offset, a, k,
                 tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_and_emission_ovd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                             tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, fr_night, fr_day, emi_offset, a, k,
                 tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_and_emission_osd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                             tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
