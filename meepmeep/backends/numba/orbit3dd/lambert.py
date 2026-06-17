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
(:func:`lambert_phase_curve_od`). Epoch folding and expansion-point lookup
happen here; the flux and its gradient are delegated to the
single-expansion-point
:func:`~meepmeep.backends.numba.point3dd.lambert.lambert_phase_curve_cd`. The
shared phase kernel (:func:`_lambert_kernel_d`) is re-exported from
``point3dd.lambert`` for backward compatibility.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.lambert import _lambert_kernel_d, _lambert_phase_curve_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _lambert_phase_curve_ow(time, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                            dflux, dpx, dpy, dpz):
    """Write-into orbit kernel for the Lambert flux and its gradient.

    Epoch-folds, looks up the expansion point, and delegates the flux and
    nine-parameter gradient evaluation to the single-expansion-point
    :func:`~meepmeep.backends.numba.point3dd.lambert._lambert_phase_curve_cd_w`.
    Writes the gradient into the caller-provided ``(9,)`` buffer ``dflux``
    and returns the flux. ``dpx``, ``dpy``, ``dpz`` are ``(7,)`` scratch
    buffers for the position gradients; vector loops allocate them once and
    reuse them.
    """
    epoch = floor((time - tpa) / p)
    tc = time - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _lambert_phase_curve_cd_w(tc - ep_times[ix] * p, ag, k, coeffs[ix], dcoeffs[ix],
                                     dflux, dpx, dpy, dpz)


@njit(fastmath=True)
def _lambert_phase_curve_osd(time, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Scalar kernel for :func:`lambert_phase_curve_od`. See that function for documentation."""
    dflux = zeros(9)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    flux = _lambert_phase_curve_ow(time, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                                   dflux, dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True)
def lambert_phase_curve_ovd(times, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Vector kernel for :func:`lambert_phase_curve_od`. See that function for documentation."""
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        flux[j] = _lambert_phase_curve_ow(times[j], ag, k, tpa, p, dt, ep_table, ep_times,
                                          coeffs, dcoeffs, dflux[j], dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_ovdp(times, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`lambert_phase_curve_ovd`.

    The phase-angle and position-gradient scratch is hoisted per thread; a
    single shared buffer would be a data race under ``prange``.
    """
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        flux[j] = _lambert_phase_curve_ow(times[j], ag, k, tpa, p, dt, ep_table, ep_times,
                                          coeffs, dcoeffs, dflux[j], dxs[tid], dys[tid], dzs[tid])
    return flux, dflux


def lambert_phase_curve_od(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
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
        return lambert_phase_curve_ovd(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    return _lambert_phase_curve_osd(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)


@overload(lambert_phase_curve_od, jit_options={'fastmath': True})
def _lambert_phase_curve_od_overload(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return lambert_phase_curve_ovd(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return _lambert_phase_curve_osd(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    return None
