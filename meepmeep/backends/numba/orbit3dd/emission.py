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

"""Multi-expansion-point cosine emission phase-curve evaluators with parameter derivatives.

Epoch folding and expansion-point lookup happen here; the flux and its gradient
are delegated to the single-expansion-point
:func:`~meepmeep.backends.numba.point3dd.emission.emission_phase_curve_cd`.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.emission import _emission_phase_curve_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _emission_phase_curve_ow(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                             dout, dpx, dpy, dpz, dvx, dvy, dvz):
    """Write-into orbit kernel for the emission flux and its gradient.

    Epoch-folds, looks up the expansion point, and delegates the flux and
    ten-parameter gradient evaluation to the single-expansion-point
    :func:`~meepmeep.backends.numba.point3dd.emission._emission_phase_curve_cd_w`.
    Writes the gradient into the caller-provided ``(10,)`` buffer ``dout`` and
    returns the flux. ``dpx``, ``dpy``, ``dpz``, ``dvx``, ``dvy``, ``dvz`` are
    ``(7,)`` scratch buffers for the position and velocity gradients; vector
    loops allocate them once and reuse them.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _emission_phase_curve_cd_w(tc - ep_times[ix] * p, k, fratio, offset, coeffs[ix], dcoeffs[ix],
                                      dout, dpx, dpy, dpz, dvx, dvy, dvz)


@njit(fastmath=True)
def _emission_phase_curve_osd(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Scalar kernel for :func:`emission_phase_curve_od`. See that function for documentation."""
    dout = zeros(10)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    flux = _emission_phase_curve_ow(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                                    dout, dpx, dpy, dpz, dvx, dvy, dvz)
    return flux, dout


@njit(fastmath=True)
def emission_phase_curve_ovd(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Vector kernel for :func:`emission_phase_curve_od`. See that function for documentation."""
    n = t.size
    flux = zeros(n)
    dout = zeros((n, 10))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    for j in range(n):
        flux[j] = _emission_phase_curve_ow(t[j], k, fratio, offset, tpa, p, dt, ep_table, ep_times,
                                           coeffs, dcoeffs, dout[j], dpx, dpy, dpz, dvx, dvy, dvz)
    return flux, dout


@njit(fastmath=True, parallel=True)
def emission_phase_curve_ovdp(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`emission_phase_curve_ovd`.

    The position- and velocity-gradient scratch is hoisted per thread; a single
    shared buffer would be a data race under ``prange``.
    """
    n = t.size
    flux = zeros(n)
    dout = zeros((n, 10))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    dvx = zeros((nt, 7))
    dvy = zeros((nt, 7))
    dvz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        flux[j] = _emission_phase_curve_ow(t[j], k, fratio, offset, tpa, p, dt, ep_table, ep_times,
                                           coeffs, dcoeffs, dout[j], dpx[tid], dpy[tid], dpz[tid],
                                           dvx[tid], dvy[tid], dvz[tid])
    return flux, dout


def emission_phase_curve_od(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Cosine emission phase-curve flux with gradients.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_emission_phase_curve_osd`) or vector
    (:func:`emission_phase_curve_ovd`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Derivative ordering: ``(tc, p, a, i, e, w, lan, k, fratio, offset)`` -
    length 10.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the flux contribution and gradient.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    fratio : float
        Dayside-to-nightside per-surface-element flux ratio (amplitude
        scaling).
    offset : float
        Hotspot offset [radians].
    tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    flux : float or ndarray
        Emitted planet-to-star flux ratio. Arrays of shape (N,) for an array
        time argument.
    dflux : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan, k, fratio, offset)``. Shape
        (10,) for a scalar time, (N, 10) for an array time.
    """
    if isinstance(t, ndarray):
        return emission_phase_curve_ovd(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    return _emission_phase_curve_osd(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)


@overload(emission_phase_curve_od, jit_options={'fastmath': True})
def _emission_phase_curve_od_overload(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return emission_phase_curve_ovd(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return _emission_phase_curve_osd(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    return None
