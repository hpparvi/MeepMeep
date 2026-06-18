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

"""Multi-expansion-point ellipsoidal-variation signal evaluators with parameter derivatives.

Epoch folding and expansion-point lookup happen here; the signal and its
gradient are delegated to the single-expansion-point
:func:`~meepmeep.backends.numba.point3dd.ev_signal.ev_signal_cd`.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.ev_signal import _ev_signal_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _ev_signal_ow(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                  dout, dpx, dpy, dpz):
    """Write-into orbit kernel for the ellipsoidal-variation signal and its gradient.

    Epoch-folds, looks up the expansion point, and delegates the signal and
    nine-parameter gradient evaluation to the single-expansion-point
    :func:`~meepmeep.backends.numba.point3dd.ev_signal._ev_signal_cd_w`.
    Writes the gradient into the caller-provided ``(9,)`` buffer ``dout``
    and returns the signal. ``dpx``, ``dpy``, ``dpz`` are ``(7,)`` scratch
    buffers for the position gradients; vector loops allocate them once and
    reuse them.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _ev_signal_cd_w(tc - ep_times[ix] * p, alpha, mass_ratio, inc, coeffs[ix], dcoeffs[ix],
                           dout, dpx, dpy, dpz)


@njit(fastmath=True)
def _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Scalar kernel for :func:`ev_signal_od`. See that function for documentation."""
    dout = zeros(9)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    out = _ev_signal_ow(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                        dout, dpx, dpy, dpz)
    return out, dout


@njit(fastmath=True)
def ev_signal_ovd(alpha, mass_ratio, inc, times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Vector kernel for :func:`ev_signal_od`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    dout = zeros((n, 9))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        out[j] = _ev_signal_ow(alpha, mass_ratio, inc, times[j], tpa, p, dt, ep_table, ep_times,
                               coeffs, dcoeffs, dout[j], dpx, dpy, dpz)
    return out, dout


@njit(fastmath=True, parallel=True)
def ev_signal_ovdp(alpha, mass_ratio, inc, times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`ev_signal_ovd`.

    The position-gradient scratch is hoisted per thread; a single shared
    buffer would be a data race under ``prange``.
    """
    n = times.size
    out = zeros(n)
    dout = zeros((n, 9))
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        out[j] = _ev_signal_ow(alpha, mass_ratio, inc, times[j], tpa, p, dt, ep_table, ep_times,
                               coeffs, dcoeffs, dout[j], dxs[tid], dys[tid], dzs[tid])
    return out, dout


def ev_signal_od(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Ellipsoidal variation signal with gradients.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_ev_signal_osd`) or vector (:func:`ev_signal_ovd`) kernel
    at compile time (inside ``@njit``) or at call time (pure Python).

    Implements
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1) / d^3`
    where :math:`c_z = z/d` and :math:`d = \\sqrt{x^2 + y^2 + z^2}`. The
    ``inc`` argument is the orbital inclination; its full derivative (the
    implicit position contribution plus the explicit ``sin^2 i`` prefactor)
    is accumulated into the single inclination slot (slot 3).

    Time argument is the 4th positional.

    Derivative ordering: ``(tc, p, a, i, e, w, lan, alpha, mass_ratio)`` -
    length 9.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. The same quantity as the ``i`` axis
        of the gradient; its full derivative lands in slot 3.
    t : float or ndarray
        Time(s) at which to evaluate the signal and gradient.
    tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    out : float or ndarray
        Ellipsoidal variation signal. Arrays of shape (N,) for an array time
        argument.
    dout : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan, alpha, mass_ratio)``.
        Shape (9,) for a scalar time, (N, 9) for an array time.
    """
    if isinstance(t, ndarray):
        return ev_signal_ovd(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    return _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)


@overload(ev_signal_od, jit_options={'fastmath': True})
def _ev_signal_od_overload(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return ev_signal_ovd(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    return None
