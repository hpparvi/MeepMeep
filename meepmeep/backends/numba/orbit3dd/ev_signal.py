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

"""Multi-knot ellipsoidal-variation signal evaluators with parameter derivatives."""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, sin, cos, sqrt, ndarray

from .position import _pos_osd, _pos_ow
from ._common import _is_1d_array


@njit(fastmath=True)
def _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`ev_signal_od`. See that function for documentation."""
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    d2 = x * x + y * y + z * z
    d = sqrt(d2)
    cz = z / d
    g = (2.0 * cz * cz - 1.0) / (d2 * d)
    out = pre * g

    d5 = d2 * d2 * d
    A = 2.0 * z * z - d2
    dout = zeros(10)
    for kk in range(7):
        xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
        dd = xdotdx / d
        dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
        dg = (dA - 5.0 * A * dd / d2) / d5
        dout[kk] = pre * dg
    dout[7] = -mass_ratio * sin2_inc * g
    dout[8] = -alpha * sin2_inc * g
    dout[9] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g
    return out, dout


@njit(fastmath=True)
def _ev_signal_ovd(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`ev_signal_od`. See that function for documentation."""
    n = times.size
    out = zeros(n)
    dout = zeros((n, 10))
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    for j in range(n):
        x, y, z = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        # S = pre · g, where g = (2 cz^2 - 1) / d^3.
        # Rewrite g = (2 z^2 - d^2) / d^5 = (2 z^2 / d^5) - 1/d^3.
        g = (2.0 * cz * cz - 1.0) / (d2 * d)
        out[j] = pre * g

        # dg/dθ via dx, dy, dz. Use g = (2 z^2 - d^2) / d^5.
        # Let A = 2 z^2 - d^2,  d^5 = d2^2 · d.
        # dA = 4 z·dz - 2(x·dx + y·dy + z·dz)
        #    = -2(x·dx + y·dy) + 2 z·dz
        # d(d^5)/dθ = 5 d^3 · dd, with dd = (x·dx + y·dy + z·dz)/d.
        # dg = (dA · d^5 - A · 5 d^3 · dd) / d^10
        #    = (dA - 5 A · dd / d^2) / d^5.
        d5 = d2 * d2 * d
        A = 2.0 * z * z - d2
        for kk in range(7):
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dd = xdotdx / d
            dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
            dg = (dA - 5.0 * A * dd / d2) / d5
            dout[j, kk] = pre * dg
        # Extras (no orbital chain).
        # alpha (7): dS/dalpha = -mass_ratio · sin2_inc · g
        dout[j, 7] = -mass_ratio * sin2_inc * g
        # mass_ratio (8): dS/dmr = -alpha · sin2_inc · g
        dout[j, 8] = -alpha * sin2_inc * g
        # inc (9): d(sin^2 inc)/dinc = 2 sin_inc · cos_inc
        dout[j, 9] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g

    return out, dout


@njit(fastmath=True, parallel=True)
def _ev_signal_ovdp(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`_ev_signal_ovd`.

    The position-gradient scratch is hoisted per thread; a single shared
    buffer would be a data race under ``prange``.
    """
    n = times.size
    out = zeros(n)
    dout = zeros((n, 10))
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        x, y, z = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        g = (2.0 * cz * cz - 1.0) / (d2 * d)
        out[j] = pre * g
        d5 = d2 * d2 * d
        A = 2.0 * z * z - d2
        for kk in range(7):
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dd = xdotdx / d
            dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
            dg = (dA - 5.0 * A * dd / d2) / d5
            dout[j, kk] = pre * dg
        dout[j, 7] = -mass_ratio * sin2_inc * g
        dout[j, 8] = -alpha * sin2_inc * g
        dout[j, 9] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g
    return out, dout


def ev_signal_od(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal with gradients.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_ev_signal_osd`) or vector (:func:`_ev_signal_ovd`) kernel
    at compile time (inside ``@njit``) or at call time (pure Python).

    Implements
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1) / d^3`
    where :math:`c_z = z/d` and :math:`d = \\sqrt{x^2 + y^2 + z^2}`. The
    function-local ``inc`` parameter is independent of the orbital
    inclination ``i`` - callers that share them should sum the two
    derivative slots.

    Time argument is the 4th positional.

    Derivative ordering: ``(tc, p, a, i, e, w, lan, alpha, mass_ratio, inc)`` -
    length 10.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. Treated as a function-local input
        independent of the orbital ``i`` axis of the gradient.
    t : float or ndarray
        Time(s) at which to evaluate the signal and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    out : float or ndarray
        Ellipsoidal variation signal. Arrays of shape (N,) for an array time
        argument.
    dout : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan, alpha, mass_ratio, inc)``.
        Shape (10,) for a scalar time, (N, 10) for an array time.
    """
    if isinstance(t, ndarray):
        return _ev_signal_ovd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(ev_signal_od, jit_options={'fastmath': True})
def _ev_signal_od_overload(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _ev_signal_ovd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
