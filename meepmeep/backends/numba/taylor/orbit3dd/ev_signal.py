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

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sin, cos, sqrt, ndarray

from .position import _pos_osd
from ._common import _is_1d_array


@njit(fastmath=True)
def _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal and derivatives at scalar time.

    Scalar counterpart of :func:`_ev_signal_ovd`. Derivative ordering:
    ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` — length 9.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient.
    mass_ratio : float
        Planet-to-star mass ratio.
    inc : float
        Orbital inclination [radians], independent of the orbital ``i`` axis.
    t : float
        Time at which to evaluate the signal and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    out : float
        Ellipsoidal variation signal.
    dout : ndarray, shape (9,)
        Gradient w.r.t. ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)``.
    """
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
    dout = zeros(9)
    for kk in range(6):
        xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
        dd = xdotdx / d
        dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
        dg = (dA - 5.0 * A * dd / d2) / d5
        dout[kk] = pre * dg
    dout[6] = -mass_ratio * sin2_inc * g
    dout[7] = -alpha * sin2_inc * g
    dout[8] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g
    return out, dout


@njit(fastmath=True)
def _ev_signal_ovd(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal and parameter derivatives.

    Implements
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1) / d^3`
    where :math:`c_z = z/d` and :math:`d = \\sqrt{x^2 + y^2 + z^2}`. The
    function-local ``inc`` parameter is independent of the orbital
    inclination ``i`` — callers that share them should sum the two
    derivative slots.

    Derivative ordering: ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` —
    length 9.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. Treated as a function-local input
        independent of the orbital ``i`` axis of the gradient.
    times : ndarray, shape (N,)
        Times at which to evaluate the signal and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    out : ndarray, shape (N,)
        Ellipsoidal variation signal per time.
    dout : ndarray, shape (N, 9)
        Gradient w.r.t.
        ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` per time.
    """
    n = times.size
    out = zeros(n)
    dout = zeros((n, 9))
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
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
        for kk in range(6):
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dd = xdotdx / d
            dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
            dg = (dA - 5.0 * A * dd / d2) / d5
            dout[j, kk] = pre * dg
        # Extras (no orbital chain).
        # alpha (6): dS/dalpha = -mass_ratio · sin2_inc · g
        dout[j, 6] = -mass_ratio * sin2_inc * g
        # mass_ratio (7): dS/dmr = -alpha · sin2_inc · g
        dout[j, 7] = -alpha * sin2_inc * g
        # inc (8): d(sin^2 inc)/dinc = 2 sin_inc · cos_inc
        dout[j, 8] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g

    return out, dout


def ev_signal_od(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal with gradients.

    Time argument is the 4th positional. See :func:`_ev_signal_osd` /
    :func:`_ev_signal_ovd`.
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
