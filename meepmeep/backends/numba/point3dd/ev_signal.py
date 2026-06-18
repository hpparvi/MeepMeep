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

"""Single-expansion-point ellipsoidal-variation signal evaluators with parameter derivatives.

Derivative-returning counterpart of ``point3d.ev_signal``. Holds the
ellipsoidal-variation (tidal) flux signal with gradients
(:func:`ev_signal_cd` / :func:`ev_signal_d`). Both the signal and its
inverse-cube distance dependence are functions of the position, so the
orbital gradient block is assembled here from the position and its parameter
derivatives. The multi-expansion-point ``orbit3dd`` gradient dispatchers
delegate to the routines here.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import floor, sin, cos, sqrt, zeros, ndarray
from numpy.typing import NDArray

from .position import _pos_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _ev_signal_cd_w(time, alpha, mass_ratio, inc, c, dc, dout, dpx, dpy, dpz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the nine-parameter signal gradient into the caller-provided
    ``(9,)`` buffer ``dout`` and returns the signal. ``dpx``, ``dpy``,
    ``dpz`` are ``(7,)`` scratch buffers for the position gradients (filled
    by :func:`_pos_cd_w`); vector loops allocate them once and reuse them.

    With ``S = pre * g``, ``pre = -alpha * q * sin^2 inc``,
    ``g = (2 z^2 - d^2) / d^5`` and ``d^2 = x^2 + y^2 + z^2``, the orbital
    block chains through the position via

        dg/dtheta = (dA - 5 A dd / d^2) / d^5,
        A = 2 z^2 - d^2,  dA = -2(x dx + y dy) + 2 z dz,
        dd = (x dx + y dy + z dz) / d.

    Inclination enters the signal both implicitly through the position
    (captured by the ``i`` slot of the orbital chain) and explicitly through
    the ``sin^2 inc`` prefactor; ``inc`` is the orbital inclination, so both
    contributions are summed into the single inclination slot (slot 3). The
    explicit-prefactor term ``-alpha q 2 sin inc cos inc * g`` is added there
    after the orbital loop.

    Gradient order ``(tc, p, a, i, e, w, lan, alpha, mass_ratio)``.
    """
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    px, py, pz = _pos_cd_w(time, c, dc, dpx, dpy, dpz)
    d2 = px * px + py * py + pz * pz
    d = sqrt(d2)
    cz = pz / d
    g = (2.0 * cz * cz - 1.0) / (d2 * d)
    out = pre * g

    d5 = d2 * d2 * d
    A = 2.0 * pz * pz - d2
    for kk in range(7):
        xdotdx = px * dpx[kk] + py * dpy[kk] + pz * dpz[kk]
        dd = xdotdx / d
        dA = -2.0 * (px * dpx[kk] + py * dpy[kk]) + 2.0 * pz * dpz[kk]
        dg = (dA - 5.0 * A * dd / d2) / d5
        dout[kk] = pre * dg
    # Add the explicit sin^2(inc) prefactor contribution to the inclination
    # slot, where the implicit position-geometry contribution already lives.
    dout[3] += -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g
    dout[7] = -mass_ratio * sin2_inc * g
    dout[8] = -alpha * sin2_inc * g
    return out


@njit(fastmath=True)
def _ev_signal_cd_s(time, alpha, mass_ratio, inc, c, dc):
    """Scalar kernel for :func:`ev_signal_cd`. See that function for documentation."""
    dout = zeros(9)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    out = _ev_signal_cd_w(time, alpha, mass_ratio, inc, c, dc, dout, dpx, dpy, dpz)
    return out, dout


@njit(fastmath=True)
def ev_signal_cd_v(time, alpha, mass_ratio, inc, c, dc):
    """Vector kernel for :func:`ev_signal_cd`. See that function for documentation."""
    n = time.size
    out = zeros(n)
    dout = zeros((n, 9))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        out[j] = _ev_signal_cd_w(time[j], alpha, mass_ratio, inc, c, dc, dout[j], dpx, dpy, dpz)
    return out, dout


@njit(fastmath=True, parallel=True)
def ev_signal_cd_vp(time, alpha, mass_ratio, inc, c, dc):
    """Parallel (prange) twin of :func:`ev_signal_cd_v`.

    Explicit twin rather than a dual-decorated shared body: the
    position-gradient scratch is hoisted per thread here
    (``zeros((get_num_threads(), 7))``, indexed with ``get_thread_id()``),
    while the serial kernel keeps its cheaper single hoisted buffers -
    a shared buffer would be a data race under ``prange``.
    """
    n = time.size
    out = zeros(n)
    dout = zeros((n, 9))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        out[j] = _ev_signal_cd_w(time[j], alpha, mass_ratio, inc, c, dc, dout[j],
                                 dpx[tid], dpy[tid], dpz[tid])
    return out, dout


def ev_signal_cd(time: float | NDArray, alpha: float, mass_ratio: float, inc: float,
                 c: NDArray, dc: NDArray):
    """
    Evaluate the ellipsoidal-variation signal and its parameter derivatives at an expansion-point-centered time.

    Derivative-returning counterpart of `ev_signal.ev_signal_c`: forms
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1)/d^3` and propagates the
    chain rule through both the position (and hence the distance) and the
    explicit :math:`\\sin^2 i` factor.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. This is the orbital inclination, the
        same quantity as the ``i`` axis of the gradient; its full derivative
        (the implicit position contribution plus the explicit ``sin^2 i``
        prefactor) is accumulated into the single inclination slot (slot 3).
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    out : float or ndarray
        Ellipsoidal variation signal. Shape (N,) for an array `time`.
    dout : NDArray
        Partial derivatives of `out` with respect to
        `(tc, p, a, i, e, w, lan, alpha, mass_ratio)`. Shape (9,) for
        a scalar `time`, (N, 9) for an array `time`.
    """
    if isinstance(time, ndarray):
        return ev_signal_cd_v(time, alpha, mass_ratio, inc, c, dc)
    return _ev_signal_cd_s(time, alpha, mass_ratio, inc, c, dc)


@overload(ev_signal_cd, jit_options={'fastmath': True})
def _ev_signal_cd_overload(time, alpha, mass_ratio, inc, c, dc):
    if _is_1d_array(time):
        def impl(time, alpha, mass_ratio, inc, c, dc):
            return ev_signal_cd_v(time, alpha, mass_ratio, inc, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, alpha, mass_ratio, inc, c, dc):
            return _ev_signal_cd_s(time, alpha, mass_ratio, inc, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _ev_signal_d_s(time, alpha, mass_ratio, inc, tc, p, c, dc, te):
    """Scalar kernel for :func:`ev_signal_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _ev_signal_cd_s(time - (tc + te + epoch * p), alpha, mass_ratio, inc, c, dc)


@njit(fastmath=True)
def ev_signal_d_v(time, alpha, mass_ratio, inc, tc, p, c, dc, te):
    """Vector kernel for :func:`ev_signal_d`. See that function for documentation."""
    n = time.size
    out = zeros(n)
    dout = zeros((n, 9))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        out[j] = _ev_signal_cd_w(time[j] - (tc + te + epoch * p), alpha, mass_ratio, inc, c, dc,
                                 dout[j], dpx, dpy, dpz)
    return out, dout


@njit(fastmath=True, parallel=True)
def ev_signal_d_vp(time, alpha, mass_ratio, inc, tc, p, c, dc, te):
    """Parallel (prange) twin of :func:`ev_signal_d_v`.

    Explicit twin with per-thread position-gradient scratch; see
    :func:`ev_signal_cd_vp`.
    """
    n = time.size
    out = zeros(n)
    dout = zeros((n, 9))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        out[j] = _ev_signal_cd_w(time[j] - (tc + te + epoch * p), alpha, mass_ratio, inc, c, dc,
                                 dout[j], dpx[tid], dpy[tid], dpz[tid])
    return out, dout


def ev_signal_d(time: float | NDArray, alpha: float, mass_ratio: float, inc: float,
                tc: float, p: float, c: NDArray, dc: NDArray, te: float = 0.0):
    """
    Evaluate the ellipsoidal-variation signal and its parameter derivatives at an absolute time.

    Direct counterpart of `ev_signal_cd`: epoch-folds the absolute time
    `time` around the expansion point and delegates to `ev_signal_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Absolute observation time(s) in the same units as `tc` and `p`.
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. This is the orbital inclination, the
        same quantity as the ``i`` axis of the gradient; its full derivative
        (the implicit position contribution plus the explicit ``sin^2 i``
        prefactor) is accumulated into the single inclination slot (slot 3).
    tc : float
        Transit-centre time (time of inferior conjunction), on the same
        time axis as `time`.
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.
    te : float, optional
        Expansion-point offset from the transit centre [days] - the same value that
        was passed to `solve3d_d`. Defaults to 0.0, the expansion point at the
        transit centre.

    Returns
    -------
    out : float or ndarray
        Ellipsoidal variation signal. Shape (N,) for an array `time`.
    dout : NDArray
        Partial derivatives of `out` with respect to
        `(tc, p, a, i, e, w, lan, alpha, mass_ratio)`. Shape (9,) for
        a scalar `time`, (N, 9) for an array `time`.
    """
    if isinstance(time, ndarray):
        return ev_signal_d_v(time, alpha, mass_ratio, inc, tc, p, c, dc, te)
    return _ev_signal_d_s(time, alpha, mass_ratio, inc, tc, p, c, dc, te)


@overload(ev_signal_d, jit_options={'fastmath': True})
def _ev_signal_d_overload(time, alpha, mass_ratio, inc, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, alpha, mass_ratio, inc, tc, p, c, dc, te=0.0):
            return ev_signal_d_v(time, alpha, mass_ratio, inc, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, alpha, mass_ratio, inc, tc, p, c, dc, te=0.0):
            return _ev_signal_d_s(time, alpha, mass_ratio, inc, tc, p, c, dc, te)
        return impl
    return None
