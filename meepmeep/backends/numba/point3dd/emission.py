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

"""Single-expansion-point cosine emission phase-curve evaluators with parameter derivatives.

Derivative-returning counterpart of ``point3d.emission``. The signal depends on
the planet position *and* velocity (the orbital normal enters the signed
in-plane component), so the orbital gradient block is assembled here from both
the position and the velocity together with their parameter derivatives. The
multi-expansion-point ``orbit3dd`` gradient dispatchers delegate to the routines
here.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import floor, sin, cos, sqrt, zeros, ndarray
from numpy.typing import NDArray

from .position import _pos_cd_w
from .velocity import _vel_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _emission_phase_curve_cd_w(time, k, fratio, offset, c, dc, dout, dpx, dpy, dpz, dvx, dvy, dvz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the ten-parameter flux gradient into the caller-provided ``(10,)``
    buffer ``dout`` and returns the flux. ``dpx``, ``dpy``, ``dpz`` are
    ``(7,)`` scratch buffers for the position gradients (filled by
    :func:`_pos_cd_w`) and ``dvx``, ``dvy``, ``dvz`` for the velocity
    gradients (filled by :func:`_vel_cd_w`); vector loops allocate them once
    and reuse them.

    Gradient order ``(tc, p, a, i, e, w, lan, k, fratio, offset)``. The orbital
    block chains through the phase-angle cosine ``cz = -z/d`` and the signed
    in-plane component ``s = -(w_x*y - w_y*x)/(L*d)``, with ``w = r x v`` and
    ``L = |w|``.
    """
    x, y, z = _pos_cd_w(time, c, dc, dpx, dpy, dpz)
    vx, vy, vz = _vel_cd_w(time, c, dc, dvx, dvy, dvz)
    d2 = x * x + y * y + z * z
    d = sqrt(d2)
    wx = y * vz - z * vy
    wy = z * vx - x * vz
    wz = x * vy - y * vx
    el = sqrt(wx * wx + wy * wy + wz * wz)
    cz = -z / d
    m = wx * y - wy * x
    ld = el * d
    s = -m / ld
    cd = cos(offset)
    sd = sin(offset)
    g = 0.5 * (1.0 + cd * cz + sd * s)
    amp = k * k * fratio
    flux = amp * g

    for kk in range(7):
        dxk = dpx[kk]; dyk = dpy[kk]; dzk = dpz[kk]
        dvxk = dvx[kk]; dvyk = dvy[kk]; dvzk = dvz[kk]
        dd = (x * dxk + y * dyk + z * dzk) / d
        dcz = -dzk / d + z * dd / d2
        dwx = dyk * vz + y * dvzk - dzk * vy - z * dvyk
        dwy = dzk * vx + z * dvxk - dxk * vz - x * dvzk
        dwz = dxk * vy + x * dvyk - dyk * vx - y * dvxk
        dl = (wx * dwx + wy * dwy + wz * dwz) / el
        dm = dwx * y + wx * dyk - dwy * x - wy * dxk
        ds = -dm / ld + m * (dl * d + el * dd) / (ld * ld)
        dout[kk] = amp * 0.5 * (cd * dcz + sd * ds)
    dout[7] = 2.0 * k * fratio * g                # d/dk
    dout[8] = k * k * g                           # d/d(fratio)
    dout[9] = amp * 0.5 * (-sd * cz + cd * s)      # d/d(offset)
    return flux


@njit(fastmath=True)
def _emission_phase_curve_cd_s(time, k, fratio, offset, c, dc):
    """Scalar kernel for :func:`emission_phase_curve_cd`. See that function for documentation."""
    dout = zeros(10)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    flux = _emission_phase_curve_cd_w(time, k, fratio, offset, c, dc, dout,
                                      dpx, dpy, dpz, dvx, dvy, dvz)
    return flux, dout


@njit(fastmath=True)
def emission_phase_curve_cd_v(time, k, fratio, offset, c, dc):
    """Vector kernel for :func:`emission_phase_curve_cd`. See that function for documentation."""
    n = time.size
    flux = zeros(n)
    dout = zeros((n, 10))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    for j in range(n):
        flux[j] = _emission_phase_curve_cd_w(time[j], k, fratio, offset, c, dc, dout[j],
                                             dpx, dpy, dpz, dvx, dvy, dvz)
    return flux, dout


@njit(fastmath=True, parallel=True)
def emission_phase_curve_cd_vp(time, k, fratio, offset, c, dc):
    """Parallel (prange) twin of :func:`emission_phase_curve_cd_v`.

    Explicit twin rather than a dual-decorated shared body: the position- and
    velocity-gradient scratch is hoisted per thread here
    (``zeros((get_num_threads(), 7))``, indexed with ``get_thread_id()``),
    while the serial kernel keeps its cheaper single hoisted buffers -
    a shared buffer would be a data race under ``prange``.
    """
    n = time.size
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
        flux[j] = _emission_phase_curve_cd_w(time[j], k, fratio, offset, c, dc, dout[j],
                                             dpx[tid], dpy[tid], dpz[tid],
                                             dvx[tid], dvy[tid], dvz[tid])
    return flux, dout


def emission_phase_curve_cd(time: float | NDArray, k: float, fratio: float, offset: float,
                            c: NDArray, dc: NDArray):
    """
    Evaluate the cosine emission phase-curve flux and its parameter derivatives at an expansion-point-centered time.

    Derivative-returning counterpart of `emission.emission_phase_curve_c`:
    forms :math:`F = k^2 f_\\mathrm{ratio} (1 + \\cos\\delta\\,c_z +
    \\sin\\delta\\,s)/2` and propagates the chain rule through the position
    and velocity (the signed in-plane component depends on the orbital
    normal :math:`w = r\\times v`).

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    fratio : float
        Dayside-to-nightside per-surface-element flux ratio (amplitude
        scaling).
    offset : float
        Hotspot offset [radians].
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    flux : float or ndarray
        Emitted planet-to-star flux ratio. Shape (N,) for an array `time`.
    dflux : NDArray
        Partial derivatives of `flux` with respect to
        `(tc, p, a, i, e, w, lan, k, fratio, offset)`. Shape (10,) for a
        scalar `time`, (N, 10) for an array `time`.
    """
    if isinstance(time, ndarray):
        return emission_phase_curve_cd_v(time, k, fratio, offset, c, dc)
    return _emission_phase_curve_cd_s(time, k, fratio, offset, c, dc)


@overload(emission_phase_curve_cd, jit_options={'fastmath': True})
def _emission_phase_curve_cd_overload(time, k, fratio, offset, c, dc):
    if _is_1d_array(time):
        def impl(time, k, fratio, offset, c, dc):
            return emission_phase_curve_cd_v(time, k, fratio, offset, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, k, fratio, offset, c, dc):
            return _emission_phase_curve_cd_s(time, k, fratio, offset, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _emission_phase_curve_d_s(time, k, fratio, offset, tc, p, c, dc, te):
    """Scalar kernel for :func:`emission_phase_curve_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _emission_phase_curve_cd_s(time - (tc + te + epoch * p), k, fratio, offset, c, dc)


@njit(fastmath=True)
def emission_phase_curve_d_v(time, k, fratio, offset, tc, p, c, dc, te):
    """Vector kernel for :func:`emission_phase_curve_d`. See that function for documentation."""
    n = time.size
    flux = zeros(n)
    dout = zeros((n, 10))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    for j in range(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        flux[j] = _emission_phase_curve_cd_w(time[j] - (tc + te + epoch * p), k, fratio, offset, c, dc,
                                             dout[j], dpx, dpy, dpz, dvx, dvy, dvz)
    return flux, dout


@njit(fastmath=True, parallel=True)
def emission_phase_curve_d_vp(time, k, fratio, offset, tc, p, c, dc, te):
    """Parallel (prange) twin of :func:`emission_phase_curve_d_v`.

    Explicit twin with per-thread position- and velocity-gradient scratch;
    see :func:`emission_phase_curve_cd_vp`.
    """
    n = time.size
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
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        flux[j] = _emission_phase_curve_cd_w(time[j] - (tc + te + epoch * p), k, fratio, offset, c, dc,
                                             dout[j], dpx[tid], dpy[tid], dpz[tid],
                                             dvx[tid], dvy[tid], dvz[tid])
    return flux, dout


def emission_phase_curve_d(time: float | NDArray, k: float, fratio: float, offset: float,
                           tc: float, p: float, c: NDArray, dc: NDArray, te: float = 0.0):
    """
    Evaluate the cosine emission phase-curve flux and its parameter derivatives at an absolute time.

    Direct counterpart of `emission_phase_curve_cd`: epoch-folds the absolute
    time `time` around the expansion point and delegates to
    `emission_phase_curve_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Absolute observation time(s) in the same units as `tc` and `p`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    fratio : float
        Dayside-to-nightside per-surface-element flux ratio (amplitude
        scaling).
    offset : float
        Hotspot offset [radians].
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
    flux : float or ndarray
        Emitted planet-to-star flux ratio. Shape (N,) for an array `time`.
    dflux : NDArray
        Partial derivatives of `flux` with respect to
        `(tc, p, a, i, e, w, lan, k, fratio, offset)`. Shape (10,) for a
        scalar `time`, (N, 10) for an array `time`.
    """
    if isinstance(time, ndarray):
        return emission_phase_curve_d_v(time, k, fratio, offset, tc, p, c, dc, te)
    return _emission_phase_curve_d_s(time, k, fratio, offset, tc, p, c, dc, te)


@overload(emission_phase_curve_d, jit_options={'fastmath': True})
def _emission_phase_curve_d_overload(time, k, fratio, offset, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, k, fratio, offset, tc, p, c, dc, te=0.0):
            return emission_phase_curve_d_v(time, k, fratio, offset, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, k, fratio, offset, tc, p, c, dc, te=0.0):
            return _emission_phase_curve_d_s(time, k, fratio, offset, tc, p, c, dc, te)
        return impl
    return None
