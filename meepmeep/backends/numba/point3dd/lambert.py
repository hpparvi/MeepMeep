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

"""Single-expansion-point Lambertian phase-curve evaluators with parameter derivatives.

Derivative-returning counterpart of ``point3d.lambert``. Holds the Lambertian
reflected-light phase curve with gradients (:func:`lambert_phase_curve_cd` /
:func:`lambert_phase_curve_d`) and its shared phase kernel
(:func:`_lambert_kernel_d`). The phase-angle cosine and its gradient come from
the ``cos_phase_angle`` write-into kernel; the multi-expansion-point ``orbit3dd``
gradient dispatchers delegate to the routines here.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import floor, pi, sqrt, arccos, zeros, ndarray
from numpy.typing import NDArray

from .cos_phase_angle import _cos_alpha_cd_w
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


@njit(fastmath=True, inline='always')
def _lambert_phase_curve_cd_w(time, ag, a, k, c, dc, dflux, dca, dpx, dpy, dpz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the nine-parameter flux gradient into the caller-provided ``(9,)``
    buffer ``dflux`` and returns the flux. ``dca`` is a ``(7,)`` scratch
    buffer for the phase-angle gradient and ``dpx``, ``dpy``, ``dpz`` are
    ``(7,)`` scratch buffers for the position gradients consumed by
    :func:`_cos_alpha_cd_w`; vector loops allocate them once and reuse them.

    Gradient order ``(tc, p, a, i, e, w, lan, ag, k)``: the orbital block
    chains through the phase angle, the ``a`` slot also picks up the
    amplitude derivative, and ``ag`` / ``k`` are the two extra slots.
    """
    amplitude = k * k * ag / (a * a)
    ca = _cos_alpha_cd_w(time, c, dc, dca, dpx, dpy, dpz)
    phase, _, dphase_dc = _lambert_kernel_d(ca)
    flux = amplitude * phase
    # Orbital block (0..6): chain through cos_alpha.
    for kk in range(7):
        dflux[kk] = amplitude * dphase_dc * dca[kk]
    # d(amplitude)/da contribution to the `a` slot (index 2): damplitude/da = -2 k^2 ag / a^3.
    dflux[2] += -2.0 * k * k * ag / (a * a * a) * phase
    # Extras: ag (index 7), k (index 8).
    dflux[7] = (k * k / (a * a)) * phase
    dflux[8] = (2.0 * k * ag / (a * a)) * phase
    return flux


@njit(fastmath=True)
def _lambert_phase_curve_cd_s(time, ag, a, k, c, dc):
    """Scalar kernel for :func:`lambert_phase_curve_cd`. See that function for documentation."""
    dflux = zeros(9)
    dca = zeros(7)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    flux = _lambert_phase_curve_cd_w(time, ag, a, k, c, dc, dflux, dca, dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True)
def lambert_phase_curve_cd_v(time, ag, a, k, c, dc):
    """Vector kernel for :func:`lambert_phase_curve_cd`. See that function for documentation."""
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    dca = zeros(7)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        flux[j] = _lambert_phase_curve_cd_w(time[j], ag, a, k, c, dc, dflux[j], dca, dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_cd_vp(time, ag, a, k, c, dc):
    """Parallel (prange) twin of :func:`lambert_phase_curve_cd_v`.

    Explicit twin rather than a dual-decorated shared body: the phase-angle
    and position-gradient scratch is hoisted per thread here
    (``zeros((get_num_threads(), 7))``, indexed with ``get_thread_id()``),
    while the serial kernel keeps its cheaper single hoisted buffers -
    a shared buffer would be a data race under ``prange``.
    """
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    nt = get_num_threads()
    dca = zeros((nt, 7))
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        flux[j] = _lambert_phase_curve_cd_w(time[j], ag, a, k, c, dc, dflux[j],
                                            dca[tid], dpx[tid], dpy[tid], dpz[tid])
    return flux, dflux


def lambert_phase_curve_cd(time: float | NDArray, ag: float, a: float, k: float, c: NDArray, dc: NDArray):
    """
    Evaluate the Lambertian phase-curve flux and its parameter derivatives at an expansion-point-centered time.

    Derivative-returning counterpart of `lambert.lambert_phase_curve_c`:
    forms the flux :math:`F = (k/a)^2\\, A_g\\, f(\\alpha)` and propagates
    the chain rule through the phase angle and the amplitude.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    flux : float or ndarray
        Reflected planet-to-star flux ratio. Shape (N,) for an array `time`.
    dflux : NDArray
        Partial derivatives of `flux` with respect to
        `(tc, p, a, i, e, w, lan, ag, k)`. Shape (9,) for a scalar `time`,
        (N, 9) for an array `time`.
    """
    if isinstance(time, ndarray):
        return lambert_phase_curve_cd_v(time, ag, a, k, c, dc)
    return _lambert_phase_curve_cd_s(time, ag, a, k, c, dc)


@overload(lambert_phase_curve_cd, jit_options={'fastmath': True})
def _lambert_phase_curve_cd_overload(time, ag, a, k, c, dc):
    if _is_1d_array(time):
        def impl(time, ag, a, k, c, dc):
            return lambert_phase_curve_cd_v(time, ag, a, k, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, ag, a, k, c, dc):
            return _lambert_phase_curve_cd_s(time, ag, a, k, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _lambert_phase_curve_d_s(time, ag, a, k, tc, p, c, dc, te):
    """Scalar kernel for :func:`lambert_phase_curve_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _lambert_phase_curve_cd_s(time - (tc + te + epoch * p), ag, a, k, c, dc)


@njit(fastmath=True)
def lambert_phase_curve_d_v(time, ag, a, k, tc, p, c, dc, te):
    """Vector kernel for :func:`lambert_phase_curve_d`. See that function for documentation."""
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    dca = zeros(7)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        flux[j] = _lambert_phase_curve_cd_w(time[j] - (tc + te + epoch * p), ag, a, k, c, dc,
                                            dflux[j], dca, dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_d_vp(time, ag, a, k, tc, p, c, dc, te):
    """Parallel (prange) twin of :func:`lambert_phase_curve_d_v`.

    Explicit twin with per-thread phase-angle and position-gradient scratch;
    see :func:`lambert_phase_curve_cd_vp`.
    """
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    nt = get_num_threads()
    dca = zeros((nt, 7))
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        flux[j] = _lambert_phase_curve_cd_w(time[j] - (tc + te + epoch * p), ag, a, k, c, dc,
                                            dflux[j], dca[tid], dpx[tid], dpy[tid], dpz[tid])
    return flux, dflux


def lambert_phase_curve_d(time: float | NDArray, ag: float, a: float, k: float, tc: float, p: float,
                          c: NDArray, dc: NDArray, te: float = 0.0):
    """
    Evaluate the Lambertian phase-curve flux and its parameter derivatives at an absolute time.

    Direct counterpart of `lambert_phase_curve_cd`: epoch-folds the absolute
    time `time` around the expansion point and delegates to
    `lambert_phase_curve_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Absolute observation time(s) in the same units as `tc` and `p`.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
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
        Reflected planet-to-star flux ratio. Shape (N,) for an array `time`.
    dflux : NDArray
        Partial derivatives of `flux` with respect to
        `(tc, p, a, i, e, w, lan, ag, k)`. Shape (9,) for a scalar `time`,
        (N, 9) for an array `time`.
    """
    if isinstance(time, ndarray):
        return lambert_phase_curve_d_v(time, ag, a, k, tc, p, c, dc, te)
    return _lambert_phase_curve_d_s(time, ag, a, k, tc, p, c, dc, te)


@overload(lambert_phase_curve_d, jit_options={'fastmath': True})
def _lambert_phase_curve_d_overload(time, ag, a, k, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, ag, a, k, tc, p, c, dc, te=0.0):
            return lambert_phase_curve_d_v(time, ag, a, k, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, ag, a, k, tc, p, c, dc, te=0.0):
            return _lambert_phase_curve_d_s(time, ag, a, k, tc, p, c, dc, te)
        return impl
    return None
