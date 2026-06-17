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
(:func:`_lambert_kernel_d`).

The reflected flux uses the instantaneous star-planet distance
``r = sqrt(x^2 + y^2 + z^2)`` for the inverse-square illumination,
``F = (k / r)^2 A_g f(alpha)``. Both ``r`` and the phase angle therefore depend
on all the orbital parameters through the position, so the full orbital
gradient block is assembled here from the position and its parameter
derivatives. The multi-expansion-point ``orbit3dd`` gradient dispatchers delegate
to the routines here.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import floor, pi, sqrt, arccos, zeros, ndarray
from numpy.typing import NDArray

from .position import _pos_cd_w
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
def _lambert_phase_curve_cd_w(time, ag, k, c, dc, dflux, dpx, dpy, dpz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the nine-parameter flux gradient into the caller-provided ``(9,)``
    buffer ``dflux`` and returns the flux. ``dpx``, ``dpy``, ``dpz`` are
    ``(7,)`` scratch buffers for the position gradients (filled by
    :func:`_pos_cd_w`); vector loops allocate them once and reuse them.

    With ``flux = k^2 ag / r^2 * phase``, ``r^2 = x^2 + y^2 + z^2`` and
    ``cos alpha = -z / r``, the orbital block chains through both the phase
    angle and the ``1/r^2`` illumination:

        d(flux)/dtheta = A [ dphase/dc * dcosa/dtheta / r^2
                             - 2 phase (x dx + y dy + z dz) / r^4 ]

    with ``A = k^2 ag``. Gradient order ``(tc, p, a, i, e, w, lan, ag, k)``;
    ``ag`` and ``k`` are the two extra slots.
    """
    px, py, pz = _pos_cd_w(time, c, dc, dpx, dpy, dpz)
    r2 = px * px + py * py + pz * pz
    r = sqrt(r2)
    inv_r = 1.0 / r
    inv_r2 = 1.0 / r2
    inv_r3 = inv_r * inv_r2
    inv_r4 = inv_r2 * inv_r2
    phase, _, dphase_dc = _lambert_kernel_d(-pz * inv_r)
    amp = k * k * ag
    flux = amp * phase * inv_r2
    for kk in range(7):
        s = px * dpx[kk] + py * dpy[kk] + pz * dpz[kk]   # = r * dr/dtheta
        dcosa = -dpz[kk] * inv_r + pz * s * inv_r3        # d(cos alpha)/dtheta
        dflux[kk] = amp * (dphase_dc * dcosa * inv_r2 - 2.0 * phase * s * inv_r4)
    dflux[7] = k * k * phase * inv_r2          # d/d(ag)
    dflux[8] = 2.0 * k * ag * phase * inv_r2   # d/dk
    return flux


@njit(fastmath=True)
def _lambert_phase_curve_cd_s(time, ag, k, c, dc):
    """Scalar kernel for :func:`lambert_phase_curve_cd`. See that function for documentation."""
    dflux = zeros(9)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    flux = _lambert_phase_curve_cd_w(time, ag, k, c, dc, dflux, dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True)
def lambert_phase_curve_cd_v(time, ag, k, c, dc):
    """Vector kernel for :func:`lambert_phase_curve_cd`. See that function for documentation."""
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        flux[j] = _lambert_phase_curve_cd_w(time[j], ag, k, c, dc, dflux[j], dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_cd_vp(time, ag, k, c, dc):
    """Parallel (prange) twin of :func:`lambert_phase_curve_cd_v`.

    Explicit twin rather than a dual-decorated shared body: the
    position-gradient scratch is hoisted per thread here
    (``zeros((get_num_threads(), 7))``, indexed with ``get_thread_id()``),
    while the serial kernel keeps its cheaper single hoisted buffers -
    a shared buffer would be a data race under ``prange``.
    """
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        flux[j] = _lambert_phase_curve_cd_w(time[j], ag, k, c, dc, dflux[j],
                                            dpx[tid], dpy[tid], dpz[tid])
    return flux, dflux


def lambert_phase_curve_cd(time: float | NDArray, ag: float, k: float, c: NDArray, dc: NDArray):
    """
    Evaluate the Lambertian phase-curve flux and its parameter derivatives at an expansion-point-centered time.

    Derivative-returning counterpart of `lambert.lambert_phase_curve_c`:
    forms the flux :math:`F = (k/r)^2\\, A_g\\, f(\\alpha)` and propagates
    the chain rule through both the phase angle and the inverse-square
    illumination :math:`1/r^2`, with :math:`r` the instantaneous
    star-planet distance in stellar radii.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    ag : float
        Geometric albedo.
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

    Notes
    -----
    Because the illumination uses the instantaneous distance, the
    semi-major axis enters only through the Taylor coefficients (and hence
    through `r`); there is no separate `a` argument. The `a` slot (index 2)
    of the gradient captures `r`'s dependence on the semi-major axis.
    """
    if isinstance(time, ndarray):
        return lambert_phase_curve_cd_v(time, ag, k, c, dc)
    return _lambert_phase_curve_cd_s(time, ag, k, c, dc)


@overload(lambert_phase_curve_cd, jit_options={'fastmath': True})
def _lambert_phase_curve_cd_overload(time, ag, k, c, dc):
    if _is_1d_array(time):
        def impl(time, ag, k, c, dc):
            return lambert_phase_curve_cd_v(time, ag, k, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, ag, k, c, dc):
            return _lambert_phase_curve_cd_s(time, ag, k, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _lambert_phase_curve_d_s(time, ag, k, tc, p, c, dc, te):
    """Scalar kernel for :func:`lambert_phase_curve_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _lambert_phase_curve_cd_s(time - (tc + te + epoch * p), ag, k, c, dc)


@njit(fastmath=True)
def lambert_phase_curve_d_v(time, ag, k, tc, p, c, dc, te):
    """Vector kernel for :func:`lambert_phase_curve_d`. See that function for documentation."""
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        flux[j] = _lambert_phase_curve_cd_w(time[j] - (tc + te + epoch * p), ag, k, c, dc,
                                            dflux[j], dpx, dpy, dpz)
    return flux, dflux


@njit(fastmath=True, parallel=True)
def lambert_phase_curve_d_vp(time, ag, k, tc, p, c, dc, te):
    """Parallel (prange) twin of :func:`lambert_phase_curve_d_v`.

    Explicit twin with per-thread position-gradient scratch; see
    :func:`lambert_phase_curve_cd_vp`.
    """
    n = time.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        flux[j] = _lambert_phase_curve_cd_w(time[j] - (tc + te + epoch * p), ag, k, c, dc,
                                            dflux[j], dpx[tid], dpy[tid], dpz[tid])
    return flux, dflux


def lambert_phase_curve_d(time: float | NDArray, ag: float, k: float, tc: float, p: float,
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
        return lambert_phase_curve_d_v(time, ag, k, tc, p, c, dc, te)
    return _lambert_phase_curve_d_s(time, ag, k, tc, p, c, dc, te)


@overload(lambert_phase_curve_d, jit_options={'fastmath': True})
def _lambert_phase_curve_d_overload(time, ag, k, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, ag, k, tc, p, c, dc, te=0.0):
            return lambert_phase_curve_d_v(time, ag, k, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, ag, k, tc, p, c, dc, te=0.0):
            return _lambert_phase_curve_d_s(time, ag, k, tc, p, c, dc, te)
        return impl
    return None
