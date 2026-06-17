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

"""Single-expansion-point 3D phase-angle cosine evaluators with parameter derivatives.

Derivative-returning counterpart of ``point3d.cos_phase_angle``. Unlike
``separation``, which collapses the position gradients to scalar
temporaries, the phase-angle cosine needs the full (x, y, z) position and
its three gradients, so the write-into kernel delegates to
``position._pos_cd_w`` and reduces through the chain rule. The position
gradients are therefore intermediate scratch reused across samples, which
makes the vector loops scratch-using: the serial kernels hoist one ``(7,)``
buffer triple and the parallel twins hoist per-thread buffers (explicit
twins rather than a dual-decorated shared body).
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import floor, sqrt, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array
from .position import _pos_cd_w


@njit(fastmath=True, inline='always')
def _cos_alpha_cd_w(time, c, dc, dca, dpx, dpy, dpz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter phase-angle-cosine gradient into the
    caller-provided ``(7,)`` buffer ``dca`` and returns the cosine.
    ``dpx``, ``dpy``, and ``dpz`` are ``(7,)`` scratch buffers for the
    position gradients; vector loops allocate them once and reuse them.
    """
    px, py, pz = _pos_cd_w(time, c, dc, dpx, dpy, dpz)
    r2 = px * px + py * py + pz * pz
    r = sqrt(r2)
    ca = -pz / r
    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    for k in range(7):
        # d(-z/r)/dtheta = -dz/r + z * (x*dx + y*dy + z*dz) / r^3
        dca[k] = -dpz[k] * inv_r + pz * (px * dpx[k] + py * dpy[k] + pz * dpz[k]) * inv_r3
    return ca


@njit(fastmath=True)
def _cos_alpha_cd_s(time, c, dc):
    """Scalar kernel for :func:`cos_alpha_cd`. See that function for documentation."""
    dca = zeros(7)
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    ca = _cos_alpha_cd_w(time, c, dc, dca, dpx, dpy, dpz)
    return ca, dca


@njit(fastmath=True)
def cos_alpha_cd_v(time, c, dc):
    """Vector kernel for :func:`cos_alpha_cd`. See that function for documentation."""
    n = time.size
    ca = zeros(n)
    dca = zeros((n, 7))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        ca[j] = _cos_alpha_cd_w(time[j], c, dc, dca[j], dpx, dpy, dpz)
    return ca, dca


@njit(fastmath=True, parallel=True)
def cos_alpha_cd_vp(time, c, dc):
    """Parallel (prange) twin of :func:`cos_alpha_cd_v`.

    Explicit twin rather than a dual-decorated shared body: the
    position-gradient scratch is hoisted per thread here
    (``zeros((get_num_threads(), 7))``, indexed with ``get_thread_id()``),
    while the serial kernel keeps its cheaper single hoisted buffer -
    one shared buffer would be a data race under ``prange``.
    """
    n = time.size
    ca = zeros(n)
    dca = zeros((n, 7))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        ca[j] = _cos_alpha_cd_w(time[j], c, dc, dca[j], dpx[tid], dpy[tid], dpz[tid])
    return ca, dca


def cos_alpha_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the cosine of the orbital phase angle and its parameter derivatives at an expansion-point-centered time.

    Derivative-returning counterpart of `cos_phase_angle.cos_alpha_c`: forms
    the phase-angle cosine `cos alpha = -z / sqrt(x^2 + y^2 + z^2)` from the
    sky position and propagates the chain rule to its seven orbital-parameter
    partials. The phase angle alpha is the star-planet-observer angle, with
    z positive toward the observer; `cos alpha = +1` at superior conjunction
    (full phase) and `-1` at inferior conjunction (new phase).

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `cos_phase_angle.cos_alpha_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    ca : float or ndarray
        Cosine of the phase angle, in [-1, 1]. Shape (N,) for an array `time`.
    dca : NDArray
        Partial derivatives of `ca` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.

    Notes
    -----
    With `cos alpha = -z/r` and `r = sqrt(x^2 + y^2 + z^2)`, the chain-rule
    reduction used here is
    `d(cos alpha)/dtheta = -dz/dtheta / r + z * (x*dx/dtheta + y*dy/dtheta
    + z*dz/dtheta) / r^3`. The expression is regular for `r > 0`.
    """
    if isinstance(time, ndarray):
        return cos_alpha_cd_v(time, c, dc)
    return _cos_alpha_cd_s(time, c, dc)


@overload(cos_alpha_cd, jit_options={'fastmath': True})
def _cos_alpha_cd_overload(time, c, dc):
    if _is_1d_array(time):
        def impl(time, c, dc):
            return cos_alpha_cd_v(time, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c, dc):
            return _cos_alpha_cd_s(time, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _cos_alpha_d_s(time, tc, p, c, dc, te):
    """Scalar kernel for :func:`cos_alpha_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _cos_alpha_cd_s(time - (tc + te + epoch * p), c, dc)


@njit(fastmath=True)
def cos_alpha_d_v(time, tc, p, c, dc, te):
    """Vector kernel for :func:`cos_alpha_d`. See that function for documentation."""
    n = time.size
    ca = zeros(n)
    dca = zeros((n, 7))
    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for j in range(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        ca[j] = _cos_alpha_cd_w(time[j] - (tc + te + epoch * p), c, dc, dca[j], dpx, dpy, dpz)
    return ca, dca


@njit(fastmath=True, parallel=True)
def cos_alpha_d_vp(time, tc, p, c, dc, te):
    """Parallel (prange) twin of :func:`cos_alpha_d_v`.

    Explicit twin with per-thread position-gradient scratch; see
    :func:`cos_alpha_cd_vp`.
    """
    n = time.size
    ca = zeros(n)
    dca = zeros((n, 7))
    nt = get_num_threads()
    dpx = zeros((nt, 7))
    dpy = zeros((nt, 7))
    dpz = zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        ca[j] = _cos_alpha_cd_w(time[j] - (tc + te + epoch * p), c, dc, dca[j], dpx[tid], dpy[tid], dpz[tid])
    return ca, dca


def cos_alpha_d(time: float | NDArray, tc: float, p: float, c: NDArray, dc: NDArray, te: float = 0.0):
    """
    Evaluate the cosine of the orbital phase angle and its parameter derivatives at an absolute time.

    Direct counterpart of `cos_alpha_cd`: epoch-folds the absolute time
    `time` around the expansion point and delegates to `cos_alpha_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `cos_phase_angle.cos_alpha`.

    Parameters
    ----------
    time : float or ndarray
        Absolute observation time(s) in the same units as `tc` and `p`.
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
    ca : float or ndarray
        Cosine of the phase angle, in [-1, 1]. Shape (N,) for an array `time`.
    dca : NDArray
        Partial derivatives of `ca` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return cos_alpha_d_v(time, tc, p, c, dc, te)
    return _cos_alpha_d_s(time, tc, p, c, dc, te)


@overload(cos_alpha_d, jit_options={'fastmath': True})
def _cos_alpha_d_overload(time, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, dc, te=0.0):
            return cos_alpha_d_v(time, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, dc, te=0.0):
            return _cos_alpha_d_s(time, tc, p, c, dc, te)
        return impl
    return None
