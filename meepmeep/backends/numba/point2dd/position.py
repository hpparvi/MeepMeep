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

"""Single-expansion-point 2D position evaluators with orbital-parameter derivatives."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _pos_cd_w(time, c, dc, dpx, dpy):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter gradients into the caller-provided ``(7,)``
    buffers ``dpx`` and ``dpy`` and returns the position values. The hot
    vector loops pass preallocated output rows here instead of allocating
    per sample.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    for k in range(7):
        dpx[k] = dc[k, 0, 0] + time * (dc[k, 0, 1] + time * (dc[k, 0, 2] + time * (dc[k, 0, 3] + time * dc[k, 0, 4])))
        dpy[k] = dc[k, 1, 0] + time * (dc[k, 1, 1] + time * (dc[k, 1, 2] + time * (dc[k, 1, 3] + time * dc[k, 1, 4])))
    return px, py


@njit(fastmath=True)
def _pos_cd_s(time, c, dc):
    """Scalar kernel for :func:`pos_cd`. See that function for documentation."""
    dpx = zeros(7)
    dpy = zeros(7)
    px, py = _pos_cd_w(time, c, dc, dpx, dpy)
    return px, py, dpx, dpy


def _pos_cd_v_body(time, c, dc):
    """Vector-kernel body for :func:`pos_cd`; see that function for documentation.

    Compiled twice: ``_pos_cd_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_pos_cd_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    px = zeros(n)
    py = zeros(n)
    dpx = zeros((n, 7))
    dpy = zeros((n, 7))
    for j in prange(n):
        px[j], py[j] = _pos_cd_w(time[j], c, dc, dpx[j], dpy[j])
    return px, py, dpx, dpy


_pos_cd_v = njit(fastmath=True)(_pos_cd_v_body)
_pos_cd_vp = njit(fastmath=True, parallel=True)(_pos_cd_v_body)


def pos_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the (x, y) position and its orbital-parameter derivatives at an expansion-point-centered time.

    Centered companion to `position.pos_c` that additionally returns the
    partial derivatives of the sky-plane position with respect to each of
    the seven orbital parameters. Both the position polynomial and the seven
    derivative polynomials are evaluated using Horner's scheme on the same
    centered time `time`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `position.pos_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`. Rows index the
        spatial dimensions (x, y) and columns the Taylor order from
        position through snap (pre-scaled by the factorial of the order).
    dc : NDArray
        A (7, 2, 5) tensor of parameter-derivative coefficients produced
        by `solve2d_d`. The leading axis enumerates the seven Keplerian
        parameters in the canonical order `(tc, p, a, i, e, w, lan)`; the
        remaining axes mirror the layout of `c`.

    Returns
    -------
    px : float or ndarray
        Sky-plane x position in units of stellar radii. Shape (N,) for an
        array `time`.
    py : float or ndarray
        Sky-plane y position in units of stellar radii. Shape (N,) for an
        array `time`.
    dpx : NDArray
        Partial derivatives of `px` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    dpy : NDArray
        Partial derivatives of `py` with respect to the same seven parameters.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.

    """
    if isinstance(time, ndarray):
        return _pos_cd_v(time, c, dc)
    return _pos_cd_s(time, c, dc)


@overload(pos_cd, jit_options={'fastmath': True})
def _pos_cd_overload(time, c, dc):
    if _is_1d_array(time):
        def impl(time, c, dc):
            return _pos_cd_v(time, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c, dc):
            return _pos_cd_s(time, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _pos_d_s(time, tc, p, c, dc, te):
    """Scalar kernel for :func:`pos_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _pos_cd_s(time - (tc + te + epoch * p), c, dc)


def _pos_d_v_body(time, tc, p, c, dc, te):
    """Vector-kernel body for :func:`pos_d`; see that function for documentation.

    Compiled twice: ``_pos_d_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_pos_d_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    px = zeros(n)
    py = zeros(n)
    dpx = zeros((n, 7))
    dpy = zeros((n, 7))
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        px[j], py[j] = _pos_cd_w(time[j] - (tc + te + epoch * p), c, dc, dpx[j], dpy[j])
    return px, py, dpx, dpy


_pos_d_v = njit(fastmath=True)(_pos_d_v_body)
_pos_d_vp = njit(fastmath=True, parallel=True)(_pos_d_v_body)


def pos_d(time: float | NDArray, tc: float, p: float, c: NDArray, dc: NDArray, te: float = 0.0):
    """
    Evaluate the (x, y) position and its orbital-parameter derivatives at an absolute time.

    Direct counterpart of `pos_cd`: accepts an absolute observation time
    `time`, folds it back into a single orbital epoch around the expansion
    time `te`, and delegates the polynomial evaluation to `pos_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `position.pos`.

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
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (7, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.
    te : float, optional
        Expansion-point offset from the transit centre [days] - the same value that
        was passed to `solve2d_d`. Defaults to 0.0, the expansion point at the
        transit centre.

    Returns
    -------
    px : float or ndarray
        Sky-plane x position in units of stellar radii. Shape (N,) for an
        array `time`.
    py : float or ndarray
        Sky-plane y position in units of stellar radii. Shape (N,) for an
        array `time`.
    dpx : NDArray
        Partial derivatives of `px` w.r.t. `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    dpy : NDArray
        Partial derivatives of `py` w.r.t. `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.

    """
    if isinstance(time, ndarray):
        return _pos_d_v(time, tc, p, c, dc, te)
    return _pos_d_s(time, tc, p, c, dc, te)


@overload(pos_d, jit_options={'fastmath': True})
def _pos_d_overload(time, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, dc, te=0.0):
            return _pos_d_v(time, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, dc, te=0.0):
            return _pos_d_s(time, tc, p, c, dc, te)
        return impl
    return None
