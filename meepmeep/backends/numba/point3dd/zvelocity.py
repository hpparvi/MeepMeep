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

"""Single-knot 3D line-of-sight (z) velocity evaluators with parameter derivatives."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _zvel_cd_w(time, c, dc, dvz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter gradient into the caller-provided ``(7,)``
    buffer ``dvz`` and returns the z velocity, so the hot vector loops
    reuse preallocated rows instead of allocating per sample.
    """
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))
    for k in range(7):
        dvz[k] = dc[k, 2, 1] + time * (2.0 * dc[k, 2, 2] + time * (3.0 * dc[k, 2, 3] + time * 4.0 * dc[k, 2, 4]))
    return vz


@njit(fastmath=True)
def _zvel_cd_s(time, c, dc):
    """Scalar kernel for :func:`zvel_cd`. See that function for documentation."""
    dvz = zeros(7)
    vz = _zvel_cd_w(time, c, dc, dvz)
    return vz, dvz


def _zvel_cd_v_body(time, c, dc):
    """Vector-kernel body for :func:`zvel_cd`; see that function for documentation.

    Compiled twice: ``_zvel_cd_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_zvel_cd_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    vz = zeros(n)
    dvz = zeros((n, 7))
    for j in prange(n):
        vz[j] = _zvel_cd_w(time[j], c, dc, dvz[j])
    return vz, dvz


_zvel_cd_v = njit(fastmath=True)(_zvel_cd_v_body)
_zvel_cd_vp = njit(fastmath=True, parallel=True)(_zvel_cd_v_body)


def zvel_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the line-of-sight velocity and its parameter derivatives at a knot-centered time.

    Centered companion to `velocity.zvel_c` that additionally
    returns the partial derivatives of the line-of-sight velocity
    with respect to each of the seven orbital parameters. Only the
    z-direction polynomials are evaluated; the x and y rows of `c`
    and `dc` are not read.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `velocity.zvel_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`, with the leading axis ordered as
        `(tc, p, a, i, e, w, lan)`. Only the slice `dc[:, 2, :]` is read.

    Returns
    -------
    vz : float or ndarray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer. Shape (N,)
        for an array `time`.
    dvz : NDArray
        Partial derivatives of `vz` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return _zvel_cd_v(time, c, dc)
    return _zvel_cd_s(time, c, dc)


@overload(zvel_cd, jit_options={'fastmath': True})
def _zvel_cd_overload(time, c, dc):
    if _is_1d_array(time):
        def impl(time, c, dc):
            return _zvel_cd_v(time, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c, dc):
            return _zvel_cd_s(time, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _zvel_d_s(time, tc, p, c, dc, tk):
    """Scalar kernel for :func:`zvel_d`. See that function for documentation."""
    epoch = floor((time - tc - tk + 0.5 * p) / p)
    return _zvel_cd_s(time - (tc + tk + epoch * p), c, dc)


def _zvel_d_v_body(time, tc, p, c, dc, tk):
    """Vector-kernel body for :func:`zvel_d`; see that function for documentation.

    Compiled twice: ``_zvel_d_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_zvel_d_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    vz = zeros(n)
    dvz = zeros((n, 7))
    for j in prange(n):
        epoch = floor((time[j] - tc - tk + 0.5 * p) / p)
        vz[j] = _zvel_cd_w(time[j] - (tc + tk + epoch * p), c, dc, dvz[j])
    return vz, dvz


_zvel_d_v = njit(fastmath=True)(_zvel_d_v_body)
_zvel_d_vp = njit(fastmath=True, parallel=True)(_zvel_d_v_body)


def zvel_d(time: float | NDArray, tc: float, p: float, c: NDArray, dc: NDArray, tk: float = 0.0):
    """
    Evaluate the line-of-sight velocity and its parameter derivatives at an absolute time.

    Direct counterpart of `zvel_cd`: epoch-folds the absolute time
    `time` around the expansion point and delegates to
    `zvel_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `velocity.zvel`.

    Parameters
    ----------
    time : float or ndarray
        Absolute observation time(s) in the same units as `tc` and `p`.
    tc : float
        Transit-centre time (time of inferior conjunction), on the same
        time axis as `time`.
    tk : float, optional
        Knot offset from the transit centre [days] - the same value that
        was passed to `solve3d_d`. Defaults to 0.0, the knot at the
        transit centre.
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`. Only the slice `dc[:, 2, :]` is read.

    Returns
    -------
    vz : float or ndarray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer. Shape (N,)
        for an array `time`.
    dvz : NDArray
        Partial derivatives of `vz` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return _zvel_d_v(time, tc, p, c, dc, tk)
    return _zvel_d_s(time, tc, p, c, dc, tk)


@overload(zvel_d, jit_options={'fastmath': True})
def _zvel_d_overload(time, tc, p, c, dc, tk=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, dc, tk=0.0):
            return _zvel_d_v(time, tc, p, c, dc, tk)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, dc, tk=0.0):
            return _zvel_d_s(time, tc, p, c, dc, tk)
        return impl
    return None
