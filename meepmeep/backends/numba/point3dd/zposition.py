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

"""Single-knot 3D line-of-sight (z) position evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import floor, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _zpos_cd_w(time, c, dc, dpz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter gradient into the caller-provided ``(7,)``
    buffer ``dpz`` and returns the z position, so the hot vector loops
    reuse preallocated rows instead of allocating per sample.
    """
    pz = c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))
    for k in range(7):
        dpz[k] = dc[k, 2, 0] + time * (dc[k, 2, 1] + time * (dc[k, 2, 2] + time * (dc[k, 2, 3] + time * dc[k, 2, 4])))
    return pz


@njit(fastmath=True)
def _zpos_cd_s(time, c, dc):
    """Scalar kernel for :func:`zpos_cd`. See that function for documentation."""
    dpz = zeros(7)
    pz = _zpos_cd_w(time, c, dc, dpz)
    return pz, dpz


@njit(fastmath=True)
def _zpos_cd_v(time, c, dc):
    """Vector kernel for :func:`zpos_cd`. See that function for documentation."""
    n = time.size
    pz = zeros(n)
    dpz = zeros((n, 7))
    for j in range(n):
        pz[j] = _zpos_cd_w(time[j], c, dc, dpz[j])
    return pz, dpz


def zpos_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the line-of-sight z position and its parameter derivatives at a knot-centered time.

    Centered companion to `position.zpos_c` that additionally returns
    the partial derivatives of the line-of-sight coordinate with
    respect to each of the seven orbital parameters. Only the z-direction
    polynomials are evaluated; the x and y rows of `c` and `dc` are not
    read.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `position.zpos_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`. Only
        row 2 (the z-direction coefficients) is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`. Only
        the slice `dc[:, 2, :]` is read.

    Returns
    -------
    pz : float or ndarray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer. Shape (N,) for an array `time`.
    dpz : NDArray
        Partial derivatives of `pz` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return _zpos_cd_v(time, c, dc)
    return _zpos_cd_s(time, c, dc)


@overload(zpos_cd, jit_options={'fastmath': True})
def _zpos_cd_overload(time, c, dc):
    if _is_1d_array(time):
        def impl(time, c, dc):
            return _zpos_cd_v(time, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c, dc):
            return _zpos_cd_s(time, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _zpos_d_s(time, tc, p, c, dc, tk):
    """Scalar kernel for :func:`zpos_d`. See that function for documentation."""
    epoch = floor((time - tc - tk + 0.5 * p) / p)
    return _zpos_cd_s(time - (tc + tk + epoch * p), c, dc)


@njit(fastmath=True)
def _zpos_d_v(time, tc, p, c, dc, tk):
    """Vector kernel for :func:`zpos_d`. See that function for documentation."""
    n = time.size
    pz = zeros(n)
    dpz = zeros((n, 7))
    for j in range(n):
        epoch = floor((time[j] - tc - tk + 0.5 * p) / p)
        pz[j] = _zpos_cd_w(time[j] - (tc + tk + epoch * p), c, dc, dpz[j])
    return pz, dpz


def zpos_d(time: float | NDArray, tc: float, p: float, c: NDArray, dc: NDArray, tk: float = 0.0):
    """
    Evaluate the line-of-sight z position and its parameter derivatives at an absolute time.

    Direct counterpart of `zpos_cd`: epoch-folds the absolute time `time`
    around the expansion point and delegates to `zpos_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `position.zpos`.

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
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    pz : float or ndarray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
        The sign distinguishes the transit (positive z) and eclipse
        (negative z) branches of the orbit. Shape (N,) for an array `time`.
    dpz : NDArray
        Partial derivatives of `pz` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return _zpos_d_v(time, tc, p, c, dc, tk)
    return _zpos_d_s(time, tc, p, c, dc, tk)


@overload(zpos_d, jit_options={'fastmath': True})
def _zpos_d_overload(time, tc, p, c, dc, tk=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, dc, tk=0.0):
            return _zpos_d_v(time, tc, p, c, dc, tk)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, dc, tk=0.0):
            return _zpos_d_s(time, tc, p, c, dc, tk)
        return impl
    return None
