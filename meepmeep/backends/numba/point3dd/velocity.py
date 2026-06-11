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

"""Single-knot 3D velocity evaluators with orbital-parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _vel_cd_w(time, c, dc, dvx, dvy, dvz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter gradients into the caller-provided ``(7,)``
    buffers ``dvx``, ``dvy``, and ``dvz`` and returns the velocity values,
    so the hot vector loops reuse preallocated rows instead of allocating
    per sample.
    """
    vx = c[0, 1] + time * (2.0 * c[0, 2] + time * (3.0 * c[0, 3] + time * 4.0 * c[0, 4]))
    vy = c[1, 1] + time * (2.0 * c[1, 2] + time * (3.0 * c[1, 3] + time * 4.0 * c[1, 4]))
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))
    for k in range(7):
        dvx[k] = dc[k, 0, 1] + time * (2.0 * dc[k, 0, 2] + time * (3.0 * dc[k, 0, 3] + time * 4.0 * dc[k, 0, 4]))
        dvy[k] = dc[k, 1, 1] + time * (2.0 * dc[k, 1, 2] + time * (3.0 * dc[k, 1, 3] + time * 4.0 * dc[k, 1, 4]))
        dvz[k] = dc[k, 2, 1] + time * (2.0 * dc[k, 2, 2] + time * (3.0 * dc[k, 2, 3] + time * 4.0 * dc[k, 2, 4]))
    return vx, vy, vz


@njit(fastmath=True)
def _vel_cd_s(time, c, dc):
    """Scalar kernel for :func:`vel_cd`. See that function for documentation."""
    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    vx, vy, vz = _vel_cd_w(time, c, dc, dvx, dvy, dvz)
    return vx, vy, vz, dvx, dvy, dvz


@njit(fastmath=True)
def _vel_cd_v(time, c, dc):
    """Vector kernel for :func:`vel_cd`. See that function for documentation."""
    n = time.size
    vx = zeros(n)
    vy = zeros(n)
    vz = zeros(n)
    dvx = zeros((n, 7))
    dvy = zeros((n, 7))
    dvz = zeros((n, 7))
    for j in range(n):
        vx[j], vy[j], vz[j] = _vel_cd_w(time[j], c, dc, dvx[j], dvy[j], dvz[j])
    return vx, vy, vz, dvx, dvy, dvz


def vel_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the (vx, vy, vz) velocity and its orbital-parameter derivatives at a knot-centered time.

    Centered velocity companion to `position.pos_cd`. The velocity
    components are obtained by analytically differentiating the
    5th-order position polynomials, yielding 4th-order polynomials in
    `time` that are evaluated using Horner's scheme. The same
    differentiation is applied to the parameter-derivative
    coefficients so the result is the velocity together with its seven
    partial derivatives with respect to `(tc, p, a, i, e, w, lan)`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `velocity.vel_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Rows index
        the spatial dimensions (x, y, z) and columns the Taylor order
        from position through snap (pre-scaled by the factorial of the
        order).
    dc : NDArray
        A (7, 3, 5) tensor of parameter-derivative coefficients
        produced by `solve3d_d`. The leading axis enumerates the seven
        Keplerian parameters in the canonical order `(tc, p, a, i, e, w, lan)`;
        the remaining axes mirror the layout of `c`.

    Returns
    -------
    vx : float or ndarray
        Sky-plane x velocity in stellar radii per unit time. Shape (N,)
        for an array `time`.
    vy : float or ndarray
        Sky-plane y velocity in stellar radii per unit time. Shape (N,)
        for an array `time`.
    vz : float or ndarray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer. Shape (N,)
        for an array `time`.
    dvx : NDArray
        Partial derivatives of `vx` w.r.t. `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    dvy : NDArray
        Partial derivatives of `vy` w.r.t. `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    dvz : NDArray
        Partial derivatives of `vz` w.r.t. `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.

    Notes
    -----
    The pre-factors `1, 2, 3, 4` in front of `c[d, 1..4]` and
    `dc[k, d, 1..4]` are the chain-rule factors from differentiating
    `c[d, n] * time^n` (and likewise the derivative coefficients) with
    respect to `time`. Differentiation drops the truncation order by
    one, so the velocity polynomials are 4th order even though the
    underlying position expansion is 5th order.
    """
    if isinstance(time, ndarray):
        return _vel_cd_v(time, c, dc)
    return _vel_cd_s(time, c, dc)


@overload(vel_cd, jit_options={'fastmath': True})
def _vel_cd_overload(time, c, dc):
    if _is_1d_array(time):
        def impl(time, c, dc):
            return _vel_cd_v(time, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c, dc):
            return _vel_cd_s(time, c, dc)
        return impl
    return None
