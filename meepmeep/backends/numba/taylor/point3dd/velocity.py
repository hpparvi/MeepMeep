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

from numba import njit
from numpy import zeros
from numpy.typing import NDArray


@njit(fastmath=True)
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

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
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
    vx : float or NDArray
        Sky-plane x velocity in stellar radii per unit time.
    vy : float or NDArray
        Sky-plane y velocity in stellar radii per unit time.
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    dvx : NDArray
        Shape (7,) partial derivatives of `vx` w.r.t. `(tc, p, a, i, e, w, lan)`.
    dvy : NDArray
        Shape (7,) partial derivatives of `vy` w.r.t. `(tc, p, a, i, e, w, lan)`.
    dvz : NDArray
        Shape (7,) partial derivatives of `vz` w.r.t. `(tc, p, a, i, e, w, lan)`.

    Notes
    -----
    The pre-factors `1, 2, 3, 4` in front of `c[d, 1..4]` and
    `dc[k, d, 1..4]` are the chain-rule factors from differentiating
    `c[d, n] * time^n` (and likewise the derivative coefficients) with
    respect to `time`. Differentiation drops the truncation order by
    one, so the velocity polynomials are 4th order even though the
    underlying position expansion is 5th order.
    """
    vx = c[0, 1] + time * (2.0 * c[0, 2] + time * (3.0 * c[0, 3] + time * 4.0 * c[0, 4]))
    vy = c[1, 1] + time * (2.0 * c[1, 2] + time * (3.0 * c[1, 3] + time * 4.0 * c[1, 4]))
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))

    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    for k in range(7):
        dvx[k] = dc[k, 0, 1] + time * (2.0 * dc[k, 0, 2] + time * (3.0 * dc[k, 0, 3] + time * 4.0 * dc[k, 0, 4]))
        dvy[k] = dc[k, 1, 1] + time * (2.0 * dc[k, 1, 2] + time * (3.0 * dc[k, 1, 3] + time * 4.0 * dc[k, 1, 4]))
        dvz[k] = dc[k, 2, 1] + time * (2.0 * dc[k, 2, 2] + time * (3.0 * dc[k, 2, 3] + time * 4.0 * dc[k, 2, 4]))

    return vx, vy, vz, dvx, dvy, dvz
