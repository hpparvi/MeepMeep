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

"""Single-knot 3D position evaluators with orbital-parameter derivatives."""

from numba import njit
from numpy import floor, zeros
from numpy.typing import NDArray


@njit(fastmath=True)
def pos_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the (x, y, z) position and its orbital-parameter derivatives at a knot-centered time.

    Centered companion to `position.pos_c` that additionally returns
    the partial derivatives of the sky-frame position with respect to
    each of the seven orbital parameters. Both the position polynomials
    and the eighteen derivative polynomials are evaluated using Horner's
    scheme on the same centered time `time`.

    Parameters
    ----------
    time : float
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Rows index
        the spatial dimensions (x, y, z) and columns the Taylor order
        from position through snap (pre-scaled by the factorial of the
        order).
    dc : NDArray
        A (7, 3, 5) tensor of parameter-derivative coefficients produced
        by `solve3d_d`. The leading axis enumerates the seven Keplerian
        parameters in the canonical order `(tc, p, a, i, e, w, lan)`; the
        remaining axes mirror the layout of `c`.

    Returns
    -------
    px : float
        Sky-plane x position in units of stellar radii.
    py : float
        Sky-plane y position in units of stellar radii.
    pz : float
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.
    dpx : NDArray
        Shape (7,) array of partial derivatives of `px` with respect to
        `(tc, p, a, i, e, w, lan)`, in that order.
    dpy : NDArray
        Shape (7,) array of partial derivatives of `py` with respect to
        the same seven parameters.
    dpz : NDArray
        Shape (7,) array of partial derivatives of `pz` with respect to
        the same seven parameters.

    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    pz = c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))

    dpx = zeros(7)
    dpy = zeros(7)
    dpz = zeros(7)
    for k in range(7):
        dpx[k] = dc[k, 0, 0] + time * (dc[k, 0, 1] + time * (dc[k, 0, 2] + time * (dc[k, 0, 3] + time * dc[k, 0, 4])))
        dpy[k] = dc[k, 1, 0] + time * (dc[k, 1, 1] + time * (dc[k, 1, 2] + time * (dc[k, 1, 3] + time * dc[k, 1, 4])))
        dpz[k] = dc[k, 2, 0] + time * (dc[k, 2, 1] + time * (dc[k, 2, 2] + time * (dc[k, 2, 3] + time * dc[k, 2, 4])))

    return px, py, pz, dpx, dpy, dpz


@njit(fastmath=True)
def pos_d(time: float | NDArray, tk: float, p: float, c: NDArray, dc: NDArray):
    """
    Evaluate the (x, y, z) position and its orbital-parameter derivatives at an absolute time.

    Direct counterpart of `pos_cd`: accepts an absolute observation time
    `time`, folds it back into a single orbital epoch around the
    expansion time `tk`, and delegates the polynomial evaluation to
    `pos_cd`.

    Parameters
    ----------
    time : float
        Absolute observation time in the same units as `tk` and `p`.
    tk : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    px : float
        Sky-plane x position in units of stellar radii.
    py : float
        Sky-plane y position in units of stellar radii.
    pz : float
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.
    dpx : NDArray
        Shape (7,) partial derivatives of `px` w.r.t. `(tc, p, a, i, e, w, lan)`.
    dpy : NDArray
        Shape (7,) partial derivatives of `py` w.r.t. `(tc, p, a, i, e, w, lan)`.
    dpz : NDArray
        Shape (7,) partial derivatives of `pz` w.r.t. `(tc, p, a, i, e, w, lan)`.

    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return pos_cd(time - (tk + epoch * p), c, dc)
