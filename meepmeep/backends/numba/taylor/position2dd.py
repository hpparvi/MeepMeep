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

from numba import njit
from numpy import floor, sqrt, zeros
from numpy.typing import NDArray


@njit(fastmath=True)
def pos_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the (x, y) position and its orbital-parameter derivatives at a knot-centered time.

    Centered companion to `position2d.pos_c` that additionally returns the
    partial derivatives of the sky-plane position with respect to each of
    the six orbital parameters. Both the position polynomial and the six
    derivative polynomials are evaluated using Horner's scheme on the same
    centered time `time`.

    Parameters
    ----------
    time : float
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`. Rows index the
        spatial dimensions (x, y) and columns the Taylor order from
        position through snap (pre-scaled by the factorial of the order).
    dc : NDArray
        A (6, 2, 5) tensor of parameter-derivative coefficients produced
        by `solve2d_d`. The leading axis enumerates the six Keplerian
        parameters in the canonical order `(t0, p, a, i, e, w)`; the
        remaining axes mirror the layout of `c`.

    Returns
    -------
    px : float
        Sky-plane x position in units of stellar radii.
    py : float
        Sky-plane y position in units of stellar radii.
    dpx : NDArray
        Shape (6,) array of partial derivatives of `px` with respect to
        `(t0, p, a, i, e, w)`, in that order.
    dpy : NDArray
        Shape (6,) array of partial derivatives of `py` with respect to
        the same six parameters.

    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))

    dpx = zeros(6)
    dpy = zeros(6)
    for k in range(6):
        dpx[k] = dc[k, 0, 0] + time * (dc[k, 0, 1] + time * (dc[k, 0, 2] + time * (dc[k, 0, 3] + time * dc[k, 0, 4])))
        dpy[k] = dc[k, 1, 0] + time * (dc[k, 1, 1] + time * (dc[k, 1, 2] + time * (dc[k, 1, 3] + time * dc[k, 1, 4])))

    return px, py, dpx, dpy


@njit(fastmath=True)
def pos_d(time: float | NDArray, t0: float, p: float, c: NDArray, dc: NDArray):
    """
    Evaluate the (x, y) position and its orbital-parameter derivatives at an absolute time.

    Direct counterpart of `pos_cd`: accepts an absolute observation time
    `time`, folds it back into a single orbital epoch around the expansion
    time `t0`, and delegates the polynomial evaluation to `pos_cd`.

    Parameters
    ----------
    time : float
        Absolute observation time in the same units as `t0` and `p`.
    t0 : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (6, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(t0, p, a, i, e, w)`.

    Returns
    -------
    px : float
        Sky-plane x position in units of stellar radii.
    py : float
        Sky-plane y position in units of stellar radii.
    dpx : NDArray
        Shape (6,) partial derivatives of `px` w.r.t. `(t0, p, a, i, e, w)`.
    dpy : NDArray
        Shape (6,) partial derivatives of `py` w.r.t. `(t0, p, a, i, e, w)`.

    """
    epoch = floor((time - t0 + 0.5 * p) / p)
    return pos_cd(time - (t0 + epoch * p), c, dc)


@njit(fastmath=True)
def sep_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the projected planet-star separation and its parameter derivatives at a knot-centered time.

    Computes the sky-plane position via `pos_cd` and reduces it to the
    Euclidean distance `d = sqrt(px^2 + py^2)`, propagating the parameter
    derivatives using the chain rule.

    Parameters
    ----------
    time : float
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (6, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(t0, p, a, i, e, w)`.

    Returns
    -------
    d : float
        Projected planet-star center distance in units of stellar radii.
    dd : NDArray
        Shape (6,) partial derivatives of `d` with respect to
        `(t0, p, a, i, e, w)`.

    Notes
    -----
    The chain-rule reduction used here is
    `dd/dtheta = (px * dpx/dtheta + py * dpy/dtheta) / d`.
    The expression is regular for `d > 0`, which is the regime of
    interest for transit modeling; the gradient is ill-defined exactly
    at the (geometrically singular) point of zero projected separation.
    """
    px, py, dpx, dpy = pos_cd(time, c, dc)
    d = sqrt(px ** 2 + py ** 2)
    dd = zeros(6)
    for k in range(6):
        dd[k] = (px * dpx[k] + py * dpy[k]) / d
    return d, dd


@njit(fastmath=True)
def sep_d(time: float | NDArray, t0: float, p: float, c: NDArray, dc: NDArray):
    """
    Evaluate the projected planet-star distance and its parameter derivatives at an absolute time.

    Direct counterpart of `sep_cd`: epoch-folds the absolute time `time`
    around the expansion point `t0` and delegates to `sep_cd`.

    Parameters
    ----------
    time : float
        Absolute observation time in the same units as `t0` and `p`.
    t0 : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (6, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(t0, p, a, i, e, w)`.

    Returns
    -------
    d : float
        Projected planet-star center distance in units of stellar radii.
    dd : NDArray
        Shape (6,) partial derivatives of `d` with respect to
        `(t0, p, a, i, e, w)`.
    """
    epoch = floor((time - t0 + 0.5 * p) / p)
    return sep_cd(time - (t0 + epoch * p), c, dc)
