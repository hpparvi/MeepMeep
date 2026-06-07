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

"""Single-knot 2D projected-separation evaluators with parameter derivatives."""

from numba import njit
from numpy import floor, sqrt, zeros
from numpy.typing import NDArray

from .position import pos_cd, pos_d


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
        A (7, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    d : float
        Projected planet-star center distance in units of stellar radii.
    dd : NDArray
        Shape (7,) partial derivatives of `d` with respect to
        `(tc, p, a, i, e, w, lan)`.

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
    dd = zeros(7)
    for k in range(7):
        dd[k] = (px * dpx[k] + py * dpy[k]) / d
    return d, dd


@njit(fastmath=True)
def sep_d(time: float | NDArray, tk: float, p: float, c: NDArray, dc: NDArray):
    """
    Evaluate the projected planet-star distance and its parameter derivatives at an absolute time.

    Direct counterpart of `sep_cd`: epoch-folds the absolute time `time`
    around the expansion point `tk` and delegates to `sep_cd`.

    Parameters
    ----------
    time : float
        Absolute observation time in the same units as `tk` and `p`.
    tk : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (7, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    d : float
        Projected planet-star center distance in units of stellar radii.
    dd : NDArray
        Shape (7,) partial derivatives of `d` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return sep_cd(time - (tk + epoch * p), c, dc)


@njit(fastmath=True)
def sep_dv(time: NDArray, tk: float, p: float, c: NDArray, dc: NDArray):
    """
    Evaluate the projected separation and its parameter derivatives over a 1-D time array.

    Vectorised counterpart of `sep_d`: applies the scalar direct evaluator at
    each element of `time` and stacks the results. The scalar `sep_d` allocates
    a length-7 gradient per call, so it cannot accept an array directly; this
    wrapper supplies the array path that array callers (e.g. the high-level
    ``Knot2D`` properties) need.

    Parameters
    ----------
    time : NDArray
        Absolute observation times, shape (N,), in the same units as `tk`
        and `p`.
    tk : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (7, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    d : NDArray
        Projected planet-star center distances, shape (N,), in units of
        stellar radii.
    dd : NDArray
        Shape (N, 7) partial derivatives of `d` w.r.t. `(tc, p, a, i, e, w, lan)`.
    """
    n = time.size
    d = zeros(n)
    dd = zeros((n, 7))
    for j in range(n):
        d[j], dd[j] = sep_d(time[j], tk, p, c, dc)
    return d, dd
