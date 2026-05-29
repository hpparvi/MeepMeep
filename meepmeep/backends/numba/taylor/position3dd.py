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
    Evaluate the (x, y, z) position and its orbital-parameter derivatives at a knot-centered time.

    Centered companion to `position3d.pos_c` that additionally returns
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


@njit(fastmath=True)
def sep_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the sky-projected planet-star separation and its parameter derivatives at a knot-centered time.

    Computes the sky-plane position and its derivatives, then reduces
    them to the projected distance `d = sqrt(px^2 + py^2)` via the
    chain rule. The line-of-sight z coordinate and its derivatives are
    not evaluated, since they do not enter the projected separation.

    Parameters
    ----------
    time : float
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`. Only
        rows 0 and 1 (x and y) are read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`. Only
        slices `dc[:, 0, :]` and `dc[:, 1, :]` are read.

    Returns
    -------
    d : float
        Sky-projected planet-star separation in units of stellar radii.
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

    Unlike the 2D analogue, which delegates to `pos_cd`, this routine
    inlines the px/py polynomial passes to avoid the seven wasted
    Horner evaluations that computing `pz` and its seven derivatives
    would otherwise incur.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    d = sqrt(px ** 2 + py ** 2)

    dd = zeros(7)
    for k in range(7):
        dpx = dc[k, 0, 0] + time * (dc[k, 0, 1] + time * (dc[k, 0, 2] + time * (dc[k, 0, 3] + time * dc[k, 0, 4])))
        dpy = dc[k, 1, 0] + time * (dc[k, 1, 1] + time * (dc[k, 1, 2] + time * (dc[k, 1, 3] + time * dc[k, 1, 4])))
        dd[k] = (px * dpx + py * dpy) / d
    return d, dd


@njit(fastmath=True)
def sep_d(time: float | NDArray, tk: float, p: float, c: NDArray, dc: NDArray):
    """
    Evaluate the sky-projected planet-star separation and its parameter derivatives at an absolute time.

    Direct counterpart of `sep_cd`: epoch-folds the absolute time
    `time` around the expansion point `tk` and delegates to `sep_cd`.

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
    d : float
        Sky-projected planet-star separation in units of stellar radii.
    dd : NDArray
        Shape (7,) partial derivatives of `d` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return sep_cd(time - (tk + epoch * p), c, dc)


@njit(fastmath=True)
def zpos_cd(time: float | NDArray, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight z position and its parameter derivatives at a knot-centered time.

    Centered companion to `position3d.zpos_c` that additionally returns
    the partial derivatives of the line-of-sight coordinate with
    respect to each of the seven orbital parameters. Only the z-direction
    polynomials are evaluated; the x and y rows of `c` and `dc` are not
    read.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) Taylor coefficient matrix produced by `solve3d`. Only
        row 2 (the z-direction coefficients) is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by `solve3d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`. Only
        the slice `dc[:, 2, :]` is read.

    Returns
    -------
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.
    dpz : NDArray
        Shape (7,) partial derivatives of `pz` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    pz = c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))
    dpz = zeros(7)
    for k in range(7):
        dpz[k] = dc[k, 2, 0] + time * (dc[k, 2, 1] + time * (dc[k, 2, 2] + time * (dc[k, 2, 3] + time * dc[k, 2, 4])))
    return pz, dpz


@njit(fastmath=True)
def zpos_d(time: float | NDArray, tk: float, p: float, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight z position and its parameter derivatives at an absolute time.

    Direct counterpart of `zpos_cd`: epoch-folds the absolute time `time`
    around the expansion point `tk` and delegates to `zpos_cd`.

    Parameters
    ----------
    time : float or NDArray
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
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
        The sign distinguishes the transit (positive z) and eclipse
        (negative z) branches of the orbit.
    dpz : NDArray
        Shape (7,) partial derivatives of `pz` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return zpos_cd(time - (tk + epoch * p), c, dc)
