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

from numba import njit
from numpy import floor, zeros
from numpy.typing import NDArray


@njit(fastmath=True)
def zvel_cd(time: float | NDArray, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight velocity and its parameter derivatives at a knot-centered time.

    Centered companion to `velocity.zvel_c` that additionally
    returns the partial derivatives of the line-of-sight velocity
    with respect to each of the seven orbital parameters. Only the
    z-direction polynomials are evaluated; the x and y rows of `c`
    and `dc` are not read.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`, with the leading axis ordered as
        `(tc, p, a, i, e, w, lan)`. Only the slice `dc[:, 2, :]` is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    dvz : NDArray
        Shape (7,) partial derivatives of `vz` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))
    dvz = zeros(7)
    for k in range(7):
        dvz[k] = dc[k, 2, 1] + time * (2.0 * dc[k, 2, 2] + time * (3.0 * dc[k, 2, 3] + time * 4.0 * dc[k, 2, 4]))
    return vz, dvz


@njit(fastmath=True)
def zvel_d(time: float | NDArray, tk: float, p: float, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight velocity and its parameter derivatives at an absolute time.

    Direct counterpart of `zvel_cd`: epoch-folds the absolute time
    `time` around the expansion point `tk` and delegates to
    `zvel_cd`.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tk` and `p`.
    tk : float
        Taylor series expansion time (knot time).
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
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    dvz : NDArray
        Shape (7,) partial derivatives of `vz` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return zvel_cd(time - (tk + epoch * p), c, dc)
