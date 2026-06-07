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

from numba import njit
from numpy import floor, zeros
from numpy.typing import NDArray


@njit(fastmath=True)
def zpos_cd(time: float | NDArray, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight z position and its parameter derivatives at a knot-centered time.

    Centered companion to `position.zpos_c` that additionally returns
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
