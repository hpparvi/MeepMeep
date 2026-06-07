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

"""Single-knot 3D line-of-sight (z) velocity evaluators."""

from numba import njit
from numpy import floor
from numpy.typing import NDArray


@njit(fastmath=True, inline='always')
def zvel_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight velocity component at a knot-centered time.

    Centered counterpart of `zvel`. Only the z-direction coefficients
    (row 2 of `c`) are read, making this the cheapest velocity
    evaluator in the module. The polynomial is the analytic derivative
    of the position z-polynomial, 4th-order in `time`.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.

    Notes
    -----
    Useful for radial-velocity computations, where only the
    line-of-sight component matters; see `rv_c` / `rv`.
    """
    return c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))


@njit(fastmath=True, inline='always')
def zvel(time: float | NDArray, tk: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight velocity component at an absolute time.

    Direct counterpart of `zvel_c`: accepts an absolute observation
    time `time`, folds it back into a single orbital epoch around the
    expansion point `tk`, and delegates to `zvel_c`.

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
        (the z-direction coefficients) is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return zvel_c(time - (tk + epoch * p), c)
