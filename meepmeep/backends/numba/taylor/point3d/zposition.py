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

"""Single-knot 3D line-of-sight (z) position evaluators."""

from numba import njit
from numpy import floor
from numpy.typing import NDArray


@njit(fastmath=True, inline='always')
def zpos_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight z position at a knot-centered time.

    Centered counterpart of `zpos`: evaluates only the z-direction Taylor
    polynomial (row 2 of `c`), skipping the x and y rows. This is the
    cheapest 3D evaluator in the module and is the right choice when
    only the transit/eclipse branch is needed (e.g. to discriminate
    primary from secondary eclipse via the sign of z).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.

    Returns
    -------
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
    """
    return c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))


@njit(fastmath=True, inline='always')
def zpos(time: float | NDArray, tk: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight z position at an absolute time.

    Folds the absolute observation time back to a knot-centered offset
    and delegates to `zpos_c`.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    tk : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
        The sign distinguishes the transit (positive z) and eclipse
        (negative z) branches of the orbit.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return zpos_c(time - (tk + epoch * p), c)
