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

"""Single-knot 2D sky-projected planet-star separation evaluators."""

from numba import njit
from numpy import sqrt
from numpy.typing import NDArray

from .position import pos_c, pos


@njit(fastmath=True, inline='always')
def sep_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation in the units of stellar radii at a knot-centered time.

    Centered counterpart of `sep`: assumes `tc` has already been shifted
    to be relative to the expansion point, evaluates the 2D position via
    `pos_c`, and returns `sqrt(x^2 + y^2)`.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`.

    Returns
    -------
    d : float
        Projected planet-star center distance in units of stellar radii.
    """
    px, py = pos_c(time, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True, inline='always')
def sep(time, tk, p, c):
    """
    Evaluate the projected planet-star separation at an absolute time.

    Computes the sky-plane (x, y) position via `pos` and returns the
    Euclidean distance `sqrt(x^2 + y^2)`. This is the quantity most
    commonly used by transit light-curve models, where it represents the
    center-to-center separation between the planet and star projected
    onto the plane of the sky.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    tk : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`.

    Returns
    -------
    d : float or NDArray
        Projected planet-star center distance in units of stellar radii.
        Always non-negative; the sign of the line-of-sight depth (transit
        vs. eclipse) is not encoded here.
    """
    px, py = pos(time, tk, p, c)
    return sqrt(px ** 2 + py ** 2)
