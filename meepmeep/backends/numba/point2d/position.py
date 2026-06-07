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

"""Single-knot 2D planet sky-plane (x, y) position evaluators."""

from numba import njit
from numpy import floor, sqrt
from numpy.typing import NDArray


@njit(fastmath=True, inline='always')
def pos_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray]:
    """
    Evaluate the planet's sky-plane (x, y) position at a knot-centered time.

    This is the "centered" variant of `pos`: it assumes the caller has
    already subtracted the expansion time `tk` (and any epoch offset) so
    that `t` is a small displacement around the knot. The polynomial is
    evaluated using Horner's scheme.

    Parameters
    ----------
    time : float
        Time relative to the Taylor series expansion point, i.e.
        `t = tc - (tk + epoch*p)`. Must lie within the knot's region of
        validity for the truncation error to remain small.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`. See `pos` for
        the column ordering convention.

    Returns
    -------
    px : float
        Sky-plane x position in units of stellar radii.
    py : float
        Sky-plane y position in units of stellar radii.

    Notes
    -----
    This is the fastest 2D position evaluator in the module since it skips
    the epoch-folding arithmetic. Prefer it whenever the knot index and
    centered time are already known (e.g. inside multi-knot dispatch loops).
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    return px, py


@njit(fastmath=True, inline='always')
def pos(time: float | NDArray, tk: float, p: float, c: NDArray):
    """
    Evaluate the planet's sky-plane (x, y) position at an absolute time using a 2D Taylor expansion.

    This is the "direct" variant of the 2D position evaluator: it accepts an
    absolute observation time `tc`, folds it back into a single orbital epoch
    around the expansion point `tk`, and then evaluates the 5th-order Taylor
    polynomial stored in `c` using Horner's scheme.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tk` and `p`
        (typically days). Scalar or array inputs are both accepted; the
        return type matches.
    tk : float
        Time at which the Taylor series was expanded (the knot time).
    p : float
        Orbital period, used to fold `tc` into the interval `[tk - p/2, tk + p/2)`.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`. Row 0 holds the
        x-direction coefficients and row 1 the y-direction coefficients,
        ordered as [position, velocity, acceleration/2, jerk/6, snap/24]
        (i.e. already pre-scaled by the factorial of the Taylor order).

    Returns
    -------
    px : float or NDArray
        Sky-plane x position(s) in units of stellar radii.
    py : float or NDArray
        Sky-plane y position(s) in units of stellar radii.

    Notes
    -----
    Epoch folding uses `epoch = floor((tc - tk + p/2) / p)`, which centers the
    residual `t = tc - (tk + epoch*p)` on the knot. This keeps the polynomial
    argument small and preserves the accuracy of the truncated Taylor series.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return pos_c(time - (tk + epoch * p), c)


@njit(fastmath=True, inline='always')
def pos_and_sep_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y) position and the projected distance jointly.

    Returns the sky-plane coordinates and the Euclidean distance
    `sqrt(x^2 + y^2)` from a single Horner-scheme evaluation, saving the
    redundant polynomial work that would occur if `pos_c` and `sep_c` were
    called separately.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.
    d : float or NDArray
        Projected planet-star center distance in units of stellar radii.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    return px, py, sqrt(px ** 2 + py ** 2)


@njit(fastmath=True, inline='always')
def pos_and_sep(time: float | NDArray, tk: float, p: float, c: NDArray) -> tuple[
    float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y) position and the projected distance jointly.

    Returns the sky-plane coordinates and the Euclidean distance
    `sqrt(x^2 + y^2)` from a single Horner-scheme evaluation, saving the
    redundant polynomial work that would occur if `pos_c` and `sep_c` were
    called separately.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.
    d : float or NDArray
        Projected planet-star center distance in units of stellar radii.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return pos_and_sep_c(time - (tk + epoch * p), c)
