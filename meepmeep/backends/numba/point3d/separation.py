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

"""Single-knot 3D sky-projected planet-star separation evaluators.

Unlike the 2D analogue, the centered evaluator inlines the x/y Horner
passes rather than delegating to ``position.pos_c``, so it avoids
computing the unused line-of-sight (z) coefficient. The module therefore
has no internal dependency on ``position``.
"""

from numba import njit
from numpy import floor, sqrt
from numpy.typing import NDArray


@njit(fastmath=True, inline='always')
def sep_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation in units of stellar radii at a knot-centered time.

    Centered counterpart of `sep`: assumes `time` has already been
    shifted to be relative to the expansion point. Only the x and y
    Taylor polynomials are evaluated; the z polynomial is skipped
    because the sky projection drops the line-of-sight component.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only rows 0
        and 1 (x and y) are read.

    Returns
    -------
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
        Always non-negative.

    Notes
    -----
    Unlike the 2D analogue, which delegates to `pos_c`, this routine
    inlines the px/py Horner evaluations to avoid the wasted work of
    computing the z coefficient that `pos_c` would also evaluate.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True, inline='always')
def sep(time: float | NDArray, tk: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation at an absolute time.

    Folds the absolute observation time back to a knot-centered offset
    and delegates to `sep_c`. This is the quantity most commonly used
    by transit light-curve models, where it represents the sky-projected
    separation between the centers of the star and planet in units of
    the stellar radius.

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
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
        Always non-negative; the sign of the line-of-sight depth
        (transit vs. eclipse) is not encoded here. Use `zpos` or `zpos_c`
        if the transit/eclipse branch is needed.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return sep_c(time - (tk + epoch * p), c)
