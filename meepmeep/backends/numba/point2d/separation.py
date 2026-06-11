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

from numba import njit, types
from numba.extending import overload
from numpy import floor, sqrt, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array
from .position import _pos_c_s


@njit(fastmath=True, inline='always')
def _sep_c_s(time, c):
    """Scalar kernel for :func:`sep_c`. See that function for documentation."""
    px, py = _pos_c_s(time, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def _sep_c_v(time, c):
    """Vector kernel for :func:`sep_c`. See that function for documentation."""
    n = time.size
    d = zeros(n)
    for j in range(n):
        d[j] = _sep_c_s(time[j], c)
    return d


def sep_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation in the units of stellar radii at a knot-centered time.

    Centered counterpart of `sep`: assumes `time` has already been shifted
    to be relative to the expansion point, evaluates the 2D position, and
    returns `sqrt(x^2 + y^2)`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`.

    Returns
    -------
    d : float or NDArray
        Projected planet-star center distance in units of stellar radii.
    """
    if isinstance(time, ndarray):
        return _sep_c_v(time, c)
    return _sep_c_s(time, c)


@overload(sep_c, jit_options={'fastmath': True}, inline='always')
def _sep_c_overload(time, c):
    if _is_1d_array(time):
        def impl(time, c):
            return _sep_c_v(time, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c):
            return _sep_c_s(time, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _sep_s(time, tk, p, c):
    """Scalar kernel for :func:`sep`. See that function for documentation."""
    epoch = floor((time - tk + 0.5 * p) / p)
    return _sep_c_s(time - (tk + epoch * p), c)


@njit(fastmath=True)
def _sep_v(time, tk, p, c):
    """Vector kernel for :func:`sep`. See that function for documentation."""
    n = time.size
    d = zeros(n)
    for j in range(n):
        epoch = floor((time[j] - tk + 0.5 * p) / p)
        d[j] = _sep_c_s(time[j] - (tk + epoch * p), c)
    return d


def sep(time: float | NDArray, tk: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the projected planet-star separation at an absolute time.

    Computes the sky-plane (x, y) position and returns the Euclidean
    distance `sqrt(x^2 + y^2)`. This is the quantity most commonly used
    by transit light-curve models, where it represents the
    center-to-center separation between the planet and star projected
    onto the plane of the sky.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

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
    if isinstance(time, ndarray):
        return _sep_v(time, tk, p, c)
    return _sep_s(time, tk, p, c)


@overload(sep, jit_options={'fastmath': True}, inline='always')
def _sep_overload(time, tk, p, c):
    if _is_1d_array(time):
        def impl(time, tk, p, c):
            return _sep_v(time, tk, p, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tk, p, c):
            return _sep_s(time, tk, p, c)
        return impl
    return None
