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

from numba import njit, types
from numba.extending import overload
from numpy import floor, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _pos_c_s(time, c):
    """Scalar kernel for :func:`pos_c`. See that function for documentation."""
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    return px, py


@njit(fastmath=True)
def _pos_c_v(time, c):
    """Vector kernel for :func:`pos_c`. See that function for documentation."""
    n = time.size
    px = zeros(n)
    py = zeros(n)
    for j in range(n):
        px[j], py[j] = _pos_c_s(time[j], c)
    return px, py


def pos_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray]:
    """
    Evaluate the planet's sky-plane (x, y) position at a knot-centered time.

    This is the "centered" variant of `pos`: it assumes the caller has
    already subtracted the expansion time `tk` (and any epoch offset) so
    that `time` is a small displacement around the knot. The polynomial is
    evaluated using Horner's scheme.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python). The array path is an explicit loop over the scalar
    kernel, which avoids the full-array temporaries that NumPy
    broadcasting would allocate for every Horner step.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point, i.e.
        `t = tc - (tk + epoch*p)`. Must lie within the knot's region of
        validity for the truncation error to remain small.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`. See `pos` for
        the column ordering convention.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.

    Notes
    -----
    This is the fastest 2D position evaluator in the module since it skips
    the epoch-folding arithmetic. Prefer it whenever the knot index and
    centered time are already known (e.g. inside multi-knot dispatch loops).
    """
    if isinstance(time, ndarray):
        return _pos_c_v(time, c)
    return _pos_c_s(time, c)


@overload(pos_c, jit_options={'fastmath': True}, inline='always')
def _pos_c_overload(time, c):
    if _is_1d_array(time):
        def impl(time, c):
            return _pos_c_v(time, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c):
            return _pos_c_s(time, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _pos_s(time, tc, p, c, tk):
    """Scalar kernel for :func:`pos`. See that function for documentation."""
    epoch = floor((time - tc - tk + 0.5 * p) / p)
    return _pos_c_s(time - (tc + tk + epoch * p), c)


@njit(fastmath=True)
def _pos_v(time, tc, p, c, tk):
    """Vector kernel for :func:`pos`. See that function for documentation."""
    n = time.size
    px = zeros(n)
    py = zeros(n)
    for j in range(n):
        epoch = floor((time[j] - tc - tk + 0.5 * p) / p)
        px[j], py[j] = _pos_c_s(time[j] - (tc + tk + epoch * p), c)
    return px, py


def pos(time: float | NDArray, tc: float, p: float, c: NDArray, tk: float = 0.0):
    """
    Evaluate the planet's sky-plane (x, y) position at an absolute time using a 2D Taylor expansion.

    This is the "direct" variant of the 2D position evaluator: it accepts an
    absolute observation time, folds it back into a single orbital epoch
    around the expansion point `tk`, and then evaluates the 5th-order Taylor
    polynomial stored in `c` using Horner's scheme.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tc` and `p`
        (typically days). Scalar or array inputs are both accepted; the
        return type matches.
    tc : float
        Transit-centre time (time of inferior conjunction), on the same
        time axis as `time`.
    p : float
        Orbital period, used to fold `time` into a single epoch around
        the expansion point.
    c : NDArray
        A (2, 5) coefficient matrix produced by `solve2d`. Row 0 holds the
        x-direction coefficients and row 1 the y-direction coefficients,
        ordered as [position, velocity, acceleration/2, jerk/6, snap/24]
        (i.e. already pre-scaled by the factorial of the Taylor order).
    tk : float, optional
        Knot offset from the transit centre [days] - the same value that
        was passed to `solve2d`. Defaults to 0.0, the knot at the
        transit centre.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position(s) in units of stellar radii.
    py : float or NDArray
        Sky-plane y position(s) in units of stellar radii.

    Notes
    -----
    Epoch folding uses `epoch = floor((time - tc - tk + p/2) / p)`, which
    centers the residual `t = time - (tc + tk + epoch*p)` on the knot. This
    keeps the polynomial argument small and preserves the accuracy of the
    truncated Taylor series.
    """
    if isinstance(time, ndarray):
        return _pos_v(time, tc, p, c, tk)
    return _pos_s(time, tc, p, c, tk)


@overload(pos, jit_options={'fastmath': True}, inline='always')
def _pos_overload(time, tc, p, c, tk=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, tk=0.0):
            return _pos_v(time, tc, p, c, tk)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, tk=0.0):
            return _pos_s(time, tc, p, c, tk)
        return impl
    return None
