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

"""Single-knot 3D planet (x, y, z) position evaluators."""

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
    pz = c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))
    return px, py, pz


@njit(fastmath=True)
def _pos_c_v(time, c):
    """Vector kernel for :func:`pos_c`. See that function for documentation."""
    n = time.size
    px = zeros(n)
    py = zeros(n)
    pz = zeros(n)
    for j in range(n):
        px[j], py[j], pz[j] = _pos_c_s(time[j], c)
    return px, py, pz


def pos_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position at a knot-centered time.

    This is the "centered" variant of `pos`: it assumes the caller has
    already subtracted the expansion time `tk` (and any epoch offset) so
    that `time` is a small displacement around the knot. Each spatial
    coordinate is evaluated as a 5th-order polynomial using Horner's
    scheme.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python). The array path is an explicit loop over the scalar
    kernel, which avoids the full-array temporaries that NumPy
    broadcasting would allocate for every Horner step.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point, i.e.
        `time = tc - (tk + epoch*p)`. Must lie within the knot's region
        of validity for the truncation error to remain small.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. See `pos` for
        the row/column ordering convention.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.

    Notes
    -----
    This is the fastest 3D position evaluator in the module since it
    skips the epoch-folding arithmetic. Prefer it whenever the knot
    index and centered time are already known (e.g. inside multi-knot
    dispatch loops in `orbit3d`).
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
def _pos_s(time, tk, p, c):
    """Scalar kernel for :func:`pos`. See that function for documentation."""
    epoch = floor((time - tk + 0.5 * p) / p)
    return _pos_c_s(time - (tk + epoch * p), c)


@njit(fastmath=True)
def _pos_v(time, tk, p, c):
    """Vector kernel for :func:`pos`. See that function for documentation."""
    n = time.size
    px = zeros(n)
    py = zeros(n)
    pz = zeros(n)
    for j in range(n):
        epoch = floor((time[j] - tk + 0.5 * p) / p)
        px[j], py[j], pz[j] = _pos_c_s(time[j] - (tk + epoch * p), c)
    return px, py, pz


def pos(time: float | NDArray, tk: float, p: float, c: NDArray) -> tuple[
    float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position at an absolute time using a 3D Taylor expansion.

    This is the "direct" variant of the 3D position evaluator: it
    accepts an absolute observation time `time`, folds it back into a
    single orbital epoch around the expansion point `tk`, and then
    evaluates the 5th-order Taylor polynomial stored in `c` using
    Horner's scheme via the centered kernel.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tk` and `p`
        (typically days). Scalar or array inputs are both accepted; the
        return type matches.
    tk : float
        Time at which the Taylor series was expanded (the knot time).
    p : float
        Orbital period, used to fold `time` into the interval
        `[tk - p/2, tk + p/2)`.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Row 0 holds
        the x-direction coefficients, row 1 the y-direction, and row 2
        the z-direction, ordered as
        [position, velocity, acceleration/2, jerk/6, snap/24]
        (i.e. already pre-scaled by the factorial of the Taylor order).

    Returns
    -------
    px : float or NDArray
        Sky-plane x position(s) in units of stellar radii.
    py : float or NDArray
        Sky-plane y position(s) in units of stellar radii.
    pz : float or NDArray
        Line-of-sight z position(s) in units of stellar radii. Positive
        values point toward the observer.

    """
    if isinstance(time, ndarray):
        return _pos_v(time, tk, p, c)
    return _pos_s(time, tk, p, c)


@overload(pos, jit_options={'fastmath': True}, inline='always')
def _pos_overload(time, tk, p, c):
    if _is_1d_array(time):
        def impl(time, tk, p, c):
            return _pos_v(time, tk, p, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tk, p, c):
            return _pos_s(time, tk, p, c)
        return impl
    return None
