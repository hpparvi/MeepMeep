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

from numba import njit, types
from numba.extending import overload
from numpy import floor, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _zvel_c_s(time, c):
    """Scalar kernel for :func:`zvel_c`. See that function for documentation."""
    return c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))


@njit(fastmath=True)
def _zvel_c_v(time, c):
    """Vector kernel for :func:`zvel_c`. See that function for documentation."""
    n = time.size
    vz = zeros(n)
    for j in range(n):
        vz[j] = _zvel_c_s(time[j], c)
    return vz


def zvel_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight velocity component at a knot-centered time.

    Centered counterpart of `zvel`. Only the z-direction coefficients
    (row 2 of `c`) are read, making this the cheapest velocity
    evaluator in the module. The polynomial is the analytic derivative
    of the position z-polynomial, 4th-order in `time`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

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
    if isinstance(time, ndarray):
        return _zvel_c_v(time, c)
    return _zvel_c_s(time, c)


@overload(zvel_c, jit_options={'fastmath': True}, inline='always')
def _zvel_c_overload(time, c):
    if _is_1d_array(time):
        def impl(time, c):
            return _zvel_c_v(time, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c):
            return _zvel_c_s(time, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _zvel_s(time, tk, p, c):
    """Scalar kernel for :func:`zvel`. See that function for documentation."""
    epoch = floor((time - tk + 0.5 * p) / p)
    return _zvel_c_s(time - (tk + epoch * p), c)


@njit(fastmath=True)
def _zvel_v(time, tk, p, c):
    """Vector kernel for :func:`zvel`. See that function for documentation."""
    n = time.size
    vz = zeros(n)
    for j in range(n):
        epoch = floor((time[j] - tk + 0.5 * p) / p)
        vz[j] = _zvel_c_s(time[j] - (tk + epoch * p), c)
    return vz


def zvel(time: float | NDArray, tk: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight velocity component at an absolute time.

    Direct counterpart of `zvel_c`: accepts an absolute observation
    time `time`, folds it back into a single orbital epoch around the
    expansion point `tk`, and delegates to the centered kernel.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

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
    if isinstance(time, ndarray):
        return _zvel_v(time, tk, p, c)
    return _zvel_s(time, tk, p, c)


@overload(zvel, jit_options={'fastmath': True}, inline='always')
def _zvel_overload(time, tk, p, c):
    if _is_1d_array(time):
        def impl(time, tk, p, c):
            return _zvel_v(time, tk, p, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tk, p, c):
            return _zvel_s(time, tk, p, c)
        return impl
    return None
