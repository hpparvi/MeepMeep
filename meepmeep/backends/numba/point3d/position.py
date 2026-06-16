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

"""Single-expansion-point 3D planet (x, y, z) position evaluators."""

from numba import njit, prange, types
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


def _pos_c_v_body(time, c):
    """Vector-kernel body for :func:`pos_c`; see that function for documentation.

    Compiled twice: ``_pos_c_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_pos_c_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    px = zeros(n)
    py = zeros(n)
    pz = zeros(n)
    for j in prange(n):
        px[j], py[j], pz[j] = _pos_c_s(time[j], c)
    return px, py, pz


_pos_c_v = njit(fastmath=True)(_pos_c_v_body)
_pos_c_vp = njit(fastmath=True, parallel=True)(_pos_c_v_body)


def pos_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position at an expansion-point-centered time.

    This is the "centered" variant of `pos`: it assumes the caller has
    already subtracted the expansion time `te` (and any epoch offset) so
    that `time` is a small displacement around the expansion point. Each spatial
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
        `time = tc - (te + epoch*p)`. Must lie within the expansion point's region
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
    skips the epoch-folding arithmetic. Prefer it whenever the expansion point
    index and centered time are already known (e.g. inside multi-expansion-point
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
def _pos_s(time, tc, p, c, te):
    """Scalar kernel for :func:`pos`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _pos_c_s(time - (tc + te + epoch * p), c)


def _pos_v_body(time, tc, p, c, te):
    """Vector-kernel body for :func:`pos`; see that function for documentation.

    Compiled twice: ``_pos_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_pos_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    px = zeros(n)
    py = zeros(n)
    pz = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        px[j], py[j], pz[j] = _pos_c_s(time[j] - (tc + te + epoch * p), c)
    return px, py, pz


_pos_v = njit(fastmath=True)(_pos_v_body)
_pos_vp = njit(fastmath=True, parallel=True)(_pos_v_body)


def pos(time: float | NDArray, tc: float, p: float, c: NDArray, te: float = 0.0) -> tuple[
    float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position at an absolute time using a 3D Taylor expansion.

    This is the "direct" variant of the 3D position evaluator: it
    accepts an absolute observation time `time`, folds it back into a
    single orbital epoch around the expansion point `te`, and then
    evaluates the 5th-order Taylor polynomial stored in `c` using
    Horner's scheme via the centered kernel.

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
        A (3, 5) coefficient matrix produced by `solve3d`. Row 0 holds
        the x-direction coefficients, row 1 the y-direction, and row 2
        the z-direction, ordered as
        [position, velocity, acceleration/2, jerk/6, snap/24]
        (i.e. already pre-scaled by the factorial of the Taylor order).
    te : float, optional
        Expansion-point offset from the transit centre [days] - the same value that
        was passed to `solve3d`. Defaults to 0.0, the expansion point at the
        transit centre.

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
        return _pos_v(time, tc, p, c, te)
    return _pos_s(time, tc, p, c, te)


@overload(pos, jit_options={'fastmath': True}, inline='always')
def _pos_overload(time, tc, p, c, te=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, te=0.0):
            return _pos_v(time, tc, p, c, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, te=0.0):
            return _pos_s(time, tc, p, c, te)
        return impl
    return None
