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

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, sqrt, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _sep_c_s(time, c):
    """Scalar kernel for :func:`sep_c`. See that function for documentation."""
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    return sqrt(px ** 2 + py ** 2)


def _sep_c_v_body(time, c):
    """Vector-kernel body for :func:`sep_c`; see that function for documentation.

    Compiled twice: ``_sep_c_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_sep_c_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    d = zeros(n)
    for j in prange(n):
        d[j] = _sep_c_s(time[j], c)
    return d


_sep_c_v = njit(fastmath=True)(_sep_c_v_body)
_sep_c_vp = njit(fastmath=True, parallel=True)(_sep_c_v_body)


def sep_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation in units of stellar radii at a knot-centered time.

    Centered counterpart of `sep`: assumes `time` has already been
    shifted to be relative to the expansion point. Only the x and y
    Taylor polynomials are evaluated; the z polynomial is skipped
    because the sky projection drops the line-of-sight component.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

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
def _sep_s(time, tc, p, c, tk):
    """Scalar kernel for :func:`sep`. See that function for documentation."""
    epoch = floor((time - tc - tk + 0.5 * p) / p)
    return _sep_c_s(time - (tc + tk + epoch * p), c)


def _sep_v_body(time, tc, p, c, tk):
    """Vector-kernel body for :func:`sep`; see that function for documentation.

    Compiled twice: ``_sep_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_sep_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    d = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - tk + 0.5 * p) / p)
        d[j] = _sep_c_s(time[j] - (tc + tk + epoch * p), c)
    return d


_sep_v = njit(fastmath=True)(_sep_v_body)
_sep_vp = njit(fastmath=True, parallel=True)(_sep_v_body)


def sep(time: float | NDArray, tc: float, p: float, c: NDArray, tk: float = 0.0) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation at an absolute time.

    Folds the absolute observation time back to a knot-centered offset
    and delegates to the centered kernel. This is the quantity most
    commonly used by transit light-curve models, where it represents the
    sky-projected separation between the centers of the star and planet
    in units of the stellar radius.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    tc : float
        Transit-centre time (time of inferior conjunction), on the same
        time axis as `time`.
    p : float
        Orbital period.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.
    tk : float, optional
        Knot offset from the transit centre [days] - the same value that
        was passed to `solve3d`. Defaults to 0.0, the knot at the
        transit centre.

    Returns
    -------
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
        Always non-negative; the sign of the line-of-sight depth
        (transit vs. eclipse) is not encoded here. Use `zpos` or `zpos_c`
        if the transit/eclipse branch is needed.
    """
    if isinstance(time, ndarray):
        return _sep_v(time, tc, p, c, tk)
    return _sep_s(time, tc, p, c, tk)


@overload(sep, jit_options={'fastmath': True}, inline='always')
def _sep_overload(time, tc, p, c, tk=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, tk=0.0):
            return _sep_v(time, tc, p, c, tk)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, tk=0.0):
            return _sep_s(time, tc, p, c, tk)
        return impl
    return None
