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

"""Single-expansion-point 3D line-of-sight (z) position evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _zpos_c_s(time, c):
    """Scalar kernel for :func:`zpos_c`. See that function for documentation."""
    return c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))


def _zpos_c_v_body(time, c):
    """Vector-kernel body for :func:`zpos_c`; see that function for documentation.

    Compiled twice: ``_zpos_c_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_zpos_c_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    pz = zeros(n)
    for j in prange(n):
        pz[j] = _zpos_c_s(time[j], c)
    return pz


_zpos_c_v = njit(fastmath=True)(_zpos_c_v_body)
_zpos_c_vp = njit(fastmath=True, parallel=True)(_zpos_c_v_body)


def zpos_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight z position at an expansion-point-centered time.

    Centered counterpart of `zpos`: evaluates only the z-direction Taylor
    polynomial (row 2 of `c`), skipping the x and y rows. This is the
    cheapest 3D evaluator in the module and is the right choice when
    only the transit/eclipse branch is needed (e.g. to discriminate
    primary from secondary eclipse via the sign of z).

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
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
    """
    if isinstance(time, ndarray):
        return _zpos_c_v(time, c)
    return _zpos_c_s(time, c)


@overload(zpos_c, jit_options={'fastmath': True}, inline='always')
def _zpos_c_overload(time, c):
    if _is_1d_array(time):
        def impl(time, c):
            return _zpos_c_v(time, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c):
            return _zpos_c_s(time, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _zpos_s(time, tc, p, c, te):
    """Scalar kernel for :func:`zpos`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _zpos_c_s(time - (tc + te + epoch * p), c)


def _zpos_v_body(time, tc, p, c, te):
    """Vector-kernel body for :func:`zpos`; see that function for documentation.

    Compiled twice: ``_zpos_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_zpos_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    pz = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        pz[j] = _zpos_c_s(time[j] - (tc + te + epoch * p), c)
    return pz


_zpos_v = njit(fastmath=True)(_zpos_v_body)
_zpos_vp = njit(fastmath=True, parallel=True)(_zpos_v_body)


def zpos(time: float | NDArray, tc: float, p: float, c: NDArray, te: float = 0.0) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight z position at an absolute time.

    Folds the absolute observation time back to an expansion-point-centered offset
    and delegates to the centered kernel.

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
    te : float, optional
        Expansion-point offset from the transit centre [days] - the same value that
        was passed to `solve3d`. Defaults to 0.0, the expansion point at the
        transit centre.

    Returns
    -------
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
        The sign distinguishes the transit (positive z) and eclipse
        (negative z) branches of the orbit.
    """
    if isinstance(time, ndarray):
        return _zpos_v(time, tc, p, c, te)
    return _zpos_s(time, tc, p, c, te)


@overload(zpos, jit_options={'fastmath': True}, inline='always')
def _zpos_overload(time, tc, p, c, te=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, te=0.0):
            return _zpos_v(time, tc, p, c, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, te=0.0):
            return _zpos_s(time, tc, p, c, te)
        return impl
    return None
