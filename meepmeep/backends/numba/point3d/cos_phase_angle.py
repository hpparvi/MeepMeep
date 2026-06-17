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

"""Single-expansion-point 3D phase-angle (star-planet-observer) cosine evaluators.

The centered evaluator delegates to ``position.pos_c`` to obtain the full
(x, y, z) position and forms the phase-angle cosine ``-z / sqrt(x^2 + y^2 +
z^2)`` from it. Unlike ``separation``, which inlines only the x/y Horner
passes to skip the unused line-of-sight coefficient, the phase angle needs
all three components, so delegating to ``pos_c`` is both cheaper to maintain
and free.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, sqrt, zeros, ndarray
from numpy.typing import NDArray

from .position import pos_c
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _cos_alpha_c_s(time, c):
    """Scalar kernel for :func:`cos_alpha_c`. See that function for documentation."""
    px, py, pz = pos_c(time, c)
    return -pz / sqrt(px ** 2 + py ** 2 + pz ** 2)


def _cos_alpha_c_v_body(time, c):
    """Vector-kernel body for :func:`cos_alpha_c`; see that function for documentation.

    Compiled twice: ``cos_alpha_c_v`` is the serial kernel (``prange``
    compiles as a plain ``range`` without ``parallel=True``) and
    ``cos_alpha_c_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    ca = zeros(n)
    for j in prange(n):
        ca[j] = _cos_alpha_c_s(time[j], c)
    return ca


cos_alpha_c_v = njit(fastmath=True)(_cos_alpha_c_v_body)
cos_alpha_c_vp = njit(fastmath=True, parallel=True)(_cos_alpha_c_v_body)


def cos_alpha_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the cosine of the orbital phase angle at an expansion-point-centered time.

    Centered counterpart of `cos_alpha`: assumes `time` has already been
    shifted to be relative to the expansion point. The phase angle
    alpha is the star-planet-observer angle. With z positive toward the
    observer, ``cos alpha = -z / r`` where ``r = sqrt(x^2 + y^2 + z^2)``.
    At superior conjunction (full phase, planet behind star)
    ``cos alpha = +1``; at inferior conjunction (new phase, planet in
    front) ``cos alpha = -1``.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. All three
        rows (x, y, z) are read.

    Returns
    -------
    cos_alpha : float or NDArray
        Cosine of the phase angle, in [-1, 1].
    """
    if isinstance(time, ndarray):
        return cos_alpha_c_v(time, c)
    return _cos_alpha_c_s(time, c)


@overload(cos_alpha_c, jit_options={'fastmath': True}, inline='always')
def _cos_alpha_c_overload(time, c):
    if _is_1d_array(time):
        def impl(time, c):
            return cos_alpha_c_v(time, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c):
            return _cos_alpha_c_s(time, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _cos_alpha_s(time, tc, p, c, te):
    """Scalar kernel for :func:`cos_alpha`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _cos_alpha_c_s(time - (tc + te + epoch * p), c)


def _cos_alpha_v_body(time, tc, p, c, te):
    """Vector-kernel body for :func:`cos_alpha`; see that function for documentation.

    Compiled twice: ``cos_alpha_v`` is the serial kernel (``prange``
    compiles as a plain ``range`` without ``parallel=True``) and
    ``cos_alpha_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    ca = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        ca[j] = _cos_alpha_c_s(time[j] - (tc + te + epoch * p), c)
    return ca


cos_alpha_v = njit(fastmath=True)(_cos_alpha_v_body)
cos_alpha_vp = njit(fastmath=True, parallel=True)(_cos_alpha_v_body)


def cos_alpha(time: float | NDArray, tc: float, p: float, c: NDArray, te: float = 0.0) -> float | NDArray:
    """
    Evaluate the cosine of the orbital phase angle at an absolute time.

    Folds the absolute observation time back to an expansion-point-centered offset
    and delegates to the centered kernel. The phase angle alpha is the
    star-planet-observer angle. With z positive toward the observer,
    ``cos alpha = -z / r`` where ``r = sqrt(x^2 + y^2 + z^2)``. At
    superior conjunction (full phase, planet behind star)
    ``cos alpha = +1``; at inferior conjunction (new phase, planet in
    front) ``cos alpha = -1``.

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
    cos_alpha : float or NDArray
        Cosine of the phase angle, in [-1, 1].
    """
    if isinstance(time, ndarray):
        return cos_alpha_v(time, tc, p, c, te)
    return _cos_alpha_s(time, tc, p, c, te)


@overload(cos_alpha, jit_options={'fastmath': True}, inline='always')
def _cos_alpha_overload(time, tc, p, c, te=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, te=0.0):
            return cos_alpha_v(time, tc, p, c, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, te=0.0):
            return _cos_alpha_s(time, tc, p, c, te)
        return impl
    return None
