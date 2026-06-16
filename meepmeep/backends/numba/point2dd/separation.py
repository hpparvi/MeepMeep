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

"""Single-expansion-point 2D projected-separation evaluators with parameter derivatives.

The centered kernel inlines the position-gradient Horner passes with scalar
temporaries (mirroring ``point3dd.separation``) rather than materialising
the intermediate position-gradient arrays, so the hot vector loops run
without per-sample allocations.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, sqrt, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _sep_cd_w(time, c, dc, dd):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter separation gradient into the caller-provided
    ``(7,)`` buffer ``dd`` and returns the separation. The position
    gradients are reduced through the chain rule with scalar temporaries,
    so no intermediate arrays are allocated.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    d = sqrt(px ** 2 + py ** 2)
    for k in range(7):
        dpx = dc[k, 0, 0] + time * (dc[k, 0, 1] + time * (dc[k, 0, 2] + time * (dc[k, 0, 3] + time * dc[k, 0, 4])))
        dpy = dc[k, 1, 0] + time * (dc[k, 1, 1] + time * (dc[k, 1, 2] + time * (dc[k, 1, 3] + time * dc[k, 1, 4])))
        dd[k] = (px * dpx + py * dpy) / d
    return d


@njit(fastmath=True)
def _sep_cd_s(time, c, dc):
    """Scalar kernel for :func:`sep_cd`. See that function for documentation."""
    dd = zeros(7)
    d = _sep_cd_w(time, c, dc, dd)
    return d, dd


def _sep_cd_v_body(time, c, dc):
    """Vector-kernel body for :func:`sep_cd`; see that function for documentation.

    Compiled twice: ``sep_cd_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``sep_cd_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    d = zeros(n)
    dd = zeros((n, 7))
    for j in prange(n):
        d[j] = _sep_cd_w(time[j], c, dc, dd[j])
    return d, dd


sep_cd_v = njit(fastmath=True)(_sep_cd_v_body)
sep_cd_vp = njit(fastmath=True, parallel=True)(_sep_cd_v_body)


def sep_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the projected planet-star separation and its parameter derivatives at an expansion-point-centered time.

    Computes the sky-plane position via `pos_cd` and reduces it to the
    Euclidean distance `d = sqrt(px^2 + py^2)`, propagating the parameter
    derivatives using the chain rule.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `separation.sep_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (7, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    d : float or ndarray
        Projected planet-star center distance in units of stellar radii.
        Shape (N,) for an array `time`.
    dd : NDArray
        Partial derivatives of `d` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.

    Notes
    -----
    The chain-rule reduction used here is
    `dd/dtheta = (px * dpx/dtheta + py * dpy/dtheta) / d`.
    The expression is regular for `d > 0`, which is the regime of
    interest for transit modeling; the gradient is ill-defined exactly
    at the (geometrically singular) point of zero projected separation.
    """
    if isinstance(time, ndarray):
        return sep_cd_v(time, c, dc)
    return _sep_cd_s(time, c, dc)


@overload(sep_cd, jit_options={'fastmath': True})
def _sep_cd_overload(time, c, dc):
    if _is_1d_array(time):
        def impl(time, c, dc):
            return sep_cd_v(time, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c, dc):
            return _sep_cd_s(time, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _sep_d_s(time, tc, p, c, dc, te):
    """Scalar kernel for :func:`sep_d`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _sep_cd_s(time - (tc + te + epoch * p), c, dc)


def _sep_d_v_body(time, tc, p, c, dc, te):
    """Vector-kernel body for :func:`sep_d`; see that function for documentation.

    Compiled twice: ``sep_d_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``sep_d_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    d = zeros(n)
    dd = zeros((n, 7))
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        d[j] = _sep_cd_w(time[j] - (tc + te + epoch * p), c, dc, dd[j])
    return d, dd


sep_d_v = njit(fastmath=True)(_sep_d_v_body)
sep_d_vp = njit(fastmath=True, parallel=True)(_sep_d_v_body)


def sep_d(time: float | NDArray, tc: float, p: float, c: NDArray, dc: NDArray, te: float = 0.0):
    """
    Evaluate the projected planet-star distance and its parameter derivatives at an absolute time.

    Direct counterpart of `sep_cd`: epoch-folds the absolute time `time`
    around the expansion point `te` and delegates to `sep_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `separation.sep`.

    Parameters
    ----------
    time : float or ndarray
        Absolute observation time(s) in the same units as `tc` and `p`.
    tc : float
        Transit-centre time (time of inferior conjunction), on the same
        time axis as `time`.
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (2, 5) Taylor coefficient matrix produced by `solve2d`.
    dc : NDArray
        A (7, 2, 5) parameter-derivative tensor produced by `solve2d_d`,
        with the leading axis ordered as `(tc, p, a, i, e, w, lan)`.
    te : float, optional
        Expansion-point offset from the transit centre [days] - the same value that
        was passed to `solve2d_d`. Defaults to 0.0, the expansion point at the
        transit centre.

    Returns
    -------
    d : float or ndarray
        Projected planet-star center distance in units of stellar radii.
        Shape (N,) for an array `time`.
    dd : NDArray
        Partial derivatives of `d` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return sep_d_v(time, tc, p, c, dc, te)
    return _sep_d_s(time, tc, p, c, dc, te)


@overload(sep_d, jit_options={'fastmath': True})
def _sep_d_overload(time, tc, p, c, dc, te=0.0):
    if _is_1d_array(time):
        def impl(time, tc, p, c, dc, te=0.0):
            return sep_d_v(time, tc, p, c, dc, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, tc, p, c, dc, te=0.0):
            return _sep_d_s(time, tc, p, c, dc, te)
        return impl
    return None
