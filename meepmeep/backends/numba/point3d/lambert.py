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

"""Single-expansion-point Lambertian phase-curve evaluators.

Holds the Lambertian reflected-light phase curve evaluated from a single
Taylor expansion (:func:`lambert_phase_curve_c` / :func:`lambert_phase_curve`)
and its shared phase kernel (:func:`_lambert_kernel`). The flux is formed from
the phase-angle cosine produced by the ``cos_phase_angle`` evaluators; the
multi-expansion-point ``orbit3d`` dispatchers delegate to the routines here.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, pi, sqrt, arccos, zeros, ndarray
from numpy.typing import NDArray

from .cos_phase_angle import _cos_alpha_c_s
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _lambert_kernel(cos_alpha):
    """Lambertian phase function evaluated at a cosine of the phase angle.

    Computes :math:`f(\\alpha) = (\\sin\\alpha + (\\pi - \\alpha)\\cos\\alpha)/\\pi`,
    the disk-integrated reflectance of a Lambertian sphere. The
    implementation substitutes :math:`\\sin\\alpha = \\sqrt{1 - \\cos^2\\alpha}`
    to skip one trig call, and clamps ``cos_alpha`` to ``[-1, 1]`` so a
    Taylor-rounding overshoot cannot produce a NaN from :func:`arccos`.

    Parameters
    ----------
    cos_alpha : float
        Cosine of the phase angle.

    Returns
    -------
    phase : float
        Value of the Lambert kernel, in :math:`[0, 1]`.
    alpha : float
        Phase angle :math:`\\arccos(\\text{cos\\_alpha})` [radians],
        returned as a by-product so callers that also need
        :math:`\\alpha` avoid a second :func:`arccos`.
    """
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
    alpha = arccos(cos_alpha)
    return (sin_alpha + (pi - alpha) * cos_alpha) / pi, alpha


@njit(fastmath=True, inline='always')
def _lambert_phase_curve_c_s(time, ag, a, k, c):
    """Scalar kernel for :func:`lambert_phase_curve_c`. See that function for documentation."""
    amplitude = k * k * ag / (a * a)
    phase, _ = _lambert_kernel(_cos_alpha_c_s(time, c))
    return amplitude * phase


def _lambert_phase_curve_c_v_body(time, ag, a, k, c):
    """Vector-kernel body for :func:`lambert_phase_curve_c`; see that function for documentation.

    Compiled twice: ``lambert_phase_curve_c_v`` is the serial kernel
    (``prange`` compiles as a plain ``range`` without ``parallel=True``) and
    ``lambert_phase_curve_c_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    res = zeros(n)
    for j in prange(n):
        res[j] = _lambert_phase_curve_c_s(time[j], ag, a, k, c)
    return res


lambert_phase_curve_c_v = njit(fastmath=True)(_lambert_phase_curve_c_v_body)
lambert_phase_curve_c_vp = njit(fastmath=True, parallel=True)(_lambert_phase_curve_c_v_body)


def lambert_phase_curve_c(time: float | NDArray, ag: float, a: float, k: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the Lambertian phase-curve flux contribution at an expansion-point-centered time.

    Centered counterpart of `lambert_phase_curve`: assumes `time` has
    already been shifted to be relative to the expansion point. Evaluates
    :math:`F = (k/a)^2\\, A_g\\, f(\\alpha)` where :math:`f` is the Lambert
    kernel and :math:`\\alpha` the instantaneous phase angle. The result is
    the planet-to-star flux ratio of reflected light at full phase scaled by
    the phase function.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    flux : float or NDArray
        Reflected planet-to-star flux ratio. Shape (N,) for an array `time`.
    """
    if isinstance(time, ndarray):
        return lambert_phase_curve_c_v(time, ag, a, k, c)
    return _lambert_phase_curve_c_s(time, ag, a, k, c)


@overload(lambert_phase_curve_c, jit_options={'fastmath': True}, inline='always')
def _lambert_phase_curve_c_overload(time, ag, a, k, c):
    if _is_1d_array(time):
        def impl(time, ag, a, k, c):
            return lambert_phase_curve_c_v(time, ag, a, k, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, ag, a, k, c):
            return _lambert_phase_curve_c_s(time, ag, a, k, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _lambert_phase_curve_s(time, ag, a, k, tc, p, c, te):
    """Scalar kernel for :func:`lambert_phase_curve`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _lambert_phase_curve_c_s(time - (tc + te + epoch * p), ag, a, k, c)


def _lambert_phase_curve_v_body(time, ag, a, k, tc, p, c, te):
    """Vector-kernel body for :func:`lambert_phase_curve`; see that function for documentation.

    Compiled twice: ``lambert_phase_curve_v`` is the serial kernel
    (``prange`` compiles as a plain ``range`` without ``parallel=True``) and
    ``lambert_phase_curve_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    res = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        res[j] = _lambert_phase_curve_c_s(time[j] - (tc + te + epoch * p), ag, a, k, c)
    return res


lambert_phase_curve_v = njit(fastmath=True)(_lambert_phase_curve_v_body)
lambert_phase_curve_vp = njit(fastmath=True, parallel=True)(_lambert_phase_curve_v_body)


def lambert_phase_curve(time: float | NDArray, ag: float, a: float, k: float, tc: float, p: float,
                        c: NDArray, te: float = 0.0) -> float | NDArray:
    """
    Evaluate the Lambertian phase-curve flux contribution at an absolute time.

    Folds the absolute observation time back to an expansion-point-centered offset
    and delegates to the centered kernel. Evaluates
    :math:`F = (k/a)^2\\, A_g\\, f(\\alpha)` where :math:`f` is the Lambert
    kernel and :math:`\\alpha` the instantaneous phase angle.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
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
    flux : float or NDArray
        Reflected planet-to-star flux ratio. Shape (N,) for an array `time`.
    """
    if isinstance(time, ndarray):
        return lambert_phase_curve_v(time, ag, a, k, tc, p, c, te)
    return _lambert_phase_curve_s(time, ag, a, k, tc, p, c, te)


@overload(lambert_phase_curve, jit_options={'fastmath': True}, inline='always')
def _lambert_phase_curve_overload(time, ag, a, k, tc, p, c, te=0.0):
    if _is_1d_array(time):
        def impl(time, ag, a, k, tc, p, c, te=0.0):
            return lambert_phase_curve_v(time, ag, a, k, tc, p, c, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, ag, a, k, tc, p, c, te=0.0):
            return _lambert_phase_curve_s(time, ag, a, k, tc, p, c, te)
        return impl
    return None
