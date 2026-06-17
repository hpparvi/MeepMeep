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

"""Single-expansion-point ellipsoidal-variation signal evaluators.

Holds the ellipsoidal-variation (tidal) flux signal evaluated from a single
Taylor expansion (:func:`ev_signal_c` / :func:`ev_signal`). The amplitude
scales with the planet-to-star mass ratio, the projected-area factor
``sin^2 inc``, and the inverse cube of the instantaneous 3D star-planet
distance ``d = sqrt(x^2 + y^2 + z^2)``. The multi-expansion-point ``orbit3d``
dispatchers delegate to the routines here.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, sin, sqrt, zeros, ndarray
from numpy.typing import NDArray

from .position import pos_c
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _ev_signal_c_s(time, alpha, mass_ratio, inc, c):
    """Scalar kernel for :func:`ev_signal_c`. See that function for documentation."""
    sin_inc = sin(inc)
    pre = -alpha * mass_ratio * sin_inc * sin_inc
    px, py, pz = pos_c(time, c)
    d2 = px * px + py * py + pz * pz
    d = sqrt(d2)
    cz = pz / d
    return pre * (2.0 * cz * cz - 1.0) / (d2 * d)


def _ev_signal_c_v_body(time, alpha, mass_ratio, inc, c):
    """Vector-kernel body for :func:`ev_signal_c`; see that function for documentation.

    Compiled twice: ``ev_signal_c_v`` is the serial kernel (``prange``
    compiles as a plain ``range`` without ``parallel=True``) and
    ``ev_signal_c_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    out = zeros(n)
    for j in prange(n):
        out[j] = _ev_signal_c_s(time[j], alpha, mass_ratio, inc, c)
    return out


ev_signal_c_v = njit(fastmath=True)(_ev_signal_c_v_body)
ev_signal_c_vp = njit(fastmath=True, parallel=True)(_ev_signal_c_v_body)


def ev_signal_c(time: float | NDArray, alpha: float, mass_ratio: float, inc: float,
                c: NDArray) -> float | NDArray:
    """
    Evaluate the ellipsoidal-variation signal at an expansion-point-centered time.

    Centered counterpart of `ev_signal`: assumes `time` has already been
    shifted to be relative to the expansion point. Returns the relative
    flux variation induced by the tidally distorted primary
    (Lillo-Box et al. 2014, Eqs. 6-10),
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1)/d^3` with
    :math:`c_z = z/d` and :math:`d = \\sqrt{x^2 + y^2 + z^2}` the
    instantaneous 3D star-planet distance in stellar radii.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. Enters only through the
        :math:`\\sin^2 i` projected-area factor.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    ev : float or NDArray
        Relative flux variation due to ellipsoidal distortion. Shape (N,)
        for an array `time`.

    Notes
    -----
    Uses the identity :math:`\\cos(2\\arccos u) = 2u^2 - 1` to skip a
    redundant arccos/cos pair.
    """
    if isinstance(time, ndarray):
        return ev_signal_c_v(time, alpha, mass_ratio, inc, c)
    return _ev_signal_c_s(time, alpha, mass_ratio, inc, c)


@overload(ev_signal_c, jit_options={'fastmath': True}, inline='always')
def _ev_signal_c_overload(time, alpha, mass_ratio, inc, c):
    if _is_1d_array(time):
        def impl(time, alpha, mass_ratio, inc, c):
            return ev_signal_c_v(time, alpha, mass_ratio, inc, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, alpha, mass_ratio, inc, c):
            return _ev_signal_c_s(time, alpha, mass_ratio, inc, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _ev_signal_s(time, alpha, mass_ratio, inc, tc, p, c, te):
    """Scalar kernel for :func:`ev_signal`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _ev_signal_c_s(time - (tc + te + epoch * p), alpha, mass_ratio, inc, c)


def _ev_signal_v_body(time, alpha, mass_ratio, inc, tc, p, c, te):
    """Vector-kernel body for :func:`ev_signal`; see that function for documentation.

    Compiled twice: ``ev_signal_v`` is the serial kernel (``prange``
    compiles as a plain ``range`` without ``parallel=True``) and
    ``ev_signal_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    out = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        out[j] = _ev_signal_c_s(time[j] - (tc + te + epoch * p), alpha, mass_ratio, inc, c)
    return out


ev_signal_v = njit(fastmath=True)(_ev_signal_v_body)
ev_signal_vp = njit(fastmath=True, parallel=True)(_ev_signal_v_body)


def ev_signal(time: float | NDArray, alpha: float, mass_ratio: float, inc: float,
              tc: float, p: float, c: NDArray, te: float = 0.0) -> float | NDArray:
    """
    Evaluate the ellipsoidal-variation signal at an absolute time.

    Folds the absolute observation time back to an expansion-point-centered offset
    and delegates to the centered kernel. Returns the relative flux
    variation induced by the tidally distorted primary,
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1)/d^3`, with :math:`d`
    the instantaneous 3D star-planet distance in stellar radii.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. Enters only through the
        :math:`\\sin^2 i` projected-area factor.
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
    ev : float or NDArray
        Relative flux variation due to ellipsoidal distortion. Shape (N,)
        for an array `time`.
    """
    if isinstance(time, ndarray):
        return ev_signal_v(time, alpha, mass_ratio, inc, tc, p, c, te)
    return _ev_signal_s(time, alpha, mass_ratio, inc, tc, p, c, te)


@overload(ev_signal, jit_options={'fastmath': True}, inline='always')
def _ev_signal_overload(time, alpha, mass_ratio, inc, tc, p, c, te=0.0):
    if _is_1d_array(time):
        def impl(time, alpha, mass_ratio, inc, tc, p, c, te=0.0):
            return ev_signal_v(time, alpha, mass_ratio, inc, tc, p, c, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, alpha, mass_ratio, inc, tc, p, c, te=0.0):
            return _ev_signal_s(time, alpha, mass_ratio, inc, tc, p, c, te)
        return impl
    return None
