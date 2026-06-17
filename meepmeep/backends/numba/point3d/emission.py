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

"""Single-expansion-point cosine emission phase-curve evaluators.

Holds a simple cosine thermal-emission phase curve evaluated from a single
Taylor expansion (:func:`emission_phase_curve_c` / :func:`emission_phase_curve`).
The model is

    F = k^2 * fratio * (1 + cos(offset) * cz + sin(offset) * s) / 2,

a phase function peaking at ``k^2 * fratio`` (hotspot fully in view) and
falling to ``0`` on the far side. ``cz = -z / d`` is the cosine of the phase
angle and ``s = -(n_x * y - n_y * x) / d`` is the signed in-plane component,
with ``n = (r x v) / |r x v|`` the (time-invariant) orbital normal recovered
from the position and velocity at the expansion point. The hotspot ``offset``
shifts the peak away from secondary eclipse, breaking the time symmetry of the
curve. The multi-expansion-point ``orbit3d`` dispatchers delegate here.
"""

from numba import njit, prange, types
from numba.extending import overload
from numpy import floor, sin, cos, sqrt, zeros, ndarray
from numpy.typing import NDArray

from .position import pos_c
from .velocity import vel_c
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _emission_phase_curve_c_s(time, k, fratio, offset, c):
    """Scalar kernel for :func:`emission_phase_curve_c`. See that function for documentation."""
    x, y, z = pos_c(time, c)
    vx, vy, vz = vel_c(time, c)
    d = sqrt(x * x + y * y + z * z)
    wx = y * vz - z * vy
    wy = z * vx - x * vz
    wz = x * vy - y * vx
    el = sqrt(wx * wx + wy * wy + wz * wz)
    cz = -z / d
    m = wx * y - wy * x
    s = -m / (el * d)
    g = 0.5 * (1.0 + cos(offset) * cz + sin(offset) * s)
    return k * k * fratio * g


def _emission_phase_curve_c_v_body(time, k, fratio, offset, c):
    """Vector-kernel body for :func:`emission_phase_curve_c`; see that function for documentation.

    Compiled twice: ``emission_phase_curve_c_v`` is the serial kernel
    (``prange`` compiles as a plain ``range`` without ``parallel=True``) and
    ``emission_phase_curve_c_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    out = zeros(n)
    for j in prange(n):
        out[j] = _emission_phase_curve_c_s(time[j], k, fratio, offset, c)
    return out


emission_phase_curve_c_v = njit(fastmath=True)(_emission_phase_curve_c_v_body)
emission_phase_curve_c_vp = njit(fastmath=True, parallel=True)(_emission_phase_curve_c_v_body)


def emission_phase_curve_c(time: float | NDArray, k: float, fratio: float, offset: float,
                           c: NDArray) -> float | NDArray:
    """
    Evaluate the cosine emission phase-curve flux at an expansion-point-centered time.

    Centered counterpart of `emission_phase_curve`: assumes `time` has
    already been shifted to be relative to the expansion point. Returns the
    planet-to-star flux ratio of a simple cosine thermal-emission model,
    :math:`F = k^2\\,f_\\mathrm{ratio}\\,(1 + \\cos\\delta\\,c_z + \\sin\\delta\\,s)/2`,
    where :math:`c_z = -z/d` is the cosine of the phase angle, :math:`s` the
    signed in-plane component built from the orbital normal, and
    :math:`\\delta` the hotspot offset. The flux peaks at
    :math:`k^2 f_\\mathrm{ratio}` when the hotspot faces the observer and
    falls to 0 on the far side.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    fratio : float
        Dayside-to-nightside per-surface-element flux ratio, scaling the
        phase-curve amplitude so the peak-to-peak swing is
        :math:`k^2 f_\\mathrm{ratio}`.
    offset : float
        Hotspot offset [radians]. Shifts the emission peak away from
        secondary eclipse.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Both the position
        and velocity (its time derivative) are read.

    Returns
    -------
    flux : float or NDArray
        Reflected/emitted planet-to-star flux ratio. Shape (N,) for an array
        `time`.
    """
    if isinstance(time, ndarray):
        return emission_phase_curve_c_v(time, k, fratio, offset, c)
    return _emission_phase_curve_c_s(time, k, fratio, offset, c)


@overload(emission_phase_curve_c, jit_options={'fastmath': True}, inline='always')
def _emission_phase_curve_c_overload(time, k, fratio, offset, c):
    if _is_1d_array(time):
        def impl(time, k, fratio, offset, c):
            return emission_phase_curve_c_v(time, k, fratio, offset, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, k, fratio, offset, c):
            return _emission_phase_curve_c_s(time, k, fratio, offset, c)
        return impl
    return None


@njit(fastmath=True, inline='always')
def _emission_phase_curve_s(time, k, fratio, offset, tc, p, c, te):
    """Scalar kernel for :func:`emission_phase_curve`. See that function for documentation."""
    epoch = floor((time - tc - te + 0.5 * p) / p)
    return _emission_phase_curve_c_s(time - (tc + te + epoch * p), k, fratio, offset, c)


def _emission_phase_curve_v_body(time, k, fratio, offset, tc, p, c, te):
    """Vector-kernel body for :func:`emission_phase_curve`; see that function for documentation.

    Compiled twice: ``emission_phase_curve_v`` is the serial kernel
    (``prange`` compiles as a plain ``range`` without ``parallel=True``) and
    ``emission_phase_curve_vp`` the parallel twin. The loop writes only into
    per-sample output elements, so no per-thread scratch is needed.
    """
    n = time.size
    out = zeros(n)
    for j in prange(n):
        epoch = floor((time[j] - tc - te + 0.5 * p) / p)
        out[j] = _emission_phase_curve_c_s(time[j] - (tc + te + epoch * p), k, fratio, offset, c)
    return out


emission_phase_curve_v = njit(fastmath=True)(_emission_phase_curve_v_body)
emission_phase_curve_vp = njit(fastmath=True, parallel=True)(_emission_phase_curve_v_body)


def emission_phase_curve(time: float | NDArray, k: float, fratio: float, offset: float,
                         tc: float, p: float, c: NDArray, te: float = 0.0) -> float | NDArray:
    """
    Evaluate the cosine emission phase-curve flux at an absolute time.

    Folds the absolute observation time back to an expansion-point-centered offset
    and delegates to the centered kernel. Returns the planet-to-star flux
    ratio of a simple cosine thermal-emission model with a hotspot offset
    (see `emission_phase_curve_c`).

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    fratio : float
        Dayside-to-nightside per-surface-element flux ratio (amplitude
        scaling); the peak-to-peak swing is :math:`k^2 f_\\mathrm{ratio}`.
    offset : float
        Hotspot offset [radians].
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
        Emitted planet-to-star flux ratio. Shape (N,) for an array `time`.
    """
    if isinstance(time, ndarray):
        return emission_phase_curve_v(time, k, fratio, offset, tc, p, c, te)
    return _emission_phase_curve_s(time, k, fratio, offset, tc, p, c, te)


@overload(emission_phase_curve, jit_options={'fastmath': True}, inline='always')
def _emission_phase_curve_overload(time, k, fratio, offset, tc, p, c, te=0.0):
    if _is_1d_array(time):
        def impl(time, k, fratio, offset, tc, p, c, te=0.0):
            return emission_phase_curve_v(time, k, fratio, offset, tc, p, c, te)
        return impl
    if isinstance(time, types.Float):
        def impl(time, k, fratio, offset, tc, p, c, te=0.0):
            return _emission_phase_curve_s(time, k, fratio, offset, tc, p, c, te)
        return impl
    return None
