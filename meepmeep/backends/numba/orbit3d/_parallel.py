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

"""Parallel (prange) twins of the multi-knot value vector kernels.

Every ``_X_ov`` kernel has a ``_X_ovp`` twin here, compiled with
``parallel=True`` and a ``prange`` sample loop but otherwise identical:
the same scalar kernels, the same hoisted invariants, the same output
layout. Each loop iteration touches only its own output elements, so no
extra synchronisation or scratch is needed.

The twins are opt-in (used by :class:`~meepmeep.orbit.Orbit` when
constructed with ``parallel=True``); the public ``X_o`` dispatchers always
route to the serial kernels. Parallelisation pays only for large time
arrays - the parallel-region launch costs tens of microseconds, so for
the value kernels the break-even is around 5e4 samples.
"""

from numba import njit, prange
from numpy import zeros, pi, sin, sqrt, arccos, cos, floor

from ..utils import mean_anomaly_at_transit
from .position import _pos_os
from .zposition import _zpos_os
from .separation import _sep_os
from .velocity import _vel_os
from .zvelocity import _zvel_os
from .phase_angle import _cos_alpha_os
from .projected_angle import _cos_v_p_angle_os
from .star_planet_distance import _star_planet_distance_os
from .ev_signal import _ev_signal_os
from .lambert import _lambert_kernel
from .light_travel_time import LTT_DAYS_PER_RSUN
from .true_anomaly import _true_anomaly_os


@njit(fastmath=True, parallel=True)
def _pos_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.position._pos_ov`."""
    n = times.size
    xs, ys, zs = zeros(n), zeros(n), zeros(n)
    for i in prange(n):
        xs[i], ys[i], zs[i] = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True, parallel=True)
def _zpos_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.zposition._zpos_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _zpos_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def _sep_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.separation._sep_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _sep_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def _vel_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.velocity._vel_ov`."""
    n = times.size
    vxs, vys, vzs = zeros(n), zeros(n), zeros(n)
    for i in prange(n):
        vxs[i], vys[i], vzs[i] = _vel_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return vxs, vys, vzs


@njit(fastmath=True, parallel=True)
def _zvel_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.zvelocity._zvel_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _zvel_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def _rv_ovp(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.radial_velocity._rv_ov`."""
    n = times.size
    rvs = zeros(n)
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    for j in prange(n):
        rvs[j] = _zvel_os(times[j], tpa, p, dt, pktable, points, coeffs) * scale
    return rvs


@njit(fastmath=True, parallel=True)
def _cos_alpha_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.phase_angle._cos_alpha_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def _cos_v_p_angle_ovp(v, times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.projected_angle._cos_v_p_angle_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _cos_v_p_angle_os(v, times[i], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(fastmath=True, parallel=True)
def _star_planet_distance_ovp(times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.star_planet_distance._star_planet_distance_ov`."""
    n = times.size
    out = zeros(n)
    for i in prange(n):
        out[i] = _star_planet_distance_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return out


@njit(parallel=True)
def _true_anomaly_ovp(times, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.true_anomaly._true_anomaly_ov`."""
    n = times.size
    f = zeros(n)
    for i in prange(n):
        f[i] = _true_anomaly_os(times[i], tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs)
    return f


@njit(fastmath=True, parallel=True)
def _lambert_phase_curve_ovp(times, ag, a, k, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.lambert._lambert_phase_curve_ov`."""
    n = times.size
    res = zeros(n)
    amplitude = k * k * ag / (a * a)
    for i in prange(n):
        cos_alpha = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
        phase, _ = _lambert_kernel(cos_alpha)
        res[i] = amplitude * phase
    return res


@njit(fastmath=True, parallel=True)
def _lambert_and_emission_ovp(times, ag, fr_night, fr_day, emi_offset, a, k,
                              tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.lambert._lambert_and_emission_ov`."""
    n = times.size
    ref, emi = zeros(n), zeros(n)
    k2 = k * k
    aref = k2 * ag / (a * a)
    for i in prange(n):
        cos_alpha = _cos_alpha_os(times[i], tpa, p, dt, pktable, points, coeffs)
        phase, alpha = _lambert_kernel(cos_alpha)
        ref[i] = aref * phase
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


@njit(fastmath=True, parallel=True)
def _ev_signal_ovp(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.ev_signal._ev_signal_ov`."""
    n = times.size
    out = zeros(n)
    sin2_inc = sin(inc) ** 2
    pre = -alpha * mass_ratio * sin2_inc
    for i in prange(n):
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        out[i] = pre * (2.0 * cz * cz - 1.0) / (d2 * d)
    return out


@njit(fastmath=True, parallel=True)
def _light_travel_time_ovp(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3d.light_travel_time._light_travel_time_ov`."""
    n = times.size
    ltt = zeros(n)
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_tr = _zpos_os(tpa + to, tpa, p, dt, pktable, points, coeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    for j in prange(n):
        ltt[j] = factor * (_zpos_os(times[j], tpa, p, dt, pktable, points, coeffs) - z_tr)
    return ltt
