#  MeepMeep: fast orbit calculations for exoplanet modelling
#  Copyright (C) 2022 Hannu Parviainen
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

"""Multi-knot Taylor-series evaluators over a full orbit.

The functions in this module evaluate orbit-spanning quantities (position,
velocity, projected distance, phase angle, radial velocity, etc.) at
arbitrary times by looking up the appropriate knot via ``pktable`` and then
delegating to the single-knot evaluators in ``position3d``/``velocity3d``.

Coefficient layout: ``coeffs`` is an ``(npt, 3, 5)`` array as produced by
``solve3d_orbit`` — ``coeffs[ix]`` is the ``(3, 5)`` matrix consumed by
``p3dc``, ``v3dc``, ``vzc``, ``d3dc``, and ``z3dc``.
"""

from numba import njit
from numpy import zeros, pi, floor, sqrt, sin, cos, arccos

from .position3d import p3dc, d3dc, z3dc
from .velocity3d import v3dc, vzc
from .solve3d import solve3d
from ..utils import mean_anomaly, mean_anomaly_at_transit


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

@njit
def solve3d_orbit(knot_times, p, a, i, e, w, npt):
    """Pre-compute Taylor coefficients at every knot of one orbit."""
    coeffs = zeros((npt, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        coeffs[ix, :, :] = solve3d(p * knot_times[ix] - to, p, a, i, e, w)
    coeffs[-1, :, :] = coeffs[0]
    return coeffs


@njit(fastmath=True)
def knot_ix(t, t0, p, dt, pktable) -> int:
    """Return the knot index for a single time."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    return pktable[int(floor(tc / (dt * p)))]


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def xyz_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position at scalar time `t` for any orbital phase."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return p3dc(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def xyz_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position at an array of times."""
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True)
def z_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Planet z-position at scalar time `t` for any orbital phase."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return z3dc(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def z_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Planet z-position at an array of times."""
    npt = times.size
    zs = zeros(npt)
    for i in range(npt):
        zs[i] = z_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return zs


@njit(fastmath=True)
def pd_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Projected planet-star distance at scalar time `t` for any orbital phase."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return d3dc(tc - points[ix] * p, coeffs[ix])


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def vxyz_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity at scalar time `t` for any orbital phase."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return v3dc(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def vxyz_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity at an array of times."""
    npt = times.size
    vxs, vys, vzs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        vxs[i], vys[i], vzs[i] = vxyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return vxs, vys, vzs


@njit(fastmath=True)
def vz_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Planet z-velocity at scalar time `t` for any orbital phase."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return vzc(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def vz_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Planet z-velocity at an array of times."""
    npt = times.size
    vzs = zeros(npt)
    for i in range(npt):
        vzs[i] = vz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return vzs


# ---------------------------------------------------------------------------
# Anomalies and angles
# ---------------------------------------------------------------------------

@njit
def true_anomaly_o5v(times, t0, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """True anomaly from the angle between the position and the eccentricity vector."""
    npt = times.size
    f = zeros(npt)
    if ex <= -0.9999:
        # Circular-orbit fallback: true anomaly == mean anomaly.
        f[:] = mean_anomaly(times, t0, p, 0.0, w)
    else:
        nes = ex ** 2 + ey ** 2 + ez ** 2
        for i in range(npt):
            x, y, z = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
            vx, vy, vz = vxyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
            edp = (x * ex + y * ey + z * ez) / sqrt((x ** 2 + y ** 2 + z ** 2) * nes)

            if edp <= -1.0:
                f[i] = pi
            elif edp >= 1.0:
                f[i] = 0.0
            elif (x * vx + y * vy + z * vz) > 0.0:
                f[i] = arccos(edp)
            else:
                f[i] = 2.0 * pi - arccos(edp)
    return f


@njit
def cos_v_p_angle_o5v(v, times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the angle between the planet position and a fixed reference vector `v`."""
    px, py, pz = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    np_ = sqrt(px ** 2 + py ** 2 + pz ** 2)
    nv = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (px * v[0] + py * v[1] + pz * v[2]) / (np_ * nv)


@njit
def cos_alpha_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at scalar time `t`."""
    x, y, z = xyz_o5s(t, t0, p, dt, pktable, points, coeffs)
    return -z / sqrt(x ** 2 + y ** 2 + z ** 2)


@njit
def cos_alpha_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at an array of times."""
    x, y, z = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    return -z / sqrt(x ** 2 + y ** 2 + z ** 2)


@njit
def star_planet_distance_o5v(times, t0, p, dt, pktable, points, coeffs):
    """3D star-planet distance at an array of times."""
    x, y, z = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    return sqrt(x ** 2 + y ** 2 + z ** 2)


# ---------------------------------------------------------------------------
# Photometric/RV signals
# ---------------------------------------------------------------------------

@njit
def lambert_phase_curve_o5s(time, ag, a, k, t0, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution at a scalar time."""
    amplitude = k ** 2 * ag / a ** 2
    cos_alpha = cos_alpha_o5s(time, t0, p, dt, pktable, points, coeffs)
    alpha = arccos(cos_alpha)
    return amplitude * (sin(alpha) + (pi - alpha) * cos_alpha) / pi


@njit
def lambert_phase_curve_o5v(times, ag, a, k, t0, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution at an array of times."""
    npt = times.size
    res = zeros(npt)
    amplitude = k ** 2 * ag / a ** 2
    for i in range(npt):
        cos_alpha = cos_alpha_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        alpha = arccos(cos_alpha)
        res[i] = amplitude * (sin(alpha) + (pi - alpha) * cos_alpha) / pi
    return res


@njit
def lambert_and_emission_o5v(times, ag, fr_night, fr_day, emi_offset, a, k,
                             t0, p, dt, pktable, points, coeffs):
    """Lambertian reflection plus a simple cosine-emission day/night model."""
    npt = times.size
    ref, emi = zeros(npt), zeros(npt)
    k2 = k ** 2
    aref = k2 * ag / a ** 2
    for i in range(npt):
        cos_alpha = cos_alpha_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        alpha = arccos(cos_alpha)
        ref[i] = aref * (sin(alpha) + (pi - alpha) * cos_alpha) / pi
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


@njit
def ev_signal_o5v(alpha, mass_ratio, inc, times, t0, p, dt, pktable, points, coeffs):
    """Ellipsoidal variation signal (Lillo-Box et al. 2014, Eqs. 6–10)."""
    x, y, z = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    distance = sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = arccos(z / distance)
    return -alpha * mass_ratio * sin(inc) ** 2 * cos(2 * theta) / distance ** 3


@njit
def rv_o5v(times, k, t0, p, a, i, e, dt, pktable, points, coeffs):
    """Radial velocity at an array of times (Perryman 2018, Eq. 2.23)."""
    npt = times.size
    rvs = zeros(npt)
    n = 2 * pi / p * (a * sin(i)) / sqrt(1 - e ** 2)
    for j in range(npt):
        rvs[j] = vz_o5s(times[j], t0, p, dt, pktable, points, coeffs) / n * k
    return rvs
