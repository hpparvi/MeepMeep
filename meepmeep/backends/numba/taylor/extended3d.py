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

from numba import njit
from numpy import cos, sin, floor, sqrt, zeros, linspace, arccos, pi, ndarray

from . import solve3d
from ..utils import mean_anomaly, mean_anomaly_at_transit


@njit(fastmath=True)
def xyz_o5s(t, t0, p, dt, pktable, points, cf):
    """Calculate planet's (x, y, z) position for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt*p)))]
    tc -= points[ix] * p
    tc2 = tc * tc
    tc3 = tc2 * tc
    tc4 = tc3 * tc
    px = cf[ix, 0] + cf[ix, 3] * tc + 0.5 * cf[ix, 6] * tc2 + cf[ix, 9]  * tc3 / 6.0 + cf[ix, 12] * tc4 / 24.
    py = cf[ix, 1] + cf[ix, 4] * tc + 0.5 * cf[ix, 7] * tc2 + cf[ix, 10] * tc3 / 6.0 + cf[ix, 13] * tc4 / 24.
    pz = cf[ix, 2] + cf[ix, 5] * tc + 0.5 * cf[ix, 8] * tc2 + cf[ix, 11] * tc3 / 6.0 + cf[ix, 14] * tc4 / 24.
    return px, py, pz


@njit(fastmath=True)
def xyz_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Calculate planet's (x, y, z) position for a vector time for any orbital phase"""
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True)
def vxyz_o5s(t, t0, p, dt, pktable, points, cf):
    """Calculate planet's (x, y, z) velocity for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt*p)))]
    tc -= points[ix] * p
    tc2 = tc * tc
    tc3 = tc2 * tc
    vx = cf[ix, 3] + cf[ix, 6] * tc + 0.5 * cf[ix, 9] * tc2 + cf[ix, 12]  * tc3 / 6.0
    vy = cf[ix, 4] + cf[ix, 7] * tc + 0.5 * cf[ix, 10] * tc2 + cf[ix, 13] * tc3 / 6.0
    vz = cf[ix, 5] + cf[ix, 8] * tc + 0.5 * cf[ix, 11] * tc2 + cf[ix, 14] * tc3 / 6.0
    return vx, vy, vz


@njit(fastmath=True)
def vxyz_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Calculate planet's (x, y, z) position for a vector time for any orbital phase"""
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = vxyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True)
def vz_o5s(t, t0, p, dt, pktable, points, cf):
    """Calculate planet's (x, y, z) velocity for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt*p)))]
    tc -= points[ix] * p
    tc2 = tc * tc
    tc3 = tc2 * tc
    return cf[ix, 5] + cf[ix, 8] * tc + 0.5 * cf[ix, 11] * tc2 + cf[ix, 14] * tc3 / 6.0


@njit(fastmath=True)
def vz_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Calculate planet's (x, y, z) position for a vector time for any orbital phase"""
    npt = times.size
    vzs = zeros(npt)
    for i in range(npt):
        vzs[i] = vz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return vzs


@njit(fastmath=True)
def pd_o5s(t, t0, p, dt, pktable, points, cf):
    """Calculate the projected planet-star center distance for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt*p)))]
    tc -= points[ix] * p
    tc2 = tc * tc
    tc3 = tc2 * tc
    tc4 = tc3 * tc
    px = cf[ix, 0] + cf[ix, 3] * tc + 0.5 * cf[ix, 6] * tc2 + cf[ix, 9]  * tc3 / 6.0 + cf[ix, 12] * tc4 / 24.
    py = cf[ix, 1] + cf[ix, 4] * tc + 0.5 * cf[ix, 7] * tc2 + cf[ix, 10] * tc3 / 6.0 + cf[ix, 13] * tc4 / 24.
    return sqrt(px**2 + py**2)



@njit(fastmath=True)
def z_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Calculate planet's (z) position for a vector time for any orbital phase"""
    npt = times.size
    zs = zeros(npt)
    for i in range(npt):
        zs[i] = z_o5s(times[i], t0, p, dt, pktable, points, coeffs)
    return zs


@njit(fastmath=True)
def xyz_t15s(tc, t0, p, x0, y0, z0, vx, vy, vz, ax, ay, az, jx, jy, jz, sx, sy, sz):
    """Calculate planet's (x,y) position near transit."""
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    px = x0 + vx * t + 0.5 * ax * t2 + jx * t3 / 6.0 + sx * t4 / 24.
    py = y0 + vy * t + 0.5 * ay * t2 + jy * t3 / 6.0 + sy * t4 / 24.
    pz = z0 + vz * t + 0.5 * az * t2 + jz * t3 / 6.0 + sz * t4 / 24.
    return px, py, pz


@njit
def true_anomaly_o5v(times, t0, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    npt = times.size
    f = zeros(npt)
    if ex <= -0.9999:
        f[:] = mean_anomaly(times, t0, p, 0.0, w)
    else:
        nes = (ex**2 + ey**2 + ez**2)
        for i in range(npt):
            x, y, z = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
            vx, vy, vz = vxyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
            edp = (x*ex + y*ey + z*ez) / sqrt((x**2 + y**2 + z**2) * nes)

            if edp <= -1.0:
                f[i] = pi
            elif edp >= 1.0:
                f[i] = 0.0
            elif (x*vx + y*vy + z*vz) > 0.0:
                f[i] = arccos(edp)
            else:
                f[i] = 2.0*pi - arccos(edp)
    return f


@njit
def cos_v_p_angle_o5v(v, times, t0, p, dt, pktable, points, coeffs):
    px, py, pz = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    np = sqrt(px**2 + py**2 + pz**2)
    nv = sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (px*v[0] + py*v[1] + pz*v[2])/(np*nv)


@njit
def cos_alpha_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle."""
    x, y, z = xyz_o5s(t, t0, p, dt, pktable, points, coeffs)
    return -z / sqrt(x**2 + y**2 + z**2)


@njit
def cos_alpha_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle."""
    x, y, z = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    return -z / sqrt(x**2 + y**2 + z**2)


@njit
def star_planet_distance_o5v(times, t0, p, dt, pktable, points, coeffs):
    x, y, z = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    return sqrt(x**2 + y**2 + z**2)

@njit
def lambert_phase_curve_o5s(time, ag, a, k, t0, p, dt, pktable, points, coeffs) -> ndarray:
    """Compute the Lambertian phase curve for a single time value.

    Parameters
    ----------
    time : array-like
        Array of time values for which the phase curve is calculated.
    ag : float
        Geometric albedo of the reflecting object.
    a : float
        Scaled orbital semi-major axis.
    k : float
        Planet-star radius ratio.
    t0 : float
        Reference time.
    p : float
        Orbital period.
    dt : float
        Time resolution or step size for calculations.
    pktable : array-like
        Precomputed table of phase coefficients.
    points : array-like
        Grid points used for modeling the phase curve.
    coeffs : array-like
        Coefficients for Taylor series expansion of the phase curve model.

    Returns
    -------
    ndarray
        Computed Lambertian phase curve values corresponding to the input time array.
    """
    amplitude = k**2 * ag / a**2
    cos_alpha = cos_alpha_o5s(time, t0, p, dt, pktable, points, coeffs)
    alpha = arccos(cos_alpha)
    return amplitude * (sin(alpha) + (pi - alpha) * cos_alpha) / pi

@njit
def lambert_phase_curve_o5v(times, ag, a, k, t0, p, dt, pktable, points, coeffs):
    npt = times.size
    res = zeros(npt)
    amplitude = k**2 * ag / a**2
    for i in range(npt):
        cos_alpha = cos_alpha_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        alpha = arccos(cos_alpha)
        res[i] = amplitude * (sin(alpha) + (pi - alpha) * cos_alpha) / pi
    return res

@njit
def lambert_and_emission_o5v(times, ag, fr_night, fr_day, emi_offset, a, k, t0, p, dt, pktable, points, coeffs):
    npt = times.size
    ref, emi = zeros(npt), zeros(npt)
    k2 = k**2
    aref = k2 * ag / a**2
    for i in range(npt):
        cos_alpha = cos_alpha_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        alpha = arccos(cos_alpha)
        ref[i] = aref * (sin(alpha) + (pi - alpha) * cos_alpha) / pi
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi

@njit
def ev_signal_o5v(alpha, mass_ratio, inc, times, t0, p, dt, pktable, points, coeffs):
    """Ellipsoidal variation signal.

    NOTES: See Eqs. 6-10 in Lillo-Box al. (2014).
    """
    x, y, z = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)
    distance = sqrt(x**2 + y**2 + z**2)
    theta = arccos(z / distance)
    return -alpha * mass_ratio * sin(inc)**2 * cos(2*theta) / distance**3





@njit
def rv_o5v(times, k, t0, p, a, i, e, dt, pktable, points, coeffs):
    npt = times.size
    rvs = zeros(npt)
    n = 2*pi/p * (a*sin(i))/sqrt(1-e**2)  # Perryman (2018) Eq. 2.23
    for i in range(npt):
        rvs[i] = vz_o5s(times[i], t0, p, dt, pktable, points, coeffs) / n * k
    return rvs