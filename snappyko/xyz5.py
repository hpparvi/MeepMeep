#  ExOrbit: fast orbit calculations for exoplanet modelling
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
from numpy import cos, sin, floor, sqrt, zeros, linspace, arccos, pi

from .newton import ta_newton_s
from .utils import mean_anomaly


@njit(fastmath=True)
def solve_xyz_p5s(phase, p, a, i, e, w):
    """Planet velocity, acceleration, jerk, and snap at mid-transit in [R_star / day]"""

    # Time step for central finite difference
    # ---------------------------------------
    # I've tried to choose a value that is small enough to
    # work with ultra-short-period orbits and large enough
    # not to cause floating point problems with the fourth
    # derivative (anything much smaller starts hitting the
    # double precision limit.)
    dt = 2e-2

    ae = a*(1. - e**2)
    ci = cos(i)
    si = sin(i)

    # Calculation of X and Y positions
    # --------------------------------
    # These could just as well be calculated with a single
    # loop with X and Y as arrays, but I've decided to
    # manually unroll it because it seems to give a small
    # speed advantage with numba.

    f0 = ta_newton_s(phase-3 * dt, 0.0, p, e, w)
    f1 = ta_newton_s(phase-2 * dt, 0.0, p, e, w)
    f2 = ta_newton_s(phase-dt, 0.0, p, e, w)
    f3 = ta_newton_s(phase, 0.0, p, e, w)
    f4 = ta_newton_s(phase+dt, 0.0, p, e, w)
    f5 = ta_newton_s(phase+2 * dt, 0.0, p, e, w)
    f6 = ta_newton_s(phase+3 * dt, 0.0, p, e, w)

    r0 = ae/(1. + e*cos(f0))
    r1 = ae/(1. + e*cos(f1))
    r2 = ae/(1. + e*cos(f2))
    r3 = ae/(1. + e*cos(f3))
    r4 = ae/(1. + e*cos(f4))
    r5 = ae/(1. + e*cos(f5))
    r6 = ae/(1. + e*cos(f6))

    x0 = -r0*cos(w + f0)
    x1 = -r1*cos(w + f1)
    x2 = -r2*cos(w + f2)
    x3 = -r3*cos(w + f3)
    x4 = -r4*cos(w + f4)
    x5 = -r5*cos(w + f5)
    x6 = -r6*cos(w + f6)

    y0 = -r0*sin(w + f0)*ci
    y1 = -r1*sin(w + f1)*ci
    y2 = -r2*sin(w + f2)*ci
    y3 = -r3*sin(w + f3)*ci
    y4 = -r4*sin(w + f4)*ci
    y5 = -r5*sin(w + f5)*ci
    y6 = -r6*sin(w + f6)*ci

    z0 = r0*sin(w + f0)*si
    z1 = r1*sin(w + f1)*si
    z2 = r2*sin(w + f2)*si
    z3 = r3*sin(w + f3)*si
    z4 = r4*sin(w + f4)*si
    z5 = r5*sin(w + f5)*si
    z6 = r6*sin(w + f6)*si

    # First time derivative of position: velocity
    # -------------------------------------------
    a, b, c = 1/60, 9/60, 45/60
    vx = (a*(x6 - x0) + b*(x1 - x5) + c*(x4 - x2))/dt
    vy = (a*(y6 - y0) + b*(y1 - y5) + c*(y4 - y2))/dt
    vz = (a*(z6 - z0) + b*(z1 - z5) + c*(z4 - z2))/dt

    # Second time derivative of position: acceleration
    # ------------------------------------------------
    a, b, c, d = 1/90, 3/20, 3/2, 49/18
    ax = (a*(x0 + x6) - b*(x1 + x5) + c*(x2 + x4) - d*x3)/dt**2
    ay = (a*(y0 + y6) - b*(y1 + y5) + c*(y2 + y4) - d*y3)/dt**2
    az = (a*(z0 + z6) - b*(z1 + z5) + c*(z2 + z4) - d*z3)/dt**2

    # Third time derivative of position: jerk
    # ---------------------------------------
    a, b, c = 1/8, 1, 13/8
    jx = (a*(x0 - x6) + b*(x5 - x1) + c*(x2 - x4))/dt**3
    jy = (a*(y0 - y6) + b*(y5 - y1) + c*(y2 - y4))/dt**3
    jz = (a*(z0 - z6) + b*(z5 - z1) + c*(z2 - z4))/dt**3

    # Fourth time derivative of position: snap
    # ----------------------------------------
    a, b, c, d = 1/6, 2, 13/2, 28/3
    sx = (-a*(x0 + x6) + b*(x1 + x5) - c*(x2 + x4) + d*x3)/dt**4
    sy = (-a*(y0 + y6) + b*(y1 + y5) - c*(y2 + y4) + d*y3)/dt**4
    sz = (-a*(z0 + z6) + b*(z1 + z5) - c*(z2 + z4) + d*z3)/dt**4

    return x3, y3, z3, vx, vy, vz, ax, ay, az, jx, jy, jz, sx, sy, sz


@njit
def solve_xyz_o5s(p, a, i, e, w, npt):
    points = linspace(0.0, p, npt)
    dt = points[1] - points[0]
    coeffs = zeros((npt, 15))
    for ix in range(npt-1):
        coeffs[ix] = solve_xyz_p5s(points[ix], p, a, i, e, w)
    coeffs[-1] = coeffs[0]
    return dt, points, coeffs


@njit(fastmath=True)
def xyz_o5s(t, t0, p, dt, points, cf):
    """Calculate planet's (x, y, z) position for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = int(floor(tc / dt + 0.5))
    #cf[ix] = x0, y0, z0, vx, vy, vz, ax, ay, az, jx, jy, jz, sx, sy, sz
    tc -= points[ix]
    tc2 = tc * tc
    tc3 = tc2 * tc
    tc4 = tc3 * tc
    px = cf[ix, 0] + cf[ix, 3] * tc + 0.5 * cf[ix, 6] * tc2 + cf[ix, 9]  * tc3 / 6.0 + cf[ix, 12] * tc4 / 24.
    py = cf[ix, 1] + cf[ix, 4] * tc + 0.5 * cf[ix, 7] * tc2 + cf[ix, 10] * tc3 / 6.0 + cf[ix, 13] * tc4 / 24.
    pz = cf[ix, 2] + cf[ix, 5] * tc + 0.5 * cf[ix, 8] * tc2 + cf[ix, 11] * tc3 / 6.0 + cf[ix, 14] * tc4 / 24.
    return px, py, pz


@njit(fastmath=True)
def xyz_o5v(times, t0, p, dt, points, coeffs):
    """Calculate planet's (x, y, z) position for a vector time for any orbital phase"""
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = xyz_o5s(times[i], t0, p, dt, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True)
def vxyz_o5s(t, t0, p, dt, points, cf):
    """Calculate planet's (x, y, z) velocity for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = int(floor(tc / dt + 0.5))
    tc -= points[ix]
    tc2 = tc * tc
    tc3 = tc2 * tc
    vx = cf[ix, 3] + cf[ix, 6] * tc + 0.5 * cf[ix, 9] * tc2 + cf[ix, 12]  * tc3 / 6.0
    vy = cf[ix, 4] + cf[ix, 7] * tc + 0.5 * cf[ix, 10] * tc2 + cf[ix, 13] * tc3 / 6.0
    vz = cf[ix, 5] + cf[ix, 8] * tc + 0.5 * cf[ix, 11] * tc2 + cf[ix, 14] * tc3 / 6.0
    return vx, vy, vz


@njit(fastmath=True)
def vxyz_o5v(times, t0, p, dt, points, coeffs):
    """Calculate planet's (x, y, z) position for a vector time for any orbital phase"""
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = vxyz_o5s(times[i], t0, p, dt, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True)
def vz_o5s(t, t0, p, dt, points, cf):
    """Calculate planet's (x, y, z) velocity for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = int(floor(tc / dt + 0.5))
    tc -= points[ix]
    tc2 = tc * tc
    tc3 = tc2 * tc
    return cf[ix, 5] + cf[ix, 8] * tc + 0.5 * cf[ix, 11] * tc2 + cf[ix, 14] * tc3 / 6.0


@njit(fastmath=True)
def vz_o5v(times, t0, p, dt, points, coeffs):
    """Calculate planet's (x, y, z) position for a vector time for any orbital phase"""
    npt = times.size
    vzs = zeros(npt)
    for i in range(npt):
        vzs[i] = vz_o5s(times[i], t0, p, dt, points, coeffs)
    return vzs


@njit(fastmath=True)
def pd_o5s(t, t0, p, dt, points, cf):
    """Calculate the projected planet-star center distance for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = int(floor(tc / dt + 0.5))
    tc -= points[ix]
    tc2 = tc * tc
    tc3 = tc2 * tc
    tc4 = tc3 * tc
    px = cf[ix, 0] + cf[ix, 3] * tc + 0.5 * cf[ix, 6] * tc2 + cf[ix, 9]  * tc3 / 6.0 + cf[ix, 12] * tc4 / 24.
    py = cf[ix, 1] + cf[ix, 4] * tc + 0.5 * cf[ix, 7] * tc2 + cf[ix, 10] * tc3 / 6.0 + cf[ix, 13] * tc4 / 24.
    return sqrt(px**2 + py**2)


@njit(fastmath=True)
def z_o5s(t, t0, p, dt, points, cf):
    """Calculate planet's (z) position for a scalar time for any orbital phase"""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = int(floor(tc / dt + 0.5))
    tc -= points[ix]
    tc2 = tc * tc
    tc3 = tc2 * tc
    tc4 = tc3 * tc
    pz = cf[ix, 2] + cf[ix, 5] * tc + 0.5 * cf[ix, 8] * tc2 + cf[ix, 11] * tc3 / 6.0 + cf[ix, 14] * tc4 / 24.
    return pz


@njit(fastmath=True)
def z_o5v(times, t0, p, dt, points, coeffs):
    """Calculate planet's (z) position for a vector time for any orbital phase"""
    npt = times.size
    zs = zeros(npt)
    for i in range(npt):
        zs[i] = z_o5s(times[i], t0, p, dt, points, coeffs)
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
def true_anomaly_o5v(times, t0, p, ex, ey, ez, w, dt, points, coeffs):
    npt = times.size
    f = zeros(npt)
    if ex <= -0.9999:
        f[:] = mean_anomaly(times, t0, p, 0.0, w)
    else:
        nes = (ex**2 + ey**2 + ez**2)
        for i in range(npt):
            x, y, z = xyz_o5s(times[i], t0, p, dt, points, coeffs)
            vx, vy, vz = vxyz_o5s(times[i], t0, p, dt, points, coeffs)
            edp = (x*ex + y*ey + z*ez) / sqrt((x**2 + y**2 + z**2) * nes)

            if (x*vx + y*vy + z*vz) > 0.0:
                f[i] = arccos(edp)
            else:
                f[i] = 2.0*pi - arccos(edp)
    return f

@njit
def cos_v_p_angle_o5v(v, times, t0, p, dt, points, coeffs):
    px, py, pz =  xyz_o5v(times, t0, p, dt, points, coeffs)
    np = sqrt(px**2 + py**2 + pz**2)
    nv = sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return (px*v[0] + py*v[1] + pz*v[2])/(np*nv)

@njit
def cos_alpha_o5s(t, t0, p, dt, points, coeffs):
    """Cosine of the phase angle."""
    x, y, z = xyz_o5s(t, t0, p, dt, points, coeffs)
    return -z / sqrt(x**2 + y**2 + z**2)


@njit
def cos_alpha_o5v(times, t0, p, dt, points, coeffs):
    """Cosine of the phase angle."""
    x, y, z = xyz_o5v(times, t0, p, dt, points, coeffs)
    return -z / sqrt(x**2 + y**2 + z**2)


@njit
def star_planet_distance_o5v(times, t0, p, dt, points, coeffs):
    x, y, z =  xyz_o5v(times, t0, p, dt, points, coeffs)
    return sqrt(x**2 + y**2 + z**2)


@njit
def ev_signal_o5v(alpha, mass_ratio, inc, times, t0, p, dt, points, coeffs):
    """Ellipsoidal variation signal.

    NOTES: See Eqs. 6-10 in Lillo-Box al. (2014).
    """
    x, y, z =  xyz_o5v(times, t0, p, dt, points, coeffs)
    distance = sqrt(x**2 + y**2 + z**2)
    theta = arccos(z / distance)
    return -alpha * mass_ratio * sin(inc)**2 * cos(2*theta) / distance**3


@njit
def zdiff_o5s(t, t0, p, dt, points, coeffs):
    return z_o5s(t, t0, p, dt, points, coeffs) - coeffs[0, 2]


@njit
def light_travel_time_o5s(t, t0, p, rstar, dt, points, coeffs):
    """Light travel time in days."""
    s = 2.685885891543453e-05  # ((1 * u.R_sun).to(u.m) / c.c).to('d').value
    return -zdiff_o5s(t, t0, p, dt, points, coeffs) * rstar * s


@njit
def light_travel_time_o5v(times, t0, p, rstar, dt, points, coeffs):
    """Light travel time in days."""
    s = 2.685885891543453e-05  # ((1 * u.R_sun).to(u.m) / c.c).to('d').value
    ltt = zeros(times.size)
    for i in range(times.size):
        ltt[i] = -zdiff_o5s(times[i], t0, p, dt, points, coeffs) * rstar * s
    return ltt


@njit
def rv_o5s(time, k, t0, p, a, i, e, dt, points, coeffs):
    n = 2*pi/p * (a*sin(i))/sqrt(1-e**2)  # Perryman (2018) Eq. 2.23
    return vz_o5s(time, t0, p, dt, points, coeffs) / n * k


@njit
def rv_o5v(times, k, t0, p, a, i, e, dt, points, coeffs):
    npt = times.size
    rvs = zeros(npt)
    n = 2*pi/p * (a*sin(i))/sqrt(1-e**2)  # Perryman (2018) Eq. 2.23
    for i in range(npt):
        rvs[i] = vz_o5s(times[i], t0, p, dt, points, coeffs) / n * k
    return rvs