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
    """Pre-compute Taylor coefficients at every knot of one orbit.

    Requires ``knot_times[-1]`` to be the periodic image of ``knot_times[0]``
    (i.e. one full period later in normalised phase). ``knots.create_knots``
    enforces this; if you build ``knot_times`` by hand, ensure the contract
    holds — the last knot's coefficients are copied from the first instead
    of being recomputed.
    """
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
    nes = ex * ex + ey * ey + ez * ez
    # Circular-orbit fast path: ``utils.eccentricity_vector`` returns the
    # sentinel ``[-1, 0, 0]`` when ``e < 1e-5``. Detecting it here lets us
    # short-circuit to mean anomaly (which equals true anomaly for a
    # circular orbit) and avoid the geometric path's noisy small-``nes``
    # arithmetic.
    if ex <= -0.9999 and nes > 0.99:
        f[:] = mean_anomaly(times, t0, p, 0.0, w)
    else:
        for i in range(npt):
            t = times[i]
            epoch = floor((t - t0) / p)
            tc = t - t0 - epoch * p
            ix = pktable[int(floor(tc / (dt * p)))]
            tcc = tc - points[ix] * p
            c = coeffs[ix]
            x, y, z = p3dc(tcc, c)
            vx, vy, vz = v3dc(tcc, c)
            edp = (x * ex + y * ey + z * ez) / sqrt((x * x + y * y + z * z) * nes)

            if edp <= -1.0:
                f[i] = pi
            elif edp >= 1.0:
                f[i] = 0.0
            elif (x * vx + y * vy + z * vz) > 0.0:
                f[i] = arccos(edp)
            else:
                f[i] = 2.0 * pi - arccos(edp)
    return f


@njit(fastmath=True)
def cos_v_p_angle_o5v(v, times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the angle between the planet position and a fixed reference vector `v`."""
    n = times.size
    out = zeros(n)
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    for i in range(n):
        x, y, z = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        out[i] = (x * v[0] + y * v[1] + z * v[2]) * inv_nv / sqrt(x * x + y * y + z * z)
    return out


@njit(fastmath=True)
def cos_alpha_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at scalar time `t`."""
    x, y, z = xyz_o5s(t, t0, p, dt, pktable, points, coeffs)
    return -z / sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def cos_alpha_o5v(times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at an array of times."""
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        out[i] = -z / sqrt(x * x + y * y + z * z)
    return out


@njit(fastmath=True)
def star_planet_distance_o5v(times, t0, p, dt, pktable, points, coeffs):
    """3D star-planet distance at an array of times."""
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        out[i] = sqrt(x * x + y * y + z * z)
    return out


# ---------------------------------------------------------------------------
# Photometric/RV signals
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _lambert_kernel(cos_alpha):
    """Lambertian phase function evaluated at a cosine of the phase angle.

    Equivalent to ``(sin(arccos(c)) + (pi - arccos(c)) * c) / pi`` but uses
    ``sqrt(1 - c**2)`` instead of ``sin(arccos(c))`` to skip one trig call,
    and clamps ``c`` to ``[-1, 1]`` so a Taylor-rounding overshoot can't
    produce a NaN from ``arccos``.
    """
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
    alpha = arccos(cos_alpha)
    return (sin_alpha + (pi - alpha) * cos_alpha) / pi, alpha


@njit(fastmath=True)
def lambert_phase_curve_o5s(time, ag, a, k, t0, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution at a scalar time."""
    amplitude = k * k * ag / (a * a)
    cos_alpha = cos_alpha_o5s(time, t0, p, dt, pktable, points, coeffs)
    phase, _ = _lambert_kernel(cos_alpha)
    return amplitude * phase


@njit(fastmath=True)
def lambert_phase_curve_o5v(times, ag, a, k, t0, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution at an array of times."""
    n = times.size
    res = zeros(n)
    amplitude = k * k * ag / (a * a)
    for i in range(n):
        cos_alpha = cos_alpha_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        phase, _ = _lambert_kernel(cos_alpha)
        res[i] = amplitude * phase
    return res


@njit(fastmath=True)
def lambert_and_emission_o5v(times, ag, fr_night, fr_day, emi_offset, a, k,
                             t0, p, dt, pktable, points, coeffs):
    """Lambertian reflection plus a simple cosine-emission day/night model."""
    n = times.size
    ref, emi = zeros(n), zeros(n)
    k2 = k * k
    aref = k2 * ag / (a * a)
    for i in range(n):
        cos_alpha = cos_alpha_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        phase, alpha = _lambert_kernel(cos_alpha)
        ref[i] = aref * phase
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


@njit(fastmath=True)
def ev_signal_o5v(alpha, mass_ratio, inc, times, t0, p, dt, pktable, points, coeffs):
    """Ellipsoidal variation signal (Lillo-Box et al. 2014, Eqs. 6–10).

    Uses the identity ``cos(2*arccos(u)) = 2*u**2 - 1`` to avoid a redundant
    arccos/cos pair.
    """
    n = times.size
    out = zeros(n)
    sin2_inc = sin(inc) ** 2
    pre = -alpha * mass_ratio * sin2_inc
    for i in range(n):
        x, y, z = xyz_o5s(times[i], t0, p, dt, pktable, points, coeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        out[i] = pre * (2.0 * cz * cz - 1.0) / (d2 * d)
    return out


@njit(fastmath=True)
def rv_o5v(times, k, t0, p, a, i, e, dt, pktable, points, coeffs):
    """Radial velocity at an array of times (Perryman 2018, Eq. 2.23)."""
    n = times.size
    rvs = zeros(n)
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    for j in range(n):
        rvs[j] = vz_o5s(times[j], t0, p, dt, pktable, points, coeffs) * scale
    return rvs


# ---------------------------------------------------------------------------
# Light travel time
# ---------------------------------------------------------------------------

# Time taken by light to traverse one solar radius, in days:
# (1 R_sun) / c = ((1 * u.R_sun).to(u.m) / c.c).to('d').value
LTT_DAYS_PER_RSUN = 2.685885891543453e-05


@njit(fastmath=True)
def light_travel_time_o5s(t, t0, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction at a scalar time, referenced to primary transit.

    The correction is

        ltt(t) = -(z(t) - z(t_transit)) · rstar · (R_sun / c)

    with z in stellar radii, rstar in solar radii, result in days. The
    reference is the primary transit (inferior conjunction): ``ltt(t_transit)
    = 0`` by construction. This matches the convention used in transit
    fitting, where the observed mid-transit time is the reference and the LTT
    correction should add to the timing offset between primary transit and
    secondary eclipse (and intermediate phases), not to the transit itself.

    Important: the convention in this module is that ``t0`` is the
    **periastron** time (the same as for every other ``*_o5*`` evaluator in
    ``orbit3d.py``). The transit time is ``t0 + to`` where
    ``to = mean_anomaly_at_transit(e, w) · p / (2π)``. The ``e, w`` arguments
    are needed to determine ``to``.

    Parameters
    ----------
    t : float
        Time at which to evaluate the correction.
    t0 : float
        Time of periastron passage.
    p : float
        Orbital period [days].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    rstar : float
        Stellar radius [R_sun].
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from ``solve3d_orbit`` / ``create_knots``.

    Returns
    -------
    ltt : float
        Light travel time correction [days].
    """
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_t = z_o5s(t, t0, p, dt, pktable, points, coeffs)
    z_tr = z_o5s(t0 + to, t0, p, dt, pktable, points, coeffs)
    return -(z_t - z_tr) * rstar * LTT_DAYS_PER_RSUN


@njit(fastmath=True)
def light_travel_time_o5v(times, t0, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction at an array of times, referenced to primary transit.

    See :func:`light_travel_time_o5s` for the sign and reference convention.
    """
    n = times.size
    ltt = zeros(n)
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_tr = z_o5s(t0 + to, t0, p, dt, pktable, points, coeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    for j in range(n):
        ltt[j] = factor * (z_o5s(times[j], t0, p, dt, pktable, points, coeffs) - z_tr)
    return ltt
