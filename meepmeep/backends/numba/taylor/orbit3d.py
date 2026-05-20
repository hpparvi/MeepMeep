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

"""Multi-knot Taylor-series evaluators over a full orbit.

The functions in this module evaluate orbit-spanning quantities (position,
velocity, projected distance, phase angle, radial velocity, etc.) at
arbitrary times by looking up the appropriate knot via ``pktable`` and then
delegating to the single-knot evaluators in ``position3d``/``velocity3d``.

Coefficient layout: ``coeffs`` is an ``(npt, 3, 5)`` array as produced by
``solve3d_orbit`` — ``coeffs[ix]`` is the ``(3, 5)`` matrix consumed by
``pos_c``, ``vel_c``, ``zvel_c``, ``sep_c``, and ``pz_c``.
"""

from numba import njit
from numpy import zeros, pi, floor, sqrt, sin, cos, arccos

from .position3d import pos_c, sep_c, pz_c
from .velocity3d import vel_c, zvel_c
from .solve3d import solve3d
from ..utils import mean_anomaly, mean_anomaly_at_transit


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

@njit
def solve3d_orbit(knot_times, p, a, i, e, w, npt):
    """Pre-compute Taylor coefficients at every knot of one orbit.

    For each interior knot this calls :func:`~meepmeep.backends.numba.taylor.solve3d.solve3d`
    once and stacks the resulting ``(3, 5)`` matrices into a single
    ``(npt, 3, 5)`` array. The last slot is the periodic image of the
    first and is copied rather than recomputed.

    Parameters
    ----------
    knot_times : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]``, with ``knot_times[-1]``
        equal to ``knot_times[0] + 1``. Built by
        :func:`~meepmeep.backends.numba.knots.create_knots`.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity, :math:`0 \\le e < 1`.
    w : float
        Argument of periastron [radians].
    npt : int
        Number of knots, including the periodic-image slot.

    Returns
    -------
    coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every knot. Each ``coeffs[ix]`` is a
        ``(3, 5)`` matrix in the layout produced by ``solve3d`` (rows: x, y,
        z; columns: position, velocity, acceleration, jerk, snap; pre-scaled
        by factorial of the Taylor order).

    Notes
    -----
    If you hand-roll ``knot_times`` you must enforce the periodic-image
    contract yourself; ``knots.create_knots`` does this automatically.
    """
    coeffs = zeros((npt, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        coeffs[ix, :, :] = solve3d(p * knot_times[ix] - to, p, a, i, e, w)
    coeffs[-1, :, :] = coeffs[0]
    return coeffs


@njit(fastmath=True, inline="always")
def knot_ix(t, t0, p, dt, pktable) -> int:
    """Return the knot index for a single time.

    Epoch-folds ``t`` into one period and dispatches it to the appropriate
    knot via ``pktable``.

    Parameters
    ----------
    t : float
        Time at which to look up the knot.
    t0 : float
        Reference time anchoring the knot grid (periastron time in the
        orbit3d convention).
    p : float
        Orbital period [days].
    dt : float
        Width of one ``pktable`` bucket in fraction of the period.
    pktable : ndarray of int
        Time-to-knot lookup table built by
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    ix : int
        Index into ``coeffs`` / ``points`` for the relevant knot.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    return pktable[int(floor(tc / (dt * p)))]


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@njit(fastmath=True, inline="always")
def pos_os(t, t0, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position at scalar time ``t`` for any orbital phase.

    Parameters
    ----------
    t : float
        Time at which to evaluate the position.
    t0 : float
        Periastron time anchoring the knot grid.
    p : float
        Orbital period [days].
    dt : float
        ``pktable`` bucket width in fraction of the period.
    pktable : ndarray of int
        Time-to-knot lookup table.
    points : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]``.
    coeffs : ndarray, shape (npt, 3, 5)
        Per-knot Taylor coefficient matrices from :func:`solve3d_orbit`.

    Returns
    -------
    x, y, z : float
        Planet position in units of the stellar radius. ``x``, ``y`` are
        the sky-plane coordinates; ``z`` is the line-of-sight depth
        (positive toward the observer).
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pos_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def pos_ov(times, t0, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the position.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    xs, ys, zs : ndarray, shape (N,)
        Planet position arrays in units of the stellar radius.
    """
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = pos_os(times[i], t0, p, dt, pktable, points, coeffs)
    return xs, ys, zs


@njit(fastmath=True, inline="always")
def zpos_os(t, t0, p, dt, pktable, points, coeffs):
    """Planet z-position at scalar time ``t`` for any orbital phase.

    Cheaper than :func:`pos_os` when only the line-of-sight coordinate is
    needed (e.g. for light travel time and eclipse-side geometry).

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-coordinate.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    z : float
        Line-of-sight planet coordinate [stellar radii], positive toward
        the observer.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pz_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def zpos_ov(times, t0, p, dt, pktable, points, coeffs):
    """Planet z-position at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-coordinate.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    zs : ndarray, shape (N,)
        Line-of-sight planet coordinates [stellar radii].
    """
    npt = times.size
    zs = zeros(npt)
    for i in range(npt):
        zs[i] = zpos_os(times[i], t0, p, dt, pktable, points, coeffs)
    return zs


@njit(fastmath=True, inline="always")
def sep_os(t, t0, p, dt, pktable, points, coeffs):
    """Sky-projected planet-star separation at scalar time ``t``.

    Returns :math:`\\sqrt{x^2 + y^2}` in units of the stellar radius —
    the quantity transit light-curve models consume directly.

    Parameters
    ----------
    t : float
        Time at which to evaluate the separation.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    sep : float
        Sky-projected separation [stellar radii], always non-negative.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return sep_c(tc - points[ix] * p, coeffs[ix])


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

@njit(fastmath=True, inline="always")
def vel_os(t, t0, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity at scalar time ``t`` for any orbital phase.

    Parameters
    ----------
    t : float
        Time at which to evaluate the velocity.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    vx, vy, vz : float
        Velocity components in :math:`R_\\star/\\mathrm{day}`. ``vx``,
        ``vy`` are the sky-plane components; ``vz`` is the line-of-sight
        component (positive toward the observer).
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return vel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def vel_ov(times, t0, p, dt, pktable, points, coeffs):
    """Planet (vx, vy, vz) velocity at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the velocity.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    vxs, vys, vzs : ndarray, shape (N,)
        Velocity component arrays in :math:`R_\\star/\\mathrm{day}`.
    """
    npt = times.size
    vxs, vys, vzs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        vxs[i], vys[i], vzs[i] = vel_os(times[i], t0, p, dt, pktable, points, coeffs)
    return vxs, vys, vzs


@njit(fastmath=True, inline="always")
def zvel_os(t, t0, p, dt, pktable, points, coeffs):
    """Planet z-velocity at scalar time ``t`` for any orbital phase.

    Cheaper than :func:`vel_os` when only the line-of-sight component is
    needed (e.g. for radial velocity).

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-velocity.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    vz : float
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`].
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def zvel_ov(times, t0, p, dt, pktable, points, coeffs):
    """Planet z-velocity at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-velocity.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    vzs : ndarray, shape (N,)
        Line-of-sight velocities [:math:`R_\\star/\\mathrm{day}`].
    """
    npt = times.size
    vzs = zeros(npt)
    for i in range(npt):
        vzs[i] = zvel_os(times[i], t0, p, dt, pktable, points, coeffs)
    return vzs


# ---------------------------------------------------------------------------
# Anomalies and angles
# ---------------------------------------------------------------------------

@njit
def true_anomaly_ov(times, t0, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """True anomaly at an array of times.

    Computed from the angle between the planet position vector and the
    eccentricity vector :math:`(e_x, e_y, e_z)`. The sign of
    :math:`\\mathbf{r}\\cdot\\mathbf{v}` disambiguates the two branches of
    :math:`\\arccos`.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the true anomaly.
    t0 : float
        Periastron time.
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector pointing from the focus
        toward periastron. ``(-1, 0, 0)`` is the sentinel that
        :func:`~meepmeep.backends.numba.utils.eccentricity_vector`
        returns for near-circular orbits and triggers the fast path.
    w : float
        Argument of periastron [radians]. Used only on the
        circular-orbit fast path, where the true anomaly is
        approximated by the mean anomaly.
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    f : ndarray, shape (N,)
        True anomaly at each input time [radians], in :math:`[0, 2\\pi)`.

    Notes
    -----
    The circular-orbit fast path skips the geometric chain to avoid
    division by a near-zero :math:`|\\mathbf{e}|`. ``utils.eccentricity_vector``
    emits the ``(-1, 0, 0)`` sentinel when ``e < 1e-5``, and the test
    here matches that sentinel.
    """
    npt = times.size
    f = zeros(npt)
    nes = ex * ex + ey * ey + ez * ez

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
            x, y, z = pos_c(tcc, c)
            vx, vy, vz = vel_c(tcc, c)
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
def cos_v_p_angle_ov(v, times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the angle between the planet position and a fixed reference vector.

    Useful for projecting the planet position onto an arbitrary
    line-of-sight axis (e.g. the spin axis of an oblate star).

    Parameters
    ----------
    v : ndarray, shape (3,)
        Reference vector. Need not be unit-norm; the cosine is computed
        from the dot product divided by the product of the norms.
    times : ndarray, shape (N,)
        Times at which to evaluate the angle.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    cos_theta : ndarray, shape (N,)
        Cosine of the angle between the planet position vector and
        ``v``, in :math:`[-1, 1]`.
    """
    n = times.size
    out = zeros(n)
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    for i in range(n):
        x, y, z = pos_os(times[i], t0, p, dt, pktable, points, coeffs)
        out[i] = (x * v[0] + y * v[1] + z * v[2]) * inv_nv / sqrt(x * x + y * y + z * z)
    return out


@njit(fastmath=True, inline="always")
def cos_alpha_os(t, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at scalar time ``t``.

    The phase angle :math:`\\alpha` is the star-planet-observer angle.
    With z positive toward the observer, :math:`\\cos\\alpha = -z/r` where
    :math:`r = \\sqrt{x^2 + y^2 + z^2}`. At superior conjunction (full
    phase, planet behind star) :math:`\\cos\\alpha = +1`; at inferior
    conjunction (new phase, planet in front) :math:`\\cos\\alpha = -1`.

    Parameters
    ----------
    t : float
        Time at which to evaluate the phase angle.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    cos_alpha : float
        Cosine of the phase angle, in :math:`[-1, 1]`.
    """
    x, y, z = pos_os(t, t0, p, dt, pktable, points, coeffs)
    return -z / sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def cos_alpha_ov(times, t0, p, dt, pktable, points, coeffs):
    """Cosine of the phase angle at an array of times.

    See :func:`cos_alpha_os` for the sign and reference conventions.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the phase angle.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    cos_alpha : ndarray, shape (N,)
        Cosine of the phase angle at each input time.
    """
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = pos_os(times[i], t0, p, dt, pktable, points, coeffs)
        out[i] = -z / sqrt(x * x + y * y + z * z)
    return out


@njit(fastmath=True)
def star_planet_distance_ov(times, t0, p, dt, pktable, points, coeffs):
    """3D star-planet distance at an array of times.

    Returns :math:`\\sqrt{x^2 + y^2 + z^2}`, the full Euclidean separation
    in 3D. Distinct from :func:`sep_os`, which projects out the
    line-of-sight component.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the separation.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    r : ndarray, shape (N,)
        3D star-planet separation [stellar radii].
    """
    n = times.size
    out = zeros(n)
    for i in range(n):
        x, y, z = pos_os(times[i], t0, p, dt, pktable, points, coeffs)
        out[i] = sqrt(x * x + y * y + z * z)
    return out


# ---------------------------------------------------------------------------
# Photometric/RV signals
# ---------------------------------------------------------------------------

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


@njit(fastmath=True, inline="always")
def lambert_phase_curve_os(time, ag, a, k, t0, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution at a scalar time.

    Evaluates :math:`F(t) = (k/a)^2\\, A_g\\, f(\\alpha(t))` where
    :math:`f` is the Lambert kernel and :math:`\\alpha(t)` is the
    instantaneous phase angle. The result is the planet-to-star flux
    ratio of reflected light at full phase scaled by the phase function.

    Parameters
    ----------
    time : float
        Time at which to evaluate the flux contribution.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    flux : float
        Reflected planet-to-star flux ratio at the given time.
    """
    amplitude = k * k * ag / (a * a)
    cos_alpha = cos_alpha_os(time, t0, p, dt, pktable, points, coeffs)
    phase, _ = _lambert_kernel(cos_alpha)
    return amplitude * phase


@njit(fastmath=True)
def lambert_phase_curve_ov(times, ag, a, k, t0, p, dt, pktable, points, coeffs):
    """Lambertian phase-curve flux contribution at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the flux contribution.
    ag, a, k, t0, p, dt, pktable, points, coeffs :
        See :func:`lambert_phase_curve_os`.

    Returns
    -------
    flux : ndarray, shape (N,)
        Reflected planet-to-star flux ratio at each input time.
    """
    n = times.size
    res = zeros(n)
    amplitude = k * k * ag / (a * a)
    for i in range(n):
        cos_alpha = cos_alpha_os(times[i], t0, p, dt, pktable, points, coeffs)
        phase, _ = _lambert_kernel(cos_alpha)
        res[i] = amplitude * phase
    return res


@njit(fastmath=True)
def lambert_and_emission_ov(times, ag, fr_night, fr_day, emi_offset, a, k,
                            t0, p, dt, pktable, points, coeffs):
    """Lambertian reflection plus a simple cosine-emission day/night model.

    Returns the reflected and thermal-emission flux ratios separately so
    callers can combine them with their own bolometric weighting. The
    emission model is a smoothly varying interpolation between night-side
    and day-side levels driven by :math:`\\cos(\\alpha + \\delta)`, where
    :math:`\\delta` is an optional offset that captures advection.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the flux contributions.
    ag : float
        Geometric albedo (reflected component).
    fr_night : float
        Night-side flux ratio (planet/star).
    fr_day : float
        Day-side flux ratio (planet/star).
    emi_offset : float
        Phase-angle offset of the emission peak [radians]. ``0`` puts
        peak emission at superior conjunction; non-zero values shift it
        to model day-to-night advection.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    ref : ndarray, shape (N,)
        Reflected-light flux ratio at each input time.
    emi : ndarray, shape (N,)
        Thermal-emission flux ratio at each input time.
    """
    n = times.size
    ref, emi = zeros(n), zeros(n)
    k2 = k * k
    aref = k2 * ag / (a * a)
    for i in range(n):
        cos_alpha = cos_alpha_os(times[i], t0, p, dt, pktable, points, coeffs)
        phase, alpha = _lambert_kernel(cos_alpha)
        ref[i] = aref * phase
        emi[i] = k2 * (fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cos(alpha + emi_offset)))
    return ref, emi


@njit(fastmath=True)
def ev_signal_ov(alpha, mass_ratio, inc, times, t0, p, dt, pktable, points, coeffs):
    """Ellipsoidal variation signal (Lillo-Box et al. 2014, Eqs. 6–10).

    Returns the relative flux variation induced by the tidally distorted
    primary as a function of the orbital phase. The amplitude scales
    with the mass ratio, the projected-area factor :math:`\\sin^2 i`,
    and the inverse cube of the instantaneous 3D separation.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians].
    times : ndarray, shape (N,)
        Times at which to evaluate the signal.
    t0, p, dt, pktable, points, coeffs :
        See :func:`pos_os`.

    Returns
    -------
    ev : ndarray, shape (N,)
        Relative flux variation due to ellipsoidal distortion at each
        input time.

    Notes
    -----
    Uses the identity :math:`\\cos(2\\arccos u) = 2u^2 - 1` to skip a
    redundant arccos/cos pair.
    """
    n = times.size
    out = zeros(n)
    sin2_inc = sin(inc) ** 2
    pre = -alpha * mass_ratio * sin2_inc
    for i in range(n):
        x, y, z = pos_os(times[i], t0, p, dt, pktable, points, coeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        out[i] = pre * (2.0 * cz * cz - 1.0) / (d2 * d)
    return out


@njit(fastmath=True)
def rv_ov(times, k, t0, p, a, i, e, dt, pktable, points, coeffs):
    """Radial velocity at an array of times (Perryman 2018, Eq. 2.23).

    Converts the internal line-of-sight velocity (in
    :math:`R_\\star/\\mathrm{day}`) to an observed radial velocity by
    multiplying with the closed-form scale factor
    :math:`K / [(2\\pi/p)(a\\sin i)/\\sqrt{1-e^2}]`.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the radial velocity.
    k : float
        Radial-velocity semi-amplitude [m s\\ :sup:`-1`].
    t0 : float
        Periastron time.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    rvs : ndarray, shape (N,)
        Radial velocity at each input time [m s\\ :sup:`-1`].
    """
    n = times.size
    rvs = zeros(n)
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    for j in range(n):
        rvs[j] = zvel_os(times[j], t0, p, dt, pktable, points, coeffs) * scale
    return rvs


# ---------------------------------------------------------------------------
# Light travel time
# ---------------------------------------------------------------------------

# Time taken by light to traverse one solar radius, in days:
# (1 R_sun) / c = ((1 * u.R_sun).to(u.m) / c.c).to('d').value
LTT_DAYS_PER_RSUN = 2.685885891543453e-05


@njit(fastmath=True)
def light_travel_time_os(t, t0, p, e, w, rstar, dt, pktable, points, coeffs):
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
    **periastron** time (the same as for every other ``*_o*`` evaluator in
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
    z_t = zpos_os(t, t0, p, dt, pktable, points, coeffs)
    z_tr = zpos_os(t0 + to, t0, p, dt, pktable, points, coeffs)
    return -(z_t - z_tr) * rstar * LTT_DAYS_PER_RSUN


@njit(fastmath=True)
def light_travel_time_ov(times, t0, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction at an array of times, referenced to primary transit.

    Vectorised version of :func:`light_travel_time_os`. Caches the
    transit-time z-coordinate ``z_tr`` once outside the loop and reuses
    it for every input time.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the correction.
    t0 : float
        Periastron time.
    p : float
        Orbital period [days].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    rstar : float
        Stellar radius [R_sun].
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    ltt : ndarray, shape (N,)
        Light travel time correction at each input time [days].
    """
    n = times.size
    ltt = zeros(n)
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_tr = zpos_os(t0 + to, t0, p, dt, pktable, points, coeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    for j in range(n):
        ltt[j] = factor * (zpos_os(times[j], t0, p, dt, pktable, points, coeffs) - z_tr)
    return ltt
