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

"""Multi-knot Taylor-series evaluators with parameter derivatives.

Derivative-returning counterparts of the routines in ``orbit3d.py``. Every
function returns both the value and its partial derivatives with respect to
the orbital parameters ``(phase, p, a, i, e, w)`` and any extra physical
inputs the routine takes (appended to the orbital block in argument order).

Coefficient layout:
- ``coeffs`` : ``(npt, 3, 5)`` — Taylor coefficients, as in ``orbit3d``.
- ``dcoeffs`` : ``(npt, 6, 3, 5)`` — derivatives of the Taylor coefficients
  w.r.t. the 6 orbital parameters, produced by ``solve3d_orbit_d``.

Vector evaluators (``*_o5v_d``) return per-coordinate derivative arrays of
shape ``(N, ndp)`` where ``ndp`` is ``6`` for orbital-only routines and
``6 + n_extra`` for routines with extra physical inputs.
"""

from numba import njit
from numpy import zeros, pi, floor, sqrt, sin, cos, arccos

from .position3d import pos_c
from .velocity3d import vel_c, zvel_c
from .position3dd import pos_cd, sep_cd, pz_cd
from .velocity3dd import v3dc_d, vzc_d, rvc_d
from .solve3dd import solve3d_d
from ..utils import mean_anomaly_at_transit, mean_anomaly_at_transit_with_derivatives


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

@njit
def solve3d_orbit_d(knot_times, p, a, i, e, w, npt):
    """Pre-compute Taylor coefficients and parameter-derivative coefficients
    at every knot of one orbit.

    Parameters
    ----------
    knot_times : ndarray
        Knot times in normalised phase ``[0, 1]``; ``knot_times[-1]`` must be
        the periodic image of ``knot_times[0]``.
    p, a, i, e, w : float
        Orbital parameters.
    npt : int
        Number of knots.

    Returns
    -------
    coeffs : ndarray (npt, 3, 5)
        Position Taylor coefficients per knot.
    dcoeffs : ndarray (npt, 6, 3, 5)
        Parameter-derivative coefficients per knot, ordered
        ``(phase, p, a, i, e, w)`` along the second axis.
    """
    coeffs = zeros((npt, 3, 5))
    dcoeffs = zeros((npt, 6, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        cf, dcf = solve3d_d(p * knot_times[ix] - to, p, a, i, e, w)
        coeffs[ix, :, :] = cf
        dcoeffs[ix, :, :, :] = dcf
    coeffs[-1, :, :] = coeffs[0]
    dcoeffs[-1, :, :, :] = dcoeffs[0]
    return coeffs, dcoeffs


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def xyz_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (x, y, z) position and orbital-parameter derivatives at scalar time.

    Returns
    -------
    px, py, pz : float
        Sky-frame position components.
    dpx, dpy, dpz : ndarray (6,)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pos_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def xyz_o5v_d(times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (x, y, z) position and orbital-parameter derivatives at array of times.

    Returns
    -------
    xs, ys, zs : ndarray (N,)
        Position components per time.
    dxs, dys, dzs : ndarray (N, 6)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    xs = zeros(n)
    ys = zeros(n)
    zs = zeros(n)
    dxs = zeros((n, 6))
    dys = zeros((n, 6))
    dzs = zeros((n, 6))
    for j in range(n):
        x, y, z, dx, dy, dz = xyz_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        xs[j] = x
        ys[j] = y
        zs[j] = z
        for k in range(6):
            dxs[j, k] = dx[k]
            dys[j, k] = dy[k]
            dzs[j, k] = dz[k]
    return xs, ys, zs, dxs, dys, dzs


@njit(fastmath=True)
def z_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives at scalar time.

    Returns
    -------
    pz : float
        z position.
    dpz : ndarray (6,)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pz_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def z_o5v_d(times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives at array of times."""
    n = times.size
    zs = zeros(n)
    dzs = zeros((n, 6))
    for j in range(n):
        z, dz = z_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        zs[j] = z
        for k in range(6):
            dzs[j, k] = dz[k]
    return zs, dzs


@njit(fastmath=True)
def pd_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Projected planet-star distance and orbital-parameter derivatives at scalar time.

    Returns
    -------
    d : float
        Projected distance ``sqrt(x**2 + y**2)``.
    dd : ndarray (6,)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return sep_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def vxyz_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives at scalar time."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return v3dc_d(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def vxyz_o5v_d(times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives at array of times."""
    n = times.size
    vxs = zeros(n)
    vys = zeros(n)
    vzs = zeros(n)
    dvxs = zeros((n, 6))
    dvys = zeros((n, 6))
    dvzs = zeros((n, 6))
    for j in range(n):
        vx, vy, vz, dvx, dvy, dvz = vxyz_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        vxs[j] = vx
        vys[j] = vy
        vzs[j] = vz
        for k in range(6):
            dvxs[j, k] = dvx[k]
            dvys[j, k] = dvy[k]
            dvzs[j, k] = dvz[k]
    return vxs, vys, vzs, dvxs, dvys, dvzs


@njit(fastmath=True)
def vz_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives at scalar time."""
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return vzc_d(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def vz_o5v_d(times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives at array of times."""
    n = times.size
    vzs = zeros(n)
    dvzs = zeros((n, 6))
    for j in range(n):
        vz, dvz = vz_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        vzs[j] = vz
        for k in range(6):
            dvzs[j, k] = dvz[k]
    return vzs, dvzs


# ---------------------------------------------------------------------------
# Anomalies and angles
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def cos_alpha_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the phase angle and orbital-parameter derivatives at scalar time.

    ``cos_alpha = -z / r`` with ``r = sqrt(x**2 + y**2 + z**2)``.

    Returns
    -------
    ca : float
        Cosine of the phase angle.
    dca : ndarray (6,)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    x, y, z, dx, dy, dz = xyz_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs)
    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    ca = -z / r
    dca = zeros(6)
    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    for k in range(6):
        # d(-z/r)/dθ = -dz/r + z·(x·dx + y·dy + z·dz)/r^3
        dca[k] = -dz[k] * inv_r + z * (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r3
    return ca, dca


@njit(fastmath=True)
def cos_alpha_o5v_d(times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the phase angle and orbital-parameter derivatives at array of times."""
    n = times.size
    cas = zeros(n)
    dcas = zeros((n, 6))
    for j in range(n):
        ca, dca = cos_alpha_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        cas[j] = ca
        for k in range(6):
            dcas[j, k] = dca[k]
    return cas, dcas


@njit(fastmath=True)
def star_planet_distance_o5v_d(times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """3D star-planet distance and orbital-parameter derivatives at array of times.

    Returns
    -------
    rs : ndarray (N,)
        Distances per time.
    drs : ndarray (N, 6)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    n = times.size
    rs = zeros(n)
    drs = zeros((n, 6))
    for j in range(n):
        x, y, z, dx, dy, dz = xyz_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        r = sqrt(x * x + y * y + z * z)
        rs[j] = r
        inv_r = 1.0 / r
        for k in range(6):
            drs[j, k] = (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r
    return rs, drs


@njit(fastmath=True)
def cos_v_p_angle_o5v_d(v, times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the angle between planet position and a fixed reference vector ``v``.

    The reference vector ``v`` is treated as a constant; derivatives are w.r.t.
    the 6 orbital parameters only.

    Returns
    -------
    cs : ndarray (N,)
        Cosine values per time.
    dcs : ndarray (N, 6)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    n = times.size
    cs = zeros(n)
    dcs = zeros((n, 6))
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    for j in range(n):
        x, y, z, dx, dy, dz = xyz_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        inv_r = 1.0 / r
        inv_r3 = inv_r / r2
        dot = x * v[0] + y * v[1] + z * v[2]
        cs[j] = dot * inv_nv * inv_r
        # d/dθ[(x·v)/(|x|·|v|)] = ((dx·v)/|x| - (x·v)·(x·dx)/|x|^3) / |v|
        for k in range(6):
            ddot = dx[k] * v[0] + dy[k] * v[1] + dz[k] * v[2]
            xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
            dcs[j, k] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3)
    return cs, dcs


# ---------------------------------------------------------------------------
# True anomaly
# ---------------------------------------------------------------------------
#
# The geometric definition uses the angle between the planet position vector
# and the eccentricity vector. Differentiating that (with the prograde sign
# correction from r·v) gives a well-defined gradient everywhere except at
# the two singular configurations ``edp = ±1`` (planet on the apsidal line).
# At those points the analytic derivative diverges; we set it to zero so
# downstream gradient-based fits don't get a NaN. The circular fast path
# (``ex ≤ -0.9999`` sentinel from ``eccentricity_vector``) collapses true
# anomaly to mean anomaly: ``f = 2π(t - t0)/p`` ⇒ analytic derivatives are
# trivial in (phase, p) and zero in the rest.

@njit
def true_anomaly_o5v_d(times, t0, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """True anomaly and its orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray (N,)
        Times at which to evaluate the true anomaly.
    t0 : float
        Time of inferior conjunction.
    p : float
        Orbital period.
    ex, ey, ez : float
        Components of the eccentricity vector.
    w : float
        Argument of periastron (kept for signature parity with the base function;
        currently unused inside this routine because the eccentricity vector is
        passed explicitly).
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    f : ndarray (N,)
        True anomaly per time.
    df : ndarray (N, 6)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``. The ``ex, ey, ez, w``
        inputs are treated as known constants — they are functions of the
        orbital parameters but the dependency is captured implicitly through
        the geometric chain rule on the position vector.

    Notes
    -----
    At the singular configurations ``edp = ±1`` (``edp`` = cosine of angle
    between position and eccentricity vector) the analytic gradient diverges
    and is replaced by zero. The circular-orbit fast path uses the
    mean-anomaly identity ``f = 2π(t - t0) / p``.
    """
    n = times.size
    f = zeros(n)
    df = zeros((n, 6))
    nes = ex * ex + ey * ey + ez * ez

    # Circular-orbit fast path: f = 2π·(t - t0) / p.
    # df/d(phase) = -2π/p (since phase parameter shifts t0 by 1 unit of phase
    # which is equivalent to a +1-day shift here — solve3d_d's "phase" is in
    # days, so dphase = +1 ⇒ dt0 = +1 ⇒ df = -2π/p).
    # df/dp = -2π·(t - t0) / p^2.
    if ex <= -0.9999 and nes > 0.99:
        twopi = 2.0 * pi
        for j in range(n):
            tau = times[j] - t0
            # Reduce to one period for the value (mean_anomaly does this in base).
            epoch = floor(tau / p)
            tau_red = tau - epoch * p
            f[j] = twopi * tau_red / p
            df[j, 0] = -twopi / p
            df[j, 1] = -twopi * tau_red / (p * p)
        return f, df

    for j in range(n):
        t = times[j]
        epoch = floor((t - t0) / p)
        tc = t - t0 - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p
        c = coeffs[ix]
        dc = dcoeffs[ix]

        x, y, z, dx, dy, dz = pos_cd(tcc, c, dc)
        vx, vy, vz, dvx, dvy, dvz = v3dc_d(tcc, c, dc)

        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        sqrt_r2_nes = sqrt(r2 * nes)
        edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes
        rdotv = x * vx + y * vy + z * vz

        if edp <= -1.0:
            f[j] = pi
            # Singular: leave df[j] = 0.
        elif edp >= 1.0:
            f[j] = 0.0
            # Singular: leave df[j] = 0.
        else:
            sign = 1.0 if rdotv > 0.0 else -1.0
            base = arccos(edp)
            f[j] = base if sign > 0.0 else 2.0 * pi - base
            # d(arccos(edp))/dθ = -dedp/sqrt(1 - edp^2)
            denom = sqrt(1.0 - edp * edp)
            inv_r2 = 1.0 / r2
            for k in range(6):
                # edp = (x·e)/(r·|e|). Treat |e| (and ex,ey,ez) as constants
                # for this routine — they're inputs. d(edp)/dθ_k
                # = (dx·e)/(r·|e|) - (x·e)·(x·dx)/(r^3·|e|)
                xdote = x * ex + y * ey + z * ez
                dxdote = dx[k] * ex + dy[k] * ey + dz[k] * ez
                xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
                dedp = dxdote / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
                # Equivalent: dedp = (dxdote * r2 - xdote * xdotdx * inv_r2 * r2) ... keep clarity.
                df_k = -dedp / denom
                df[j, k] = df_k if sign > 0.0 else -df_k
    return f, df


# ---------------------------------------------------------------------------
# Photometric / RV signals
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _lambert_kernel_d(cos_alpha):
    """Lambertian phase function, alpha, and ``dphase/dcos_alpha``.

    The analytic derivative of ``phase(c) = (sqrt(1-c**2) + (pi - arccos c)·c) / pi``
    simplifies to ``(pi - arccos c) / pi`` because the contributions from
    ``d/dc sqrt(1-c**2)`` and ``c · d/dc arccos c`` cancel exactly.
    """
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
    alpha = arccos(cos_alpha)
    phase = (sin_alpha + (pi - alpha) * cos_alpha) / pi
    dphase_dc = (pi - alpha) / pi
    return phase, alpha, dphase_dc


@njit(fastmath=True)
def lambert_phase_curve_o5s_d(time, ag, a, k, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux and parameter derivatives at scalar time.

    Derivative ordering: ``(phase, p, a, i, e, w, ag, k)`` — length 8.

    Returns
    -------
    flux : float
    dflux : ndarray (8,)
    """
    amplitude = k * k * ag / (a * a)
    ca, dca = cos_alpha_o5s_d(time, t0, p, dt, pktable, points, coeffs, dcoeffs)
    phase, _, dphase_dc = _lambert_kernel_d(ca)
    flux = amplitude * phase

    dflux = zeros(8)
    # Orbital block — chain through cos_alpha and through amplitude (only `a` matters).
    for kk in range(6):
        dflux[kk] = amplitude * dphase_dc * dca[kk]
    # Add d(amplitude)/da contribution to the `a` slot (index 2):
    # damplitude/da = -2 k^2 ag / a^3.
    dflux[2] += -2.0 * k * k * ag / (a * a * a) * phase
    # Extras: ag (index 6), k (index 7).
    dflux[6] = (k * k / (a * a)) * phase
    dflux[7] = (2.0 * k * ag / (a * a)) * phase
    return flux, dflux


@njit(fastmath=True)
def lambert_phase_curve_o5v_d(times, ag, a, k, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux and parameter derivatives at array of times.

    Returns
    -------
    flux : ndarray (N,)
    dflux : ndarray (N, 8)
        Derivatives w.r.t. ``(phase, p, a, i, e, w, ag, k)``.
    """
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 8))
    inv_a2 = 1.0 / (a * a)
    amplitude = k * k * ag * inv_a2
    da_amp = -2.0 * k * k * ag / (a * a * a)
    dag_amp = k * k * inv_a2
    dk_amp = 2.0 * k * ag * inv_a2
    for j in range(n):
        ca, dca = cos_alpha_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        phase, _, dphase_dc = _lambert_kernel_d(ca)
        flux[j] = amplitude * phase
        for kk in range(6):
            dflux[j, kk] = amplitude * dphase_dc * dca[kk]
        dflux[j, 2] += da_amp * phase
        dflux[j, 6] = dag_amp * phase
        dflux[j, 7] = dk_amp * phase
    return flux, dflux


@njit(fastmath=True)
def lambert_and_emission_o5v_d(times, ag, fr_night, fr_day, emi_offset, a, k,
                               t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus cosine-emission day/night model with parameter derivatives.

    Derivative ordering: ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``
    — length 11.

    Returns
    -------
    ref : ndarray (N,)
        Reflected (Lambertian) flux contribution.
    emi : ndarray (N,)
        Thermal emission contribution.
    dref : ndarray (N, 11)
    demi : ndarray (N, 11)
    """
    n = times.size
    ref = zeros(n)
    emi = zeros(n)
    dref = zeros((n, 11))
    demi = zeros((n, 11))
    k2 = k * k
    inv_a2 = 1.0 / (a * a)
    aref = k2 * ag * inv_a2
    daref_da = -2.0 * k2 * ag / (a * a * a)
    daref_dag = k2 * inv_a2
    daref_dk = 2.0 * k * ag * inv_a2

    for j in range(n):
        ca, dca = cos_alpha_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        phase, alpha, dphase_dc = _lambert_kernel_d(ca)

        # --- reflected component ---
        ref[j] = aref * phase
        for kk in range(6):
            dref[j, kk] = aref * dphase_dc * dca[kk]
        dref[j, 2] += daref_da * phase
        dref[j, 6] = daref_dag * phase
        # fr_night, fr_day, emi_offset (indices 7..9) are zero for ref.
        dref[j, 10] = daref_dk * phase

        # --- emission component ---
        # emi = k^2 · (fr_night + (fr_day - fr_night) · 0.5 · (1 - cos(alpha + emi_offset)))
        cs = cos(alpha + emi_offset)
        sn = sin(alpha + emi_offset)
        bracket = fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cs)
        emi[j] = k2 * bracket

        # d(alpha)/d(cos_alpha) = -1/sqrt(1 - ca^2). Avoid blow-up at |ca|=1
        # by clamping like _lambert_kernel_d does (interior tests safe).
        ca_clamped = ca
        if ca_clamped > 1.0:
            ca_clamped = 1.0
        elif ca_clamped < -1.0:
            ca_clamped = -1.0
        s = sqrt(1.0 - ca_clamped * ca_clamped)
        if s < 1e-12:
            dalpha_dc = 0.0
        else:
            dalpha_dc = -1.0 / s
        # demi/dorbital via cos_alpha → alpha → bracket
        # demi/dα = k^2 · (fr_day - fr_night) · 0.5 · sin(alpha + emi_offset)
        demi_dalpha = k2 * (fr_day - fr_night) * 0.5 * sn
        for kk in range(6):
            demi[j, kk] = demi_dalpha * dalpha_dc * dca[kk]
        # ag (6) does not enter emi; leave 0.
        # fr_night (7): k^2 · (1 - 0.5·(1-cs)) = k^2 · (0.5 + 0.5·cs)
        demi[j, 7] = k2 * (1.0 - 0.5 * (1.0 - cs))
        # fr_day (8):   k^2 · 0.5 · (1 - cs)
        demi[j, 8] = k2 * 0.5 * (1.0 - cs)
        # emi_offset (9): k^2 · (fr_day - fr_night) · 0.5 · sin(alpha + emi_offset)
        demi[j, 9] = k2 * (fr_day - fr_night) * 0.5 * sn
        # k (10): 2k · bracket
        demi[j, 10] = 2.0 * k * bracket

    return ref, emi, dref, demi


@njit(fastmath=True)
def ev_signal_o5v_d(alpha, mass_ratio, inc, times, t0, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal and parameter derivatives.

    Implements ``S = -alpha · mass_ratio · sin²(inc) · (2 cz² - 1) / d**3``
    where ``cz = z / d`` and ``d = sqrt(x**2 + y**2 + z**2)``. The function-
    local ``inc`` parameter is independent of the orbital inclination ``i`` —
    callers that share them should sum the two derivative slots.

    Derivative ordering: ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` —
    length 9.
    """
    n = times.size
    out = zeros(n)
    dout = zeros((n, 9))
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    for j in range(n):
        x, y, z, dx, dy, dz = xyz_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        # S = pre · g, where g = (2 cz^2 - 1) / d^3.
        # Rewrite g = (2 z^2 - d^2) / d^5 = (2 z^2 / d^5) - 1/d^3.
        g = (2.0 * cz * cz - 1.0) / (d2 * d)
        out[j] = pre * g

        # dg/dθ via dx, dy, dz. Use g = (2 z^2 - d^2) / d^5.
        # Let A = 2 z^2 - d^2,  d^5 = d2^2 · d.
        # dA = 4 z·dz - 2(x·dx + y·dy + z·dz)
        #    = -2(x·dx + y·dy) + 2 z·dz
        # d(d^5)/dθ = 5 d^3 · dd, with dd = (x·dx + y·dy + z·dz)/d.
        # dg = (dA · d^5 - A · 5 d^3 · dd) / d^10
        #    = (dA - 5 A · dd / d^2) / d^5.
        d5 = d2 * d2 * d
        A = 2.0 * z * z - d2
        for kk in range(6):
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dd = xdotdx / d
            dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
            dg = (dA - 5.0 * A * dd / d2) / d5
            dout[j, kk] = pre * dg
        # Extras (no orbital chain).
        # alpha (6): dS/dalpha = -mass_ratio · sin2_inc · g
        dout[j, 6] = -mass_ratio * sin2_inc * g
        # mass_ratio (7): dS/dmr = -alpha · sin2_inc · g
        dout[j, 7] = -alpha * sin2_inc * g
        # inc (8): d(sin^2 inc)/dinc = 2 sin_inc · cos_inc
        dout[j, 8] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g

    return out, dout


# ---------------------------------------------------------------------------
# Light travel time
# ---------------------------------------------------------------------------

# Time taken by light to traverse one solar radius, in days. Kept in sync
# with ``orbit3d.LTT_DAYS_PER_RSUN``.
LTT_DAYS_PER_RSUN = 2.685885891543453e-05


@njit(fastmath=True)
def _ltt_transit_z_and_d(t0, p, e, w, dt, pktable, points, coeffs, dcoeffs):
    """Helper: compute ``z(t_transit)`` and its full chain-rule derivative
    w.r.t. ``(phase, p, a, i, e, w)`` for use by the light-travel-time
    derivatives.

    The transit time depends on the orbital parameters via
    ``to = M_tr(e, w) · p / (2π)``, so the total derivative is

        d/dθ_k [z(t_transit(θ); θ)]
            = vz(t_transit) · (dto/dθ_k) + (∂z/∂θ_k)|_{t=t_transit}

    where:
      - dto/dp = M_tr(e, w) / (2π)
      - dto/de = (dM_tr/de) · p / (2π)
      - dto/dw = (dM_tr/dw) · p / (2π)
      - dto/dθ_k = 0 for k ∈ {phase, a, i}

    The "phase" slot inherits the multi-knot caveat documented at module
    level: it reflects a per-knot phase shift at the knot containing
    ``t_transit``, not a global user-facing T0 shift.
    """
    m_tr, dm_tr_de, dm_tr_dw = mean_anomaly_at_transit_with_derivatives(e, w)
    two_pi = 2.0 * pi
    to = m_tr / two_pi * p
    t_transit = t0 + to

    # Evaluate z and its (∂z/∂θ)|_{t=t_transit}.
    z_tr, dz_tr_partial = z_o5s_d(t_transit, t0, p, dt, pktable, points, coeffs, dcoeffs)
    # Velocity at transit (for the dt_transit/dθ chain term).
    vz_tr = vz_o5s(t_transit, t0, p, dt, pktable, points, coeffs)

    # dto/dθ: only slots 1 (p), 4 (e), 5 (w) are non-zero.
    dto = zeros(6)
    dto[1] = m_tr / two_pi
    dto[4] = dm_tr_de * p / two_pi
    dto[5] = dm_tr_dw * p / two_pi

    dz_tr_total = zeros(6)
    for k in range(6):
        dz_tr_total[k] = vz_tr * dto[k] + dz_tr_partial[k]
    return z_tr, dz_tr_total


@njit(fastmath=True)
def vz_o5s(t, t0, p, dt, pktable, points, coeffs):
    """Local z-velocity helper used by ``_ltt_transit_z_and_d``.

    Mirrors ``orbit3d.vz_o5s`` but kept private here to avoid a cross-module
    import cycle.
    """
    epoch = floor((t - t0) / p)
    tc = t - t0 - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def light_travel_time_o5s_d(t, t0, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Light travel time correction and orbital-parameter derivatives at scalar time.

    The correction is referenced to primary transit:

        ltt(t) = -(z(t) - z(t_transit)) · rstar · (R_sun / c)

    where ``t_transit = t0 + mean_anomaly_at_transit(e, w) · p / (2π)``.

    Per spec, the partial derivative w.r.t. ``rstar`` is *not* returned — only
    the 6 orbital derivatives in the canonical ``(phase, p, a, i, e, w)`` order.

    The reference ``z(t_transit)`` and its parameter derivatives are computed
    by ``_ltt_transit_z_and_d``, which includes the chain rule through
    ``t_transit(p, e, w)`` using ``vz(t_transit)``.

    Returns
    -------
    ltt : float
        Light travel time correction [days].
    dltt : ndarray (6,)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    z_t, dz_t = z_o5s_d(t, t0, p, dt, pktable, points, coeffs, dcoeffs)
    z_tr, dz_tr = _ltt_transit_z_and_d(t0, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    ltt = factor * (z_t - z_tr)
    dltt = zeros(6)
    for k in range(6):
        dltt[k] = factor * (dz_t[k] - dz_tr[k])
    return ltt, dltt


@njit(fastmath=True)
def light_travel_time_o5v_d(times, t0, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Light travel time correction and orbital-parameter derivatives at array of times.

    See :func:`light_travel_time_o5s_d` for the sign, reference, and parameter
    conventions.

    Returns
    -------
    ltt : ndarray (N,)
        Light travel time corrections [days].
    dltt : ndarray (N, 6)
        Derivatives w.r.t. ``(phase, p, a, i, e, w)``.
    """
    n = times.size
    ltt = zeros(n)
    dltt = zeros((n, 6))
    factor = -rstar * LTT_DAYS_PER_RSUN
    # Reference (z and its full derivative chain) computed once.
    z_tr, dz_tr = _ltt_transit_z_and_d(t0, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    for j in range(n):
        z, dz = z_o5s_d(times[j], t0, p, dt, pktable, points, coeffs, dcoeffs)
        ltt[j] = factor * (z - z_tr)
        for k in range(6):
            dltt[j, k] = factor * (dz[k] - dz_tr[k])
    return ltt, dltt


@njit(fastmath=True)
def rv_o5v_d(times, k, t0, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Radial velocity and parameter derivatives at array of times.

    Derivative ordering: ``(phase, p, a, i, e, w, k)`` — length 7.

    Returns
    -------
    rvs : ndarray (N,)
    drvs : ndarray (N, 7)
    """
    n = times.size
    rvs = zeros(n)
    drvs = zeros((n, 7))
    for j in range(n):
        t = times[j]
        epoch = floor((t - t0) / p)
        tc = t - t0 - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p
        rv_val, drv_orb = rvc_d(tcc, k, p, a, i, e, coeffs[ix], dcoeffs[ix])
        rvs[j] = rv_val
        for kk in range(6):
            drvs[j, kk] = drv_orb[kk]
        # drv/dk = rv / k  (rv is linear in k via the scale factor s = k/n).
        drvs[j, 6] = rv_val / k if k != 0.0 else 0.0
    return rvs, drvs
