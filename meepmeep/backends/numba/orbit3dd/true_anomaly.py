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

"""Multi-knot true-anomaly evaluators with parameter derivatives.

The geometric definition uses the angle between the planet position vector
and the eccentricity vector. Differentiating that (with the prograde sign
correction from r·v) gives a well-defined gradient everywhere except at
the two singular configurations ``edp = ±1`` (planet on the apsidal line).
At those points the analytic derivative diverges; we set it to zero so
downstream gradient-based fits don't get a NaN. The circular fast path
(``ex ≤ -0.9999`` sentinel from ``eccentricity_vector``) collapses true
anomaly to mean anomaly: ``f = 2π(t - tpa)/p`` ⇒ analytic derivatives are
trivial in (tc, p) and zero in the rest.
"""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, pi, floor, sqrt, arccos, ndarray

from ..point3dd.position import pos_cd
from ..point3dd.velocity import vel_cd
from ._common import _is_1d_array


@njit
def _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`true_anomaly_od`. See that function for documentation."""
    df = zeros(7)
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        twopi = 2.0 * pi
        tau = t - tpa
        epoch = floor(tau / p)
        tau_red = tau - epoch * p
        f = twopi * tau_red / p
        df[0] = -twopi / p
        df[1] = -twopi * tau_red / (p * p)
        return f, df

    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    tcc = tc - points[ix] * p
    c = coeffs[ix]
    dc = dcoeffs[ix]

    x, y, z, dx, dy, dz = pos_cd(tcc, c, dc)
    vx, vy, vz, dvx, dvy, dvz = vel_cd(tcc, c, dc)

    r2 = x * x + y * y + z * z
    sqrt_r2_nes = sqrt(r2 * nes)
    edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes
    rdotv = x * vx + y * vy + z * vz

    if edp <= -1.0:
        return pi, df
    if edp >= 1.0:
        return 0.0, df

    sign = 1.0 if rdotv > 0.0 else -1.0
    base = arccos(edp)
    f = base if sign > 0.0 else 2.0 * pi - base
    denom = sqrt(1.0 - edp * edp)
    xdote = x * ex + y * ey + z * ez
    for k in range(7):
        dxdote = dx[k] * ex + dy[k] * ey + dz[k] * ez
        xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
        dedp = dxdote / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
        df_k = -dedp / denom
        df[k] = df_k if sign > 0.0 else -df_k
    return f, df


@njit
def _true_anomaly_ovd(times, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`true_anomaly_od`. See that function for documentation."""
    n = times.size
    f = zeros(n)
    df = zeros((n, 7))
    nes = ex * ex + ey * ey + ez * ez

    # Circular-orbit fast path: f = 2π·(t - tpa) / p.
    # Slot 0 is d/dtc (transit-centre time). The orbit depends on (t - tc),
    # so df/dtc = -2π/p, matching solve3d_d's slot-0 convention used by the
    # eccentric branch below.
    # df/dp = -2π·(t - tpa) / p^2.
    if ex <= -0.9999 and nes > 0.99:
        twopi = 2.0 * pi
        for j in range(n):
            tau = times[j] - tpa
            # Reduce to one period for the value (mean_anomaly does this in base).
            epoch = floor(tau / p)
            tau_red = tau - epoch * p
            f[j] = twopi * tau_red / p
            df[j, 0] = -twopi / p
            df[j, 1] = -twopi * tau_red / (p * p)
        return f, df

    for j in range(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p
        c = coeffs[ix]
        dc = dcoeffs[ix]

        x, y, z, dx, dy, dz = pos_cd(tcc, c, dc)
        vx, vy, vz, dvx, dvy, dvz = vel_cd(tcc, c, dc)

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
            for k in range(7):
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


def true_anomaly_od(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """True anomaly and its orbital-parameter derivatives.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_true_anomaly_osd`) or vector (:func:`_true_anomaly_ovd`)
    kernel at compile time (inside ``@njit``) or at call time (pure Python).

    Computed from the geometric angle between the planet position vector
    and the eccentricity vector :math:`(e_x, e_y, e_z)`, with
    :math:`r \\cdot v` resolving the two branches of :math:`\\arccos`.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the true anomaly and gradient.
    tpa : float
        Periastron time anchoring the knot grid (see :func:`_pos_osd`).
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector. ``(-1, 0, 0)`` is the
        sentinel produced by
        :func:`~meepmeep.backends.numba.utils.eccentricity_vector` for
        near-circular orbits and triggers the closed-form fast path.
    w : float
        Argument of periastron [radians]. Kept for signature parity with
        the base function; currently unused inside this routine because
        the eccentricity vector is passed explicitly.
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit_d` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    f : float or ndarray
        True anomaly [radians], in :math:`[0, 2\\pi)`. Arrays of shape (N,)
        for an array ``t``.
    df : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a scalar
        ``t``, (N, 7) for an array ``t``. The ``ex, ey, ez, w`` inputs are
        treated as known constants - they are functions of the orbital
        parameters but the dependency is captured implicitly through the
        geometric chain rule on the position vector.

    Notes
    -----
    At the singular configurations ``edp = +/-1`` (``edp`` = cosine of the
    angle between position and eccentricity vector) the analytic gradient
    diverges and is replaced by zero. The circular-orbit fast path uses
    the mean-anomaly identity :math:`f = 2\\pi(t - t_\\mathrm{pa}) / p`.
    """
    if isinstance(t, ndarray):
        return _true_anomaly_ovd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)
    return _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)


@overload(true_anomaly_od)
def _true_anomaly_od_overload(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
            return _true_anomaly_ovd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
            return _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
