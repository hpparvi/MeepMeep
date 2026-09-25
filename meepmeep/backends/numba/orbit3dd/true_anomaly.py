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

"""Multi-expansion-point true-anomaly evaluators with parameter derivatives.

The geometric definition uses the angle between the planet position vector
and the eccentricity vector. Differentiating that with respect to both
vectors (with the prograde sign correction from the mean anomaly) gives a
well-defined gradient everywhere except at
the two singular configurations ``edp = +-1`` (planet on the apsidal line).
At those ep_times the analytic derivative diverges; we set it to zero so
downstream gradient-based fits don't get a NaN. The circular fast path
(``ex <= -0.9999`` sentinel from ``eccentricity_vector``) collapses true
anomaly to mean anomaly: ``f = 2 pi (t - tpa)/p``. Its gradient does not come
from ``dcoeffs``, so the caller states the basis with ``timing_is_tc``: in
the periastron basis only the timing and period slots are non-zero, in the
transit-centre basis ``tpa`` moves with ``p``, ``e`` and ``w`` as well.
"""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, pi, floor, sqrt, arccos, ndarray

from ..point3dd.position import pos_cd, _pos_cd_w
from ..utils import mean_anomaly_at_transit_with_derivatives
from ._common import _is_1d_array


@njit(inline='always')
def _circular_w(t, tpa, p, w, timing_is_tc, df):
    """Circular-orbit fast path: the true anomaly is the mean anomaly.

    Returns ``f = 2 pi (t - tpa) / p`` folded into one period and writes its
    gradient into the caller's zeroed ``(7,)`` row ``df``. In the periastron
    basis only the timing and period slots are non-zero. In the
    transit-centre basis ``tpa = tc - M_tr(e, w) p / (2 pi)`` moves with
    ``p``, ``e`` and ``w`` too, which adds ``-df[0] * dtpa/dtheta`` to those
    slots (the ``tp_to_tc_gradient`` transform). The sentinel stands for
    ``e ~ 0``, so ``M_tr`` and its derivatives are taken at ``e = 0``. The
    row is only written, never read back, so the inlined vector loops avoid
    the numba 0.61 miscompilation described in
    :func:`~meepmeep.backends.numba.point3dd.zposition._zpos_cd_w`.
    """
    twopi = 2.0 * pi
    tau = t - tpa
    epoch = floor(tau / p)
    tau_red = tau - epoch * p
    d0 = -twopi / p
    # Period slot, including the period-folding chain term (see position._pos_ow).
    d1 = -twopi * tau_red / (p * p) + epoch * d0
    df[0] = d0
    if timing_is_tc:
        m_tr, dm_tr_de, dm_tr_dw = mean_anomaly_at_transit_with_derivatives(0.0, w)
        df[1] = d1 - d0 * m_tr / twopi
        df[4] = -d0 * dm_tr_de * p / twopi
        df[5] = -d0 * dm_tr_dw * p / twopi
    else:
        df[1] = d1
    return twopi * tau_red / p


@njit
def _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc=True):
    """Scalar kernel for :func:`true_anomaly_od`. See that function for documentation."""
    df = zeros(7)
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        f = _circular_w(t, tpa, p, w, timing_is_tc, df)
        return f, df

    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    tcc = tc - ep_times[ix] * p
    c = coeffs[ix]
    dc = dcoeffs[ix]

    x, y, z, dx, dy, dz = pos_cd(tcc, c, dc)

    r2 = x * x + y * y + z * z
    sqrt_r2_nes = sqrt(r2 * nes)
    edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes

    if edp <= -1.0:
        return pi, df
    if edp >= 1.0:
        return 0.0, df

    # Branch selection from the mean anomaly: the folded time since
    # periastron gives M = 2*pi*tc/p exactly, and f and M always share the
    # half-plane, so M selects the arccos branch. The sign of r.v would do
    # the same in exact arithmetic, but it is O(e) and drowns in the Taylor
    # truncation noise for near-circular orbits.
    sign = 1.0 if tc < 0.5 * p else -1.0
    base = arccos(edp)
    f = base if sign > 0.0 else 2.0 * pi - base
    denom = sqrt(1.0 - edp * edp)
    xdote = x * ex + y * ey + z * ez
    for k in range(7):
        dxdote = dx[k] * ex + dy[k] * ey + dz[k] * ez
        xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
        xdotdev = x * dev[0, k] + y * dev[1, k] + z * dev[2, k]
        edotdev = ex * dev[0, k] + ey * dev[1, k] + ez * dev[2, k]
        dedp = ((dxdote + xdotdev) / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
                - xdote * edotdev / (nes * sqrt_r2_nes))
        df_k = -dedp / denom
        df[k] = df_k if sign > 0.0 else -df_k
    # Period-folding chain term (see position._pos_ow).
    df[1] += epoch * df[0]
    return f, df


@njit
def true_anomaly_ovd(times, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc=True):
    """Vector kernel for :func:`true_anomaly_od`. See that function for documentation."""
    n = times.size
    f = zeros(n)
    df = zeros((n, 7))
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        for j in range(n):
            f[j] = _circular_w(times[j], tpa, p, w, timing_is_tc, df[j])
        return f, df

    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    for j in range(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = ep_table[int(floor(tc / (dt * p)))]
        tcc = tc - ep_times[ix] * p
        c = coeffs[ix]
        dc = dcoeffs[ix]

        x, y, z = _pos_cd_w(tcc, c, dc, dx, dy, dz)

        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        sqrt_r2_nes = sqrt(r2 * nes)
        edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes

        if edp <= -1.0:
            f[j] = pi
            # Singular: leave df[j] = 0.
        elif edp >= 1.0:
            f[j] = 0.0
            # Singular: leave df[j] = 0.
        else:
            # Branch selection from the mean anomaly; see _true_anomaly_osd.
            sign = 1.0 if tc < 0.5 * p else -1.0
            base = arccos(edp)
            f[j] = base if sign > 0.0 else 2.0 * pi - base
            # d(arccos(edp))/dtheta = -dedp/sqrt(1 - edp^2)
            denom = sqrt(1.0 - edp * edp)
            for k in range(7):
                # edp = dot(x, e)/(r |e|), with both x and the eccentricity
                # vector e depending on the parameters (de = dev[:, k]):
                # d(edp)/dtheta_k = (dot(dx, e) + dot(x, de))/(r |e|)
                #                 - dot(x, e) dot(x, dx)/(r^3 |e|) - dot(x, e) dot(e, de)/(r |e|^3)
                xdote = x * ex + y * ey + z * ez
                dxdote = dx[k] * ex + dy[k] * ey + dz[k] * ez
                xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
                xdotdev = x * dev[0, k] + y * dev[1, k] + z * dev[2, k]
                edotdev = ex * dev[0, k] + ey * dev[1, k] + ez * dev[2, k]
                dedp = ((dxdote + xdotdev) / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
                        - xdote * edotdev / (nes * sqrt_r2_nes))
                df_k = -dedp / denom
                df[j, k] = df_k if sign > 0.0 else -df_k
            # Period-folding chain term (see position._pos_ow).
            df[j, 1] += epoch * df[j, 0]
    return f, df


@njit(parallel=True)
def true_anomaly_ovdp(times, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc=True):
    """Parallel (prange) twin of :func:`true_anomaly_ovd`.

    Mirrors the serial vector body (rather than looping the scalar kernel)
    so the positions are evaluated under the same non-fastmath flags: near
    the apsides ``df`` is sensitive to ulp-level differences in ``edp``,
    so the twin must match the serial kernel's rounding exactly. The
    position-gradient scratch is hoisted per thread.
    """
    n = times.size
    f = zeros(n)
    df = zeros((n, 7))
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        for j in prange(n):
            f[j] = _circular_w(times[j], tpa, p, w, timing_is_tc, df[j])
        return f, df

    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = ep_table[int(floor(tc / (dt * p)))]
        tcc = tc - ep_times[ix] * p

        x, y, z = _pos_cd_w(tcc, coeffs[ix], dcoeffs[ix], dx, dy, dz)

        r2 = x * x + y * y + z * z
        sqrt_r2_nes = sqrt(r2 * nes)
        edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes

        if edp <= -1.0:
            f[j] = pi
            # Singular: leave df[j] = 0.
        elif edp >= 1.0:
            f[j] = 0.0
            # Singular: leave df[j] = 0.
        else:
            # Branch selection from the mean anomaly; see _true_anomaly_osd.
            sign = 1.0 if tc < 0.5 * p else -1.0
            base = arccos(edp)
            f[j] = base if sign > 0.0 else 2.0 * pi - base
            denom = sqrt(1.0 - edp * edp)
            for kk in range(7):
                xdote = x * ex + y * ey + z * ez
                dxdote = dx[kk] * ex + dy[kk] * ey + dz[kk] * ez
                xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
                xdotdev = x * dev[0, kk] + y * dev[1, kk] + z * dev[2, kk]
                edotdev = ex * dev[0, kk] + ey * dev[1, kk] + ez * dev[2, kk]
                dedp = ((dxdote + xdotdev) / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
                        - xdote * edotdev / (nes * sqrt_r2_nes))
                df_k = -dedp / denom
                df[j, kk] = df_k if sign > 0.0 else -df_k
            # Period-folding chain term (see position._pos_ow).
            df[j, 1] += epoch * df[j, 0]
    return f, df


def true_anomaly_od(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc=True):
    """True anomaly and its orbital-parameter derivatives.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_true_anomaly_osd`) or vector (:func:`true_anomaly_ovd`)
    kernel at compile time (inside ``@njit``) or at call time (pure Python).

    Computed from the geometric angle between the planet position vector
    and the eccentricity vector :math:`(e_x, e_y, e_z)`, with the mean
    anomaly (computed exactly from the periastron anchor) resolving the
    two branches of :math:`\\arccos`.

    Parameters
    ----------
    t : float or NDArray
        Time(s) at which to evaluate the true anomaly and gradient.
    tpa : float
        Periastron time anchoring the expansion-point grid (see :func:`_pos_osd`).
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector. ``(-1, 0, 0)`` is the
        sentinel produced by
        :func:`~meepmeep.backends.numba.utils.eccentricity_vector` for
        near-circular orbits and triggers the closed-form fast path.
    w : float
        Argument of periastron [radians]. Used only by the circular fast path,
        whose transit-centre-basis gradient depends on it through the
        mean anomaly at transit.
    dev : NDArray, shape (3, 7)
        Jacobian of ``(ex, ey, ez)`` with respect to ``(tc, p, a, i, e, w, lan)``,
        from :func:`~meepmeep.numba3d.eccentricity_vector_d`. The
        eccentricity vector turns with ``w`` and ``lan``, so this term is what
        cancels the position gradients' ``w`` and ``lan`` dependence. Pass
        zeros to hold the vector constant. Unused by the circular fast path.
    dt, ep_table, ep_times, coeffs, dcoeffs
        Multi-expansion-point dispatch arrays from :func:`solve3d_orbit_d` /
        :func:`~meepmeep.backends.numba.expansion_points.create_expansion_points`.
    timing_is_tc : bool, optional
        State the basis of ``dcoeffs``: True (default) for the transit-centre
        basis (after ``tp_to_tc_gradient_orbit``), False for the periastron
        basis ``solve3d_orbit_d`` returns. The eccentric path inherits the
        basis from ``dcoeffs``; the circular fast path, which does not read
        ``dcoeffs``, needs to be told.

    Returns
    -------
    f : float or NDArray
        True anomaly [radians], in :math:`[0, 2\\pi)`. Arrays of shape (N,)
        for an array ``t``.
    df : NDArray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a scalar
        ``t``, (N, 7) for an array ``t``. The eccentricity vector enters
        through ``dev``, so the gradient is the full derivative when ``dev``
        comes from :func:`~meepmeep.numba3d.eccentricity_vector_d`.

    Notes
    -----
    At the singular configurations ``edp = +/-1`` (``edp`` = cosine of the
    angle between position and eccentricity vector) the analytic gradient
    diverges and is replaced by zero. The circular-orbit fast path uses
    the mean-anomaly identity :math:`f = 2\\pi(t - t_\\mathrm{pa}) / p`, with
    the mean anomaly at transit taken at ``e = 0`` in its basis transform.
    """
    if isinstance(t, ndarray):
        return true_anomaly_ovd(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc)
    return _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc)


@overload(true_anomaly_od)
def _true_anomaly_od_overload(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs,
                              timing_is_tc=True):
    if _is_1d_array(t):
        def impl(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc=True):
            return true_anomaly_ovd(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs,
                                    timing_is_tc)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs, timing_is_tc=True):
            return _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dev, dt, ep_table, ep_times, coeffs, dcoeffs,
                                     timing_is_tc)
        return impl
    return None
