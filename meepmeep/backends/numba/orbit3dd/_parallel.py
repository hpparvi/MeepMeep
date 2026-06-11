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

"""Parallel (prange) twins of the multi-knot gradient vector kernels.

Every ``_X_ovd`` kernel has a ``_X_ovdp`` twin here, compiled with
``parallel=True`` and a ``prange`` sample loop. The simple kernels write
only into per-sample output rows and parallelise as-is. The derived
kernels reuse intermediate-gradient scratch buffers; the serial versions
hoist a single shared buffer out of the loop, which would be a data race
under ``prange``, so the twins hoist one buffer *per thread*
(``zeros((get_num_threads(), 7))``) and index it with ``get_thread_id()``
inside the loop.

The twins are opt-in (used by :class:`~meepmeep.orbit.Orbit` when
constructed with ``parallel=True``); the public ``X_od`` dispatchers
always route to the serial kernels. For the gradient kernels the
break-even is around 1e4 samples.
"""

from numba import njit, prange, get_num_threads, get_thread_id
from numpy import zeros, pi, floor, sqrt, arccos, cos, sin

from ..point3dd.position import _pos_cd_w
from ..point3dd.radial_velocity import _rv_scale, _rv_cd_w
from .position import _pos_ow
from .zposition import _zpos_ow
from .separation import _sep_ow
from .velocity import _vel_ow
from .zvelocity import _zvel_ow
from .phase_angle import _cos_alpha_ow
from .lambert import _lambert_kernel_d
from .light_travel_time import _ltt_transit_z_and_d, LTT_DAYS_PER_RSUN


@njit(fastmath=True, parallel=True)
def _pos_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.position._pos_ovd`."""
    n = times.size
    xs, ys, zs = zeros(n), zeros(n), zeros(n)
    dxs, dys, dzs = zeros((n, 7)), zeros((n, 7)), zeros((n, 7))
    for j in prange(n):
        xs[j], ys[j], zs[j] = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                                      dxs[j], dys[j], dzs[j])
    return xs, ys, zs, dxs, dys, dzs


@njit(fastmath=True, parallel=True)
def _zpos_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.zposition._zpos_ovd`."""
    n = times.size
    zs = zeros(n)
    dzs = zeros((n, 7))
    for j in prange(n):
        zs[j] = _zpos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dzs[j])
    return zs, dzs


@njit(fastmath=True, parallel=True)
def _sep_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.separation._sep_ovd`."""
    n = times.size
    ds = zeros(n)
    dds = zeros((n, 7))
    for j in prange(n):
        ds[j] = _sep_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dds[j])
    return ds, dds


@njit(fastmath=True, parallel=True)
def _vel_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.velocity._vel_ovd`."""
    n = times.size
    vxs, vys, vzs = zeros(n), zeros(n), zeros(n)
    dvxs, dvys, dvzs = zeros((n, 7)), zeros((n, 7)), zeros((n, 7))
    for j in prange(n):
        vxs[j], vys[j], vzs[j] = _vel_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                                         dvxs[j], dvys[j], dvzs[j])
    return vxs, vys, vzs, dvxs, dvys, dvzs


@njit(fastmath=True, parallel=True)
def _zvel_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.zvelocity._zvel_ovd`."""
    n = times.size
    vzs = zeros(n)
    dvzs = zeros((n, 7))
    for j in prange(n):
        vzs[j] = _zvel_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dvzs[j])
    return vzs, dvzs


@njit(fastmath=True, parallel=True)
def _rv_ovdp(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.radial_velocity._rv_ovd`."""
    n = times.size
    rvs = zeros(n)
    drvs = zeros((n, 8))
    s, dsp, dsa, dsi, dse = _rv_scale(k, p, a, i, e)
    dvz = zeros((get_num_threads(), 7))
    for j in prange(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        rv_val = _rv_cd_w(tc - points[ix] * p, s, dsp, dsa, dsi, dse,
                          coeffs[ix], dcoeffs[ix], drvs[j, :7], dvz[get_thread_id()])
        rvs[j] = rv_val
        drvs[j, 7] = rv_val / k if k != 0.0 else 0.0
    return rvs, drvs


@njit(fastmath=True, parallel=True)
def _cos_alpha_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.phase_angle._cos_alpha_ovd`."""
    n = times.size
    cas = zeros(n)
    dcas = zeros((n, 7))
    nt = get_num_threads()
    dx, dy, dz = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        cas[j] = _cos_alpha_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                               dcas[j], dx[tid], dy[tid], dz[tid])
    return cas, dcas


@njit(fastmath=True, parallel=True)
def _cos_v_p_angle_ovdp(v, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.projected_angle._cos_v_p_angle_ovd`."""
    n = times.size
    cs = zeros(n)
    dcs = zeros((n, 7))
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        x, y, z = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        inv_r = 1.0 / r
        inv_r3 = inv_r / r2
        dot = x * v[0] + y * v[1] + z * v[2]
        cs[j] = dot * inv_nv * inv_r
        for kk in range(7):
            ddot = dx[kk] * v[0] + dy[kk] * v[1] + dz[kk] * v[2]
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dcs[j, kk] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3)
    return cs, dcs


@njit(fastmath=True, parallel=True)
def _star_planet_distance_ovdp(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.star_planet_distance._star_planet_distance_ovd`."""
    n = times.size
    rs = zeros(n)
    drs = zeros((n, 7))
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        x, y, z = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
        r = sqrt(x * x + y * y + z * z)
        rs[j] = r
        inv_r = 1.0 / r
        for kk in range(7):
            drs[j, kk] = (x * dx[kk] + y * dy[kk] + z * dz[kk]) * inv_r
    return rs, drs


@njit(parallel=True)
def _true_anomaly_ovdp(times, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.true_anomaly._true_anomaly_ovd`.

    Mirrors the serial vector body (rather than looping the scalar kernel)
    so the positions are evaluated under the same non-fastmath flags: near
    the apsides ``df`` is sensitive to ulp-level differences in ``edp``,
    so the twin must match the serial kernel's rounding exactly.
    """
    n = times.size
    f = zeros(n)
    df = zeros((n, 7))
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        twopi = 2.0 * pi
        for j in prange(n):
            tau = times[j] - tpa
            epoch = floor(tau / p)
            tau_red = tau - epoch * p
            f[j] = twopi * tau_red / p
            df[j, 0] = -twopi / p
            df[j, 1] = -twopi * tau_red / (p * p)
        return f, df

    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p

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
                dedp = dxdote / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
                df_k = -dedp / denom
                df[j, kk] = df_k if sign > 0.0 else -df_k
    return f, df


@njit(fastmath=True, parallel=True)
def _lambert_phase_curve_ovdp(times, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.lambert._lambert_phase_curve_ovd`."""
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 9))
    inv_a2 = 1.0 / (a * a)
    amplitude = k * k * ag * inv_a2
    da_amp = -2.0 * k * k * ag / (a * a * a)
    dag_amp = k * k * inv_a2
    dk_amp = 2.0 * k * ag * inv_a2
    nt = get_num_threads()
    dcas, dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dca = dcas[tid]
        ca = _cos_alpha_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                           dca, dxs[tid], dys[tid], dzs[tid])
        phase, _, dphase_dc = _lambert_kernel_d(ca)
        flux[j] = amplitude * phase
        for kk in range(7):
            dflux[j, kk] = amplitude * dphase_dc * dca[kk]
        dflux[j, 2] += da_amp * phase
        dflux[j, 7] = dag_amp * phase
        dflux[j, 8] = dk_amp * phase
    return flux, dflux


@njit(fastmath=True, parallel=True)
def _lambert_and_emission_ovdp(times, ag, fr_night, fr_day, emi_offset, a, k,
                               tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.lambert._lambert_and_emission_ovd`."""
    n = times.size
    ref = zeros(n)
    emi = zeros(n)
    dref = zeros((n, 12))
    demi = zeros((n, 12))
    k2 = k * k
    inv_a2 = 1.0 / (a * a)
    aref = k2 * ag * inv_a2
    daref_da = -2.0 * k2 * ag / (a * a * a)
    daref_dag = k2 * inv_a2
    daref_dk = 2.0 * k * ag * inv_a2
    nt = get_num_threads()
    dcas, dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dca = dcas[tid]
        ca = _cos_alpha_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                           dca, dxs[tid], dys[tid], dzs[tid])
        phase, alpha, dphase_dc = _lambert_kernel_d(ca)

        ref[j] = aref * phase
        for kk in range(7):
            dref[j, kk] = aref * dphase_dc * dca[kk]
        dref[j, 2] += daref_da * phase
        dref[j, 7] = daref_dag * phase
        dref[j, 11] = daref_dk * phase

        cs = cos(alpha + emi_offset)
        sn = sin(alpha + emi_offset)
        bracket = fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cs)
        emi[j] = k2 * bracket

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
        demi_dalpha = k2 * (fr_day - fr_night) * 0.5 * sn
        for kk in range(7):
            demi[j, kk] = demi_dalpha * dalpha_dc * dca[kk]
        demi[j, 8] = k2 * (1.0 - 0.5 * (1.0 - cs))
        demi[j, 9] = k2 * 0.5 * (1.0 - cs)
        demi[j, 10] = k2 * (fr_day - fr_night) * 0.5 * sn
        demi[j, 11] = 2.0 * k * bracket
    return ref, emi, dref, demi


@njit(fastmath=True, parallel=True)
def _ev_signal_ovdp(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.ev_signal._ev_signal_ovd`."""
    n = times.size
    out = zeros(n)
    dout = zeros((n, 10))
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        x, y, z = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        g = (2.0 * cz * cz - 1.0) / (d2 * d)
        out[j] = pre * g
        d5 = d2 * d2 * d
        A = 2.0 * z * z - d2
        for kk in range(7):
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dd = xdotdx / d
            dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
            dg = (dA - 5.0 * A * dd / d2) / d5
            dout[j, kk] = pre * dg
        dout[j, 7] = -mass_ratio * sin2_inc * g
        dout[j, 8] = -alpha * sin2_inc * g
        dout[j, 9] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g
    return out, dout


@njit(fastmath=True, parallel=True)
def _light_travel_time_ovdp(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Parallel twin of :func:`~meepmeep.backends.numba.orbit3dd.light_travel_time._light_travel_time_ovd`."""
    n = times.size
    ltt = zeros(n)
    dltt = zeros((n, 7))
    factor = -rstar * LTT_DAYS_PER_RSUN
    z_tr, dz_tr = _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    dz = zeros((get_num_threads(), 7))
    for j in prange(n):
        dzj = dz[get_thread_id()]
        z = _zpos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dzj)
        ltt[j] = factor * (z - z_tr)
        for kk in range(7):
            dltt[j, kk] = factor * (dzj[kk] - dz_tr[kk])
    return ltt, dltt
