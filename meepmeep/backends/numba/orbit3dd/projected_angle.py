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

"""Multi-expansion-point evaluators for the angle to a fixed vector, with parameter derivatives."""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_ow
from ._common import _is_1d_array


@njit(fastmath=True)
def _cos_v_p_angle_osd(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Scalar kernel for :func:`cos_v_p_angle_od`. See that function for documentation."""
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    x, y, z = _pos_ow(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dx, dy, dz)
    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    dot = x * v[0] + y * v[1] + z * v[2]
    cs = dot * inv_nv * inv_r
    dcs = zeros(7)
    for k in range(7):
        ddot = dx[k] * v[0] + dy[k] * v[1] + dz[k] * v[2]
        xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
        dcs[k] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3)
    return cs, dcs


@njit(fastmath=True)
def cos_v_p_angle_ovd(v, times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Vector kernel for :func:`cos_v_p_angle_od`. See that function for documentation."""
    n = times.size
    cs = zeros(n)
    dcs = zeros((n, 7))
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    for j in range(n):
        x, y, z = _pos_ow(times[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dx, dy, dz)
        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        inv_r = 1.0 / r
        inv_r3 = inv_r / r2
        dot = x * v[0] + y * v[1] + z * v[2]
        cs[j] = dot * inv_nv * inv_r
        # d/dθ[(x·v)/(|x|·|v|)] = ((dx·v)/|x| - (x·v)·(x·dx)/|x|^3) / |v|
        for k in range(7):
            ddot = dx[k] * v[0] + dy[k] * v[1] + dz[k] * v[2]
            xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
            dcs[j, k] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3)
    return cs, dcs


@njit(fastmath=True, parallel=True)
def cos_v_p_angle_ovdp(v, times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`cos_v_p_angle_ovd`.

    The position-gradient scratch is hoisted per thread; a single shared
    buffer would be a data race under ``prange``.
    """
    n = times.size
    cs = zeros(n)
    dcs = zeros((n, 7))
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    nt = get_num_threads()
    dxs, dys, dzs = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        dx, dy, dz = dxs[tid], dys[tid], dzs[tid]
        x, y, z = _pos_ow(times[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dx, dy, dz)
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


def cos_v_p_angle_od(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Cosine of the angle between planet position and a fixed reference vector ``v``, with gradients.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_cos_v_p_angle_osd`) or vector (:func:`cos_v_p_angle_ovd`)
    kernel at compile time (inside ``@njit``) or at call time (pure Python).

    The reference vector ``v`` is treated as a constant; gradients are
    computed w.r.t. the seven orbital parameters only.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Fixed reference vector. Need not be unit-norm; the cosine is
        normalised internally.
    t : float or ndarray
        Time(s) at which to evaluate the cosine and gradient.
    tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    cs : float or ndarray
        Cosine of the angle. Arrays of shape (N,) for an array ``t``.
    dcs : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a scalar
        ``t``, (N, 7) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return cos_v_p_angle_ovd(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    return _cos_v_p_angle_osd(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)


@overload(cos_v_p_angle_od, jit_options={'fastmath': True})
def _cos_v_p_angle_od_overload(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return cos_v_p_angle_ovd(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return _cos_v_p_angle_osd(v, t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    return None
