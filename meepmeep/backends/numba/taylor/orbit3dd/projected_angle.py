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

"""Multi-knot evaluators for the angle to a fixed vector, with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_osd
from ._common import _is_1d_array


@njit(fastmath=True)
def _cos_v_p_angle_osd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the angle between planet position and a fixed reference vector at scalar time.

    Scalar counterpart of :func:`_cos_v_p_angle_ovd`. The reference vector
    ``v`` is treated as a constant; gradients are w.r.t. the seven orbital
    parameters only.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Fixed reference vector.
    t : float
        Time at which to evaluate the cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    cs : float
        Cosine of the angle.
    dcs : ndarray, shape (7,)
        Gradient w.r.t. ``(phase, p, a, i, e, w, lan)``.
    """
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
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
def _cos_v_p_angle_ovd(v, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the angle between planet position and a fixed reference vector ``v``.

    The reference vector ``v`` is treated as a constant; gradients are
    computed w.r.t. the seven orbital parameters only.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Fixed reference vector. Need not be unit-norm; the cosine is
        normalised internally.
    times : ndarray, shape (N,)
        Times at which to evaluate the cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    cs : ndarray, shape (N,)
        Cosine values per time.
    dcs : ndarray, shape (N, 7)
        Gradients w.r.t. ``(phase, p, a, i, e, w, lan)`` per time.
    """
    n = times.size
    cs = zeros(n)
    dcs = zeros((n, 7))
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
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


def cos_v_p_angle_od(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of angle between planet position and fixed vector, with gradients.

    See :func:`_cos_v_p_angle_osd` / :func:`_cos_v_p_angle_ovd`.
    """
    if isinstance(t, ndarray):
        return _cos_v_p_angle_ovd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _cos_v_p_angle_osd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(cos_v_p_angle_od, jit_options={'fastmath': True})
def _cos_v_p_angle_od_overload(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _cos_v_p_angle_ovd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _cos_v_p_angle_osd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
