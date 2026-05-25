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

"""Multi-knot evaluators for the angle between the planet and a fixed vector."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _cos_v_p_angle_os(v, t, tpa, p, dt, pktable, points, coeffs):
    """Cosine of the angle between the planet position and a fixed reference vector at scalar time.

    Scalar counterpart of :func:`_cos_v_p_angle_ov`.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Reference vector. Need not be unit-norm.
    t : float
        Time at which to evaluate the angle.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    cos_theta : float
        Cosine of the angle, in :math:`[-1, 1]`.
    """
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    x, y, z = _pos_os(t, tpa, p, dt, pktable, points, coeffs)
    return (x * v[0] + y * v[1] + z * v[2]) * inv_nv / sqrt(x * x + y * y + z * z)


@njit(fastmath=True)
def _cos_v_p_angle_ov(v, times, tpa, p, dt, pktable, points, coeffs):
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
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

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
        x, y, z = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
        out[i] = (x * v[0] + y * v[1] + z * v[2]) * inv_nv / sqrt(x * x + y * y + z * z)
    return out


def cos_v_p_angle_o(v, t, tpa, p, dt, pktable, points, coeffs):
    """Cosine of angle between planet position and a fixed vector.

    See :func:`_cos_v_p_angle_os` / :func:`_cos_v_p_angle_ov`.
    """
    if isinstance(t, ndarray):
        return _cos_v_p_angle_ov(v, t, tpa, p, dt, pktable, points, coeffs)
    return _cos_v_p_angle_os(v, t, tpa, p, dt, pktable, points, coeffs)


@overload(cos_v_p_angle_o, jit_options={'fastmath': True})
def _cos_v_p_angle_o_overload(v, t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(v, t, tpa, p, dt, pktable, points, coeffs):
            return _cos_v_p_angle_ov(v, t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(v, t, tpa, p, dt, pktable, points, coeffs):
            return _cos_v_p_angle_os(v, t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
