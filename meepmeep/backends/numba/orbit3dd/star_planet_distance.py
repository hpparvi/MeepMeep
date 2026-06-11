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

"""Multi-knot 3D star-planet distance evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_ow
from ._common import _is_1d_array


@njit(fastmath=True)
def _star_planet_distance_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`star_planet_distance_od`. See that function for documentation."""
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    x, y, z = _pos_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
    r = sqrt(x * x + y * y + z * z)
    inv_r = 1.0 / r
    dr = zeros(7)
    for k in range(7):
        dr[k] = (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r
    return r, dr


@njit(fastmath=True)
def _star_planet_distance_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`star_planet_distance_od`. See that function for documentation."""
    n = times.size
    rs = zeros(n)
    drs = zeros((n, 7))
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    for j in range(n):
        x, y, z = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
        r = sqrt(x * x + y * y + z * z)
        rs[j] = r
        inv_r = 1.0 / r
        for k in range(7):
            drs[j, k] = (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r
    return rs, drs


def star_planet_distance_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """3D star-planet distance and orbital-parameter derivatives.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_star_planet_distance_osd`) or vector
    (:func:`_star_planet_distance_ovd`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Returns :math:`r = \\sqrt{x^2 + y^2 + z^2}` and
    :math:`\\partial r/\\partial \\theta_k = (x\\,\\partial x/\\partial \\theta_k
    + y\\,\\partial y/\\partial \\theta_k + z\\,\\partial z/\\partial \\theta_k)/r`.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    r : float or ndarray
        3D star-planet distance [stellar radii]. Arrays of shape (N,) for an
        array ``t``.
    dr : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a scalar
        ``t``, (N, 7) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _star_planet_distance_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _star_planet_distance_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(star_planet_distance_od, jit_options={'fastmath': True})
def _star_planet_distance_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _star_planet_distance_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _star_planet_distance_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
