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

"""Multi-knot radial-velocity evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.radial_velocity import rv_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`rv_od`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    tcc = tc - points[ix] * p
    rv_val, drv_orb = rv_cd(tcc, k, p, a, i, e, coeffs[ix], dcoeffs[ix])
    drv = zeros(8)
    for kk in range(7):
        drv[kk] = drv_orb[kk]
    drv[7] = rv_val / k if k != 0.0 else 0.0
    return rv_val, drv


@njit(fastmath=True)
def _rv_ovd(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`rv_od`. See that function for documentation."""
    n = times.size
    rvs = zeros(n)
    drvs = zeros((n, 8))
    for j in range(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p
        rv_val, drv_orb = rv_cd(tcc, k, p, a, i, e, coeffs[ix], dcoeffs[ix])
        rvs[j] = rv_val
        for kk in range(7):
            drvs[j, kk] = drv_orb[kk]
        # drv/dk = rv / k  (rv is linear in k via the scale factor s = k/n).
        drvs[j, 7] = rv_val / k if k != 0.0 else 0.0
    return rvs, drvs


def rv_od(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Radial velocity and parameter derivatives.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_rv_osd`) or vector (:func:`_rv_ovd`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Derivative ordering: ``(tc, p, a, i, e, w, lan, k)`` - length 8.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the radial velocity and gradient.
    k : float
        Radial-velocity semi-amplitude [m s\\ :sup:`-1`].
    tpa : float
        Periastron time anchoring the knot grid (see :func:`_pos_osd`).
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    rv : float or ndarray
        Radial velocity [m s\\ :sup:`-1`]. Arrays of shape (N,) for an array
        ``t``.
    drv : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan, k)``. Shape (8,) for a
        scalar ``t``, (N, 8) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _rv_ovd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)
    return _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)


@overload(rv_od, jit_options={'fastmath': True})
def _rv_od_overload(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
            return _rv_ovd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
            return _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
