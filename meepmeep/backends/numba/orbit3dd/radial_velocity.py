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

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.radial_velocity import _rv_scale, _rv_cd_w
from ._common import _is_1d_array


@njit(fastmath=True)
def _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`rv_od`. See that function for documentation."""
    s, dsp, dsa, dsi, dse = _rv_scale(k, p, a, i, e)
    drv = zeros(8)
    dvz = zeros(7)
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    rv_val = _rv_cd_w(tc - points[ix] * p, s, dsp, dsa, dsi, dse,
                      coeffs[ix], dcoeffs[ix], drv[:7], dvz)
    drv[7] = rv_val / k if k != 0.0 else 0.0
    return rv_val, drv


@njit(fastmath=True)
def _rv_ovd(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`rv_od`. See that function for documentation."""
    n = times.size
    rvs = zeros(n)
    drvs = zeros((n, 8))
    s, dsp, dsa, dsi, dse = _rv_scale(k, p, a, i, e)
    dvz = zeros(7)
    for j in range(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        rv_val = _rv_cd_w(tc - points[ix] * p, s, dsp, dsa, dsi, dse,
                          coeffs[ix], dcoeffs[ix], drvs[j, :7], dvz)
        rvs[j] = rv_val
        # drv/dk = rv / k  (rv is linear in k via the scale factor s = k/n).
        drvs[j, 7] = rv_val / k if k != 0.0 else 0.0
    return rvs, drvs


@njit(fastmath=True, parallel=True)
def _rv_ovdp(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`_rv_ovd`.

    The z-velocity gradient scratch is hoisted per thread; a single shared
    buffer would be a data race under ``prange``.
    """
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
