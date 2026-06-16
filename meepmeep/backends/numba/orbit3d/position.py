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

"""Multi-expansion-point planet (x, y, z) position evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.position import pos_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _pos_os(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`pos_o`. See that function for documentation."""
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return pos_c(tc - ep_times[ix] * p, coeffs[ix])


@njit(fastmath=True)
def pos_ov(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`pos_o`. See that function for documentation."""
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = _pos_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return xs, ys, zs


@njit(fastmath=True, parallel=True)
def pos_ovp(times, tpa, p, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`pos_ov`."""
    n = times.size
    xs, ys, zs = zeros(n), zeros(n), zeros(n)
    for i in prange(n):
        xs[i], ys[i], zs[i] = _pos_os(times[i], tpa, p, dt, ep_table, ep_times, coeffs)
    return xs, ys, zs


def pos_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Planet (x, y, z) position for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_pos_os`) or vector (:func:`pos_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the position.
    tpa : float
        Periastron time anchoring the expansion-point grid.
    p : float
        Orbital period [days].
    dt : float
        ``ep_table`` bucket width in fraction of the period.
    ep_table : ndarray of int
        Time-to-expansion-point lookup table.
    ep_times : ndarray, shape (npt,)
        Normalised expansion-point phases in ``[0, 1]``.
    coeffs : ndarray, shape (npt, 3, 5)
        Per-expansion-point Taylor coefficient matrices from :func:`solve3d_orbit`.

    Returns
    -------
    x, y, z : float or ndarray
        Planet position in units of the stellar radius. ``x``, ``y`` are
        the sky-plane coordinates; ``z`` is the line-of-sight depth
        (positive toward the observer). Arrays of shape (N,) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return pos_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return _pos_os(t, tpa, p, dt, ep_table, ep_times, coeffs)


@overload(pos_o, jit_options={'fastmath': True})
def _pos_o_overload(t, tpa, p, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return pos_ov(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs):
            return _pos_os(t, tpa, p, dt, ep_table, ep_times, coeffs)
        return impl
    return None
