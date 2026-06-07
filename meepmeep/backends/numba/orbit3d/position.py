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

"""Multi-knot planet (x, y, z) position evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3d.position import pos_c
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _pos_os(t, tpa, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position at scalar time ``t`` for any orbital phase.

    Parameters
    ----------
    t : float
        Time at which to evaluate the position.
    tpa : float
        Periastron time anchoring the knot grid.
    p : float
        Orbital period [days].
    dt : float
        ``pktable`` bucket width in fraction of the period.
    pktable : ndarray of int
        Time-to-knot lookup table.
    points : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]``.
    coeffs : ndarray, shape (npt, 3, 5)
        Per-knot Taylor coefficient matrices from :func:`solve3d_orbit`.

    Returns
    -------
    x, y, z : float
        Planet position in units of the stellar radius. ``x``, ``y`` are
        the sky-plane coordinates; ``z`` is the line-of-sight depth
        (positive toward the observer).
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pos_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _pos_ov(times, tpa, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position at an array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the position.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_os`.

    Returns
    -------
    xs, ys, zs : ndarray, shape (N,)
        Planet position arrays in units of the stellar radius.
    """
    npt = times.size
    xs, ys, zs = zeros(npt), zeros(npt), zeros(npt)
    for i in range(npt):
        xs[i], ys[i], zs[i] = _pos_os(times[i], tpa, p, dt, pktable, points, coeffs)
    return xs, ys, zs


def pos_o(t, tpa, p, dt, pktable, points, coeffs):
    """Planet (x, y, z) position. See :func:`_pos_os` / :func:`_pos_ov`."""
    if isinstance(t, ndarray):
        return _pos_ov(t, tpa, p, dt, pktable, points, coeffs)
    return _pos_os(t, tpa, p, dt, pktable, points, coeffs)


@overload(pos_o, jit_options={'fastmath': True})
def _pos_o_overload(t, tpa, p, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _pos_ov(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs):
            return _pos_os(t, tpa, p, dt, pktable, points, coeffs)
        return impl
    return None
