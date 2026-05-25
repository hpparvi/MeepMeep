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

"""Multi-knot radial-velocity evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, pi, sin, sqrt, ndarray

from .zvelocity import _zvel_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _rv_os(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
    """Radial velocity at scalar time (Perryman 2018, Eq. 2.23).

    Scalar counterpart of :func:`_rv_ov`.

    Parameters
    ----------
    t : float
        Time at which to evaluate the radial velocity.
    k : float
        Radial-velocity semi-amplitude [m s\\ :sup:`-1`].
    tpa : float
        Periastron time.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    rv : float
        Radial velocity [m s\\ :sup:`-1`].
    """
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    return _zvel_os(t, tpa, p, dt, pktable, points, coeffs) * scale


@njit(fastmath=True)
def _rv_ov(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
    """Radial velocity at an array of times (Perryman 2018, Eq. 2.23).

    Converts the internal line-of-sight velocity (in
    :math:`R_\\star/\\mathrm{day}`) to an observed radial velocity by
    multiplying with the closed-form scale factor
    :math:`K / [(2\\pi/p)(a\\sin i)/\\sqrt{1-e^2}]`.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the radial velocity.
    k : float
        Radial-velocity semi-amplitude [m s\\ :sup:`-1`].
    tpa : float
        Periastron time.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    rvs : ndarray, shape (N,)
        Radial velocity at each input time [m s\\ :sup:`-1`].
    """
    n = times.size
    rvs = zeros(n)
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    for j in range(n):
        rvs[j] = _zvel_os(times[j], tpa, p, dt, pktable, points, coeffs) * scale
    return rvs


def rv_o(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
    """Radial velocity. See :func:`_rv_os` / :func:`_rv_ov`."""
    if isinstance(t, ndarray):
        return _rv_ov(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs)
    return _rv_os(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs)


@overload(rv_o, jit_options={'fastmath': True})
def _rv_o_overload(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
            return _rv_ov(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs):
            return _rv_os(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs)
        return impl
    return None
