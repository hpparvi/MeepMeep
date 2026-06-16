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

"""Multi-expansion-point radial-velocity evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, pi, sin, sqrt, ndarray

from .zvelocity import _zvel_os
from ._common import _is_1d_array


@njit(fastmath=True, inline="always")
def _rv_os(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
    """Scalar kernel for :func:`rv_o`. See that function for documentation."""
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    return _zvel_os(t, tpa, p, dt, ep_table, ep_times, coeffs) * scale


@njit(fastmath=True)
def _rv_ov(times, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
    """Vector kernel for :func:`rv_o`. See that function for documentation."""
    n = times.size
    rvs = zeros(n)
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    for j in range(n):
        rvs[j] = _zvel_os(times[j], tpa, p, dt, ep_table, ep_times, coeffs) * scale
    return rvs


@njit(fastmath=True, parallel=True)
def _rv_ovp(times, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
    """Parallel (prange) twin of :func:`_rv_ov`."""
    n = times.size
    rvs = zeros(n)
    scale = k / (2 * pi / p * (a * sin(i)) / sqrt(1 - e * e))
    for j in prange(n):
        rvs[j] = _zvel_os(times[j], tpa, p, dt, ep_table, ep_times, coeffs) * scale
    return rvs


def rv_o(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
    """Radial velocity at an array of times (Perryman 2018, Eq. 2.23).

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_rv_os`) or vector (:func:`_rv_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Converts the internal line-of-sight velocity (in
    :math:`R_\\star/\\mathrm{day}`) to an observed radial velocity by
    multiplying with the closed-form scale factor
    :math:`K / [(2\\pi/p)(a\\sin i)/\\sqrt{1-e^2}]`.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the radial velocity.
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
    dt, ep_table, ep_times, coeffs :
        Multi-expansion-point dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.expansion_points.create_expansion_points`.

    Returns
    -------
    rv : float or ndarray
        Radial velocity [m s\\ :sup:`-1`]. Arrays of shape (N,) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _rv_ov(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs)
    return _rv_os(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs)


@overload(rv_o, jit_options={'fastmath': True})
def _rv_o_overload(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
    if _is_1d_array(t):
        def impl(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
            return _rv_ov(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
            return _rv_os(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs)
        return impl
    return None
