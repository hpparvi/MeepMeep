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

"""Multi-knot light-travel-time correction evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, pi, ndarray

from .zposition import _zpos_os
from ..utils import mean_anomaly_at_transit
from ._common import _is_1d_array


# Time taken by light to traverse one solar radius, in days:
# (1 R_sun) / c = ((1 * u.R_sun).to(u.m) / c.c).to('d').value
LTT_DAYS_PER_RSUN = 2.685885891543453e-05


@njit(fastmath=True)
def _light_travel_time_os(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`light_travel_time_o`. See that function for documentation."""
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_t = _zpos_os(t, tpa, p, dt, pktable, points, coeffs)
    z_tr = _zpos_os(tpa + to, tpa, p, dt, pktable, points, coeffs)
    return -(z_t - z_tr) * rstar * LTT_DAYS_PER_RSUN


@njit(fastmath=True)
def _light_travel_time_ov(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Vector kernel for :func:`light_travel_time_o`. See that function for documentation."""
    n = times.size
    ltt = zeros(n)
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_tr = _zpos_os(tpa + to, tpa, p, dt, pktable, points, coeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    for j in range(n):
        ltt[j] = factor * (_zpos_os(times[j], tpa, p, dt, pktable, points, coeffs) - z_tr)
    return ltt


def light_travel_time_o(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction, referenced to primary transit.

    The correction is

        ltt(t) = -(z(t) - z(t_transit)) * rstar * (R_sun / c)

    with z in stellar radii, rstar in solar radii, result in days. The
    reference is the primary transit (inferior conjunction): ``ltt(t_transit)
    = 0`` by construction. This matches the convention used in transit
    fitting, where the observed mid-transit time is the reference and the LTT
    correction should add to the timing offset between primary transit and
    secondary eclipse (and intermediate phases), not to the transit itself.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_light_travel_time_os`) or vector
    (:func:`_light_travel_time_ov`) kernel at compile time (inside
    ``@njit``) or at call time (pure Python).

    Important: the convention in this module is that ``tc`` is the
    **periastron** time (the same as for every other ``*_o*`` evaluator in
    ``orbit3d``). The transit time is ``tc + to`` where
    ``to = mean_anomaly_at_transit(e, w) * p / (2*pi)``. The ``e, w`` arguments
    are needed to determine ``to``.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the correction.
    tpa : float
        Time of periastron passage.
    p : float
        Orbital period [days].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    rstar : float
        Stellar radius [R_sun].
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from ``solve3d_orbit`` / ``create_knots``.

    Returns
    -------
    ltt : float or ndarray
        Light travel time correction [days]. Arrays of shape (N,) for an
        array time argument.
    """
    if isinstance(t, ndarray):
        return _light_travel_time_ov(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs)
    return _light_travel_time_os(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs)


@overload(light_travel_time_o, jit_options={'fastmath': True})
def _light_travel_time_o_overload(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
            return _light_travel_time_ov(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
            return _light_travel_time_os(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs)
        return impl
    return None
