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
from ...utils import mean_anomaly_at_transit
from ._common import _is_1d_array


# Time taken by light to traverse one solar radius, in days:
# (1 R_sun) / c = ((1 * u.R_sun).to(u.m) / c.c).to('d').value
LTT_DAYS_PER_RSUN = 2.685885891543453e-05


@njit(fastmath=True)
def _light_travel_time_os(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction at a scalar time, referenced to primary transit.

    The correction is

        ltt(t) = -(z(t) - z(t_transit)) · rstar · (R_sun / c)

    with z in stellar radii, rstar in solar radii, result in days. The
    reference is the primary transit (inferior conjunction): ``ltt(t_transit)
    = 0`` by construction. This matches the convention used in transit
    fitting, where the observed mid-transit time is the reference and the LTT
    correction should add to the timing offset between primary transit and
    secondary eclipse (and intermediate phases), not to the transit itself.

    Important: the convention in this module is that ``t0`` is the
    **periastron** time (the same as for every other ``*_o*`` evaluator in
    ``orbit3d``). The transit time is ``t0 + to`` where
    ``to = mean_anomaly_at_transit(e, w) · p / (2π)``. The ``e, w`` arguments
    are needed to determine ``to``.

    Parameters
    ----------
    t : float
        Time at which to evaluate the correction.
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
    ltt : float
        Light travel time correction [days].
    """
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_t = _zpos_os(t, tpa, p, dt, pktable, points, coeffs)
    z_tr = _zpos_os(tpa + to, tpa, p, dt, pktable, points, coeffs)
    return -(z_t - z_tr) * rstar * LTT_DAYS_PER_RSUN


@njit(fastmath=True)
def _light_travel_time_ov(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction at an array of times, referenced to primary transit.

    Vectorised version of :func:`_light_travel_time_os`. Caches the
    transit-time z-coordinate ``z_tr`` once outside the loop and reuses
    it for every input time.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the correction.
    tpa : float
        Periastron time.
    p : float
        Orbital period [days].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    rstar : float
        Stellar radius [R_sun].
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    ltt : ndarray, shape (N,)
        Light travel time correction at each input time [days].
    """
    n = times.size
    ltt = zeros(n)
    to = mean_anomaly_at_transit(e, w) / (2.0 * pi) * p
    z_tr = _zpos_os(tpa + to, tpa, p, dt, pktable, points, coeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    for j in range(n):
        ltt[j] = factor * (_zpos_os(times[j], tpa, p, dt, pktable, points, coeffs) - z_tr)
    return ltt


def light_travel_time_o(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs):
    """Light travel time correction.

    See :func:`_light_travel_time_os` / :func:`_light_travel_time_ov`.
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
