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

from numba import njit
from numpy import cos, sin, zeros, pi
from numpy.typing import NDArray

from ..utils import mean_anomaly, ta_from_ea, z_from_ta, eclipse_time_offset


@njit(fastmath=True)
def ea_from_ma(ma, ecc):
    """Solve Kepler's equation for the eccentric anomaly.

    Inverts ``E - e * sin(E) = M`` using Newton-Raphson iteration.

    Parameters
    ----------
    ma : float or NDArray
        Mean anomaly in radians.
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).

    Returns
    -------
    ea : float or NDArray
        Eccentric anomaly in radians satisfying ``E - e * sin(E) = M`` to
        within ``|dE| < 1e-13`` or after 50 iterations, whichever comes first.

    Notes
    -----
    The initial guess is ``E_0 = M`` for moderate eccentricities and
    ``E_0 = pi`` for ``e > 0.8`` to improve convergence near apoastron
    where Kepler's equation becomes stiff.
    """
    ea = ma
    if ecc > 0.8:
        ea = pi

    for _ in range(50):
        f = ea - ecc * sin(ea) - ma
        df = 1.0 - ecc * cos(ea)
        dea = -f / df
        ea += dea
        if abs(dea) < 1e-13:
            break
    return ea


@njit
def ea_newton_s(t, tc, p, e, w):
    """Eccentric anomaly at a scalar time via Newton-Raphson.

    Computes the mean anomaly from the orbital elements at time ``t``,
    then solves Kepler's equation by iteration.

    Parameters
    ----------
    t : float
        Observation time, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    ea : float
        Eccentric anomaly in radians.

    """
    ma = mean_anomaly(t, tc, p, e, w)
    return ea_from_ma(ma, e)


@njit
def ea_newton_v(t, tc, p, e, w):
    """Eccentric anomaly evaluated at an array of times.

    Parameters
    ----------
    t : NDArray
        1D array of observation times, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    ea : NDArray
        Eccentric anomaly in radians.
    """
    ea = zeros(t.size)
    for i in range(len(t)):
        ma = mean_anomaly(t[i], tc, p, e, w)
        ea[i] = ea_from_ma(ma, e)
    return ea


@njit
def ta_newton_s(t, tc, p, e, w):
    """True anomaly at a scalar time.

    Composes `ea_newton_s` with `ta_from_ea` to map time directly to
    true anomaly, the angle between periastron and the planet's
    current position as seen from the focus.

    Parameters
    ----------
    t : float
        Observation time, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    f : float
        True anomaly in radians, wrapped to ``(-pi, pi]`` by arctan2.
    """
    return ta_from_ea(ea_newton_s(t, tc, p, e, w), e)


@njit
def ta_newton_v(t, tc, p, e, w):
    """True anomaly evaluated at an array of times.

    Parameters
    ----------
    t : NDArray
        1D array of observation times, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    f : NDArray
        True anomaly in radians for each input time.
    """
    return ta_from_ea(ea_newton_v(t, tc, p, e, w), e)


@njit(fastmath=True)
def xy_newton_v(time, tc, p, a, i, e, w):
    """Sky-plane (x, y) position of the planet at an array of times.

    Parameters
    ----------
    time : NDArray
        1D array of observation times, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis (a / R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    x : NDArray
        Sky-plane x coordinate in units of stellar radii.
    y : NDArray
        Sky-plane y coordinate in units of stellar radii.

    Notes
    -----
    The x-axis points right and the y-axis up in the projected sky
    plane (see the coordinate-system convention in the project
    documentation). The radial distance is ``r = a(1-e^2)/(1+e cos f)``
    and the projection drops the line-of-sight component.
    """
    f = ta_newton_v(time, tc, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    return x, y


@njit(fastmath=True)
def xyz_newton_v(time, tc, p, a, i, e, w):
    """3D position of the planet at an array of times.

    Returns the full position vector including the line-of-sight
    component, so callers can distinguish transit (z > 0) from
    secondary eclipse (z < 0).

    Parameters
    ----------
    time : NDArray
        1D array of observation times, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis (a / R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    x : NDArray
        Sky-plane x coordinate in units of stellar radii.
    y : NDArray
        Sky-plane y coordinate in units of stellar radii.
    z : NDArray
        Line-of-sight coordinate in units of stellar radii; positive
        toward the observer (transit side), negative on the far side
        of the orbit (eclipse side).
    """
    f = ta_newton_v(time, tc, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    z =  r * sin(w + f) * sin(i)
    return x, y, z


@njit
def z_newton_s(time, tc, p, a, i, e, w):
    """Signed sky-projected planet-star separation at a scalar time.

    Parameters
    ----------
    time : float
        Observation time, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis (a / R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    z : float
        Projected planet-star separation in units of stellar radii.
        Positive during transit (planet in front of the star),
        negative during secondary eclipse (planet behind the star).
    """
    ta = ta_newton_s(time, tc, p, e, w)
    return z_from_ta(ta, a, i, e, w)


@njit
def z_newton_v(time, tc, p, a, i, e, w):
    """Signed sky-projected planet-star separation at an array of times.

    Parameters
    ----------
    time : NDArray
        1D array of observation times, in the same units as ``p``.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis (a / R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    z : NDArray
        Projected separation in stellar radii per input time, signed by
        transit/eclipse side as in `z_newton_s`.
    """
    ta = ta_newton_v(time, tc, p, e, w)
    return z_from_ta(ta, a, i, e, w)


@njit
def rv_newton_v(times, k, tc, p, e, w):
    """Stellar radial velocity induced by a planet at an array of times.

    Implements the standard Keplerian radial-velocity model
    ``RV(t) = K * (cos(w + f) + e * cos(w))``,
    where ``f`` is the true anomaly and ``K`` is the velocity
    semi-amplitude. The constant offset ``-K * e * cos(w)`` ensures
    that the systemic velocity is zero (i.e. the model returns the
    reflex motion around the barycenter).

    Parameters
    ----------
    times : NDArray
        1D array of observation times, in the same units as ``p``.
    k : float
        Radial-velocity semi-amplitude in the desired velocity units
        (e.g. m/s).
    tc : float
        Time of primary transit center.
    p : float
        Orbital period.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    rv : NDArray
        Predicted radial velocity at each input time, in the same
        units as ``k``.
    """
    ta_n = ta_newton_v(times, tc, p, e, w)
    return k * (cos(w + ta_n) + e * cos(w))


@njit
def eclipse_light_travel_time(p: float, a: float, i: float, e: float, w: float, rstar: float):
    """Light travel time difference between primary transit and secondary eclipse.

    Photons from a secondary eclipse traverse an extra path equal to
    the line-of-sight displacement of the planet between the two
    events. This function returns that delay so observed
    transit/eclipse mid-times can be corrected to the same emission
    frame.

    Parameters
    ----------
    p : float
        Orbital period in days.
    a : float
        Scaled semi-major axis (a / R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.
    rstar : float
        Stellar radius in solar radii.

    Returns
    -------
    dt : float
        Light travel time delay between primary transit and secondary
        eclipse, in days. Positive when the planet is closer to the
        observer at transit than at eclipse (the usual case).

    Notes
    -----
    The line-of-sight coordinate is evaluated as
    ``z = r * sin(w + f) * sin(i)`` at the true anomaly for primary
    transit (``t = 0``) and for secondary eclipse (``t =
    eclipse_time_offset(p, i, e, w)``). The difference is converted
    to a time delay using the light-crossing time of one solar
    radius, ``s = R_sun / c ~ 2.686e-5 d``.
    """
    s = 2.685885891543453e-05  # Light travel time for a distance of one solar radius in days

    ae = a * (1. - e ** 2)
    si = sin(i)

    f = ta_newton_s(0.0, 0.0, p, e, w)
    r = ae / (1. + e * cos(f))
    ztr = r * sin(w + f) * si

    f = ta_newton_s(eclipse_time_offset(p, i, e, w), 0.0, p, e, w)
    r = ae / (1. + e * cos(f))
    zec = r * sin(w + f) * si

    return (ztr - zec) * rstar * s