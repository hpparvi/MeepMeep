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
from numpy import pi, arctan2, sqrt, sin, cos, arccos, mod, copysign, sign, array, arcsin
from numpy.typing import NDArray
from scipy.constants import G

HALF_PI = 0.5*pi
TWO_PI = 2.0*pi


@njit(fastmath=True)
def eccentricity_vector(i, e, w):
    """
    Compute the 3D eccentricity vector in the observer's coordinate system.

    The eccentricity vector points toward periastron with a magnitude equal
    to the eccentricity. This function rotates that vector from the orbital
    plane into the observer's frame based on the inclination.

    Parameters
    ----------
    i : float
        Orbital inclination in radians. An inclination of pi/2 (90 degrees)
        corresponds to an edge-on orbit relative to the observer.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians. Defines the orientation of the
        ellipse within the orbital plane.

    Returns
    -------
    vec : NDArray
        A 1D float array of shape (3,) representing the [ex, ey, ez]
        components of the eccentricity vector. If e < 1e-5, returns
        [-1.0, 0.0, 0.0] as a stable reference for circular orbits.

    Notes
    -----
    The coordinate system is defined such that the z-axis points along the
    line of sight toward the observer. The components are calculated as:

    * ex = -e * cos(w)
    * ey = -e * sin(w) * cos(i)
    * ez =  e * sin(w) * sin(i)
    """
    if e > 1e-5:
        ci = cos(i)
        si = sin(i)
        ex = -e*cos(w)
        ey = -e*sin(w)*ci
        ez =  e*sin(w)*si
        return array([ex, ey, ez])
    else:
        return array([-1.0, 0.0, 0.0])


@njit
def eclipse_time_offset(p, i, e, w):
    """
        Calculate the time offset of the secondary eclipse relative to the primary transit.

        For eccentric orbits, the secondary eclipse does not occur at exactly 0.5 phase.
        This function computes the exact time offset using Keplerian dynamics,
        accounting for the non-uniform orbital velocity of the planet.

        Parameters
        ----------
        p : float
            Orbital period in units of time (e.g., days).
        i : float
            Orbital inclination in radians. Note: While 'i' is an input, it is
            not explicitly used in the calculation as the center of eclipse
            depends primarily on the longitudinal geometry (e, w).
        e : float
            Orbital eccentricity (0 <= e < 1).
        w : float
            Argument of periastron in radians.

        Returns
        -------
        offset : float
            The time elapsed between the primary transit center and the
            secondary eclipse center, in the same units as `p`.
            The result is bounded between [0, p].

        Notes
        -----
        The function solves Kepler's equation for both the transit and eclipse
        positions. The transit center is assumed to occur at a true anomaly of
        f = pi/2 - w.
    """
    etr = arctan2(sqrt(1. - e**2) * sin(HALF_PI - w), e + cos(HALF_PI - w))
    eec = arctan2(sqrt(1. - e**2) * sin(HALF_PI + pi - w), e + cos(HALF_PI + pi - w))
    mtr = etr - e * sin(etr)
    mec = eec - e * sin(eec)
    offset = (mec - mtr) * p / TWO_PI
    return offset if offset > 0. else p + offset


@njit(fastmath=True)
def transit_distance_factor(e, w):
    """
    Calculate the dimensionless distance factor at the time of primary transit.

    This represents the ratio of the planet-star separation at transit (r_tr)
    to the semi-major axis (a), specifically (r_tr / a). It accounts for
    the "foreshortening" or "stretching" of the transit duration caused
    by orbital eccentricity and orientation.

    Parameters
    ----------
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    factor : float
        The ratio (1 - e^2) / (1 + e * sin(w)). This is the transverse
        distance factor used in transit duration and velocity calculations.

    Notes
    -----
    In a circular orbit (e=0), this factor is exactly 1.0.
    In an eccentric orbit:
    * If w = pi/2 (periastron at transit), the factor is (1 - e), meaning
      the planet is closer to the star and moving faster.
    * If w = 3pi/2 (apoastron at transit), the factor is (1 + e), meaning
      the planet is further and moving slower.
    """
    return (1.0-e**2)/(1.0 + e*sin(w))


@njit
def i_from_baew(b, a, e, w):
    """
    Compute the orbital inclination from the impact parameter and orbital elements.

    This function inverts the standard relation for the impact parameter 'b'
    to find the required inclination 'i'. It accounts for the non-circular
    geometry of the orbit at the moment of transit.

    Parameters
    ----------
    b : float
        Impact parameter. The projected distance between the
        planet and star centers at the moment of transit, in units of the
        stellar radius.
    a : float
        Scaled semi-major axis (a/R_star). The semi-major axis expressed
        in units of the stellar radius.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    i : float
        Orbital inclination in radians.

    Notes
    -----
    The impact parameter is defined as:
    b = (a / R_star) * (r_tr / a) * cos(i)

    Where (r_tr / a) is the 'transit_distance_factor'. This function
    rearranges the formula to solve for i:
    i = arccos( b / (a * factor) )
    """
    return arccos(b / (a * transit_distance_factor(e, w)))


@njit(fastmath=True)
def as_from_rhop(rho, period):
    r"""
    Compute the scaled semi-major axis (a/R_star) from stellar density and orbital period.

    This calculation is derived from Kepler's Third Law, assuming the planet's
    mass is negligible compared to the stellar mass ($M_p \ll M_*$). It relates
    the geometry of the orbit directly to the physical properties of the star.

    Parameters
    ----------
    rho : float
        Mean stellar density in g/cm^3.
    period : float
        Orbital period in days.

    Returns
    -------
    as_scaled : float
        The scaled semi-major axis (a/R_star), representing the distance
        in units of the stellar radius.

    Notes
    -----
    The relationship is based on the following form of Kepler's Third Law:
    $$ \frac{a}{R_star} = \left( \frac{G \cdot \rho \cdot P^2}{3\pi} \right)^{1/3} $$

    The constant 86400.0 converts days to seconds, and 1e3 is used to
    convert the density from g/cm^3 to the kg/m^3 required for SI units
    if $G$ is in $m^3 kg^{-1} s^{-2}$.
    """
    return (G/(3*pi))**(1/3) * ((period * 86400.0)**2 * 1e3 * rho)**(1 / 3)


@njit
def ta_from_ea(e, ecc):
    r"""
    Convert Eccentric Anomaly to True Anomaly.

    This function calculates the position of the body along its orbit (True
    Anomaly) given its position relative to the auxiliary circle (Eccentric
    Anomaly).

    Parameters
    ----------
    e : float or NDArray
        Eccentric Anomaly in radians.
    ecc : float
        Orbital eccentricity (0 <= e < 1).

    Returns
    -------
    f : float or NDArray
        True Anomaly in radians.

    Notes
    -----
    The relationship is derived from the geometry of the ellipse:
    $$ \cos(f) = \frac{\cos(E) - e}{1 - e \cos(E)} $$
    $$ \sin(f) = \frac{\sqrt{1 - e^2} \sin(E)}{1 - e \cos(E)} $$
    Using arctan2 ensures the True Anomaly is placed in the correct quadrant.
    """
    sta = sqrt(1.0 - ecc ** 2) * sin(e) / (1.0 - ecc * cos(e))
    cta = (cos(e) - ecc) / (1.0 - ecc * cos(e))
    return arctan2(sta, cta)


@njit
def mean_anomaly_at_transit(ecc, w):
    r"""
    Compute the Mean Anomaly at the moment of primary transit.

    For an eccentric orbit, the transit center does not occur at a Mean
    Anomaly of zero. This function calculates the angular time-offset required
    to align the transit event with the orbital clock.

    Parameters
    ----------
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    m_at_transit : float
        The Mean Anomaly at transit center in radians.

    Notes
    -----
    The function first finds the Eccentric Anomaly (E) by relating the
    geometry of the transit (where true anomaly f = pi/2 - w) to the
    eccentricity vector. It then solves Kepler's Equation:
    $$ M = E - e \sin(E) $$
    """
    m_at_transit = arctan2(sqrt(1.0 - ecc ** 2) * sin(HALF_PI - w), ecc + cos(HALF_PI - w))
    m_at_transit -= ecc * sin(m_at_transit)
    return m_at_transit


@njit(fastmath=True)
def mean_anomaly_at_transit_with_derivatives(ecc, w):
    """
    Compute the Mean Anomaly at transit and its derivatives w.r.t. e and w.

    Parameters
    ----------
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    m_tr : float
        Mean Anomaly at the moment of primary transit in radians.
    dm_tr_de : float
        Partial derivative of the Mean Anomaly w.r.t. eccentricity.
    dm_tr_dw : float
        Partial derivative of the Mean Anomaly w.r.t. argument of periastron.
    """
    sqe2 = sqrt(1.0 - ecc ** 2)
    cw = cos(w)
    sw = sin(w)
    y_e = sqe2 * cw
    x_e = ecc + sw
    e_off = arctan2(y_e, x_e)
    se = sin(e_off)
    ce = cos(e_off)
    m_at_transit = e_off - ecc * se
    denom = x_e**2 + y_e**2
    de_off_de = (x_e * (-ecc / sqe2) * cw - y_e * 1.0) / denom
    de_off_dw = (x_e * (-sqe2 * sw) - y_e * cw) / denom
    dm_tr_de = de_off_de * (1.0 - ecc * ce) - se
    dm_tr_dw = de_off_dw * (1.0 - ecc * ce)
    return m_at_transit, dm_tr_de, dm_tr_dw


@njit(fastmath=True)
def tc_to_tp_gradient(dc, p, e, w):
    """Reparametrise a transit-centre-basis gradient block into the periastron basis.

    Converts a gradient whose leading axis is ordered ``(tc, p, a, i, e, w, lan)``
    (the transit-centre parametrisation produced by ``solve2d_d`` / ``solve3d_d``)
    into the periastron parametrisation ``(tp, p, a, i, e, w, lan)``, in which the
    shape derivatives are taken holding the time of periastron passage fixed.

    The two conventions are related by ``tc = tp + M_tr(e, w) * p / (2 * pi)``, so
    the chain rule adds multiples of the timing row (index 0) to the p, e, and w
    rows (indices 1, 4, 5)::

        out[1] = dc[1] + dc[0] * (M_tr / (2 pi))
        out[4] = dc[4] + dc[0] * (dM_tr/de * p / (2 pi))
        out[5] = dc[5] + dc[0] * (dM_tr/dw * p / (2 pi))

    Rows 0 (timing, now d/dtp), 2 (a), 3 (i), and 6 (lan) are unchanged.

    Parameters
    ----------
    dc : ndarray
        Gradient block with the parameter axis first, shape ``(7, ...)``. The
        trailing dimensions are arbitrary (e.g. ``(7, 3, 5)`` for a single 3D
        knot or ``(7, 2, 5)`` for a 2D knot).
    p : float
        Orbital period [days].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [rad].

    Returns
    -------
    ndarray
        A new array of the same shape as ``dc``, in the periastron
        parametrisation. The input is not modified.
    """
    m_tr, dm_tr_de, dm_tr_dw = mean_anomaly_at_transit_with_derivatives(e, w)
    c = 1.0 / TWO_PI
    out = dc.copy()
    out[1] = dc[1] + dc[0] * (m_tr * c)
    out[4] = dc[4] + dc[0] * (dm_tr_de * p * c)
    out[5] = dc[5] + dc[0] * (dm_tr_dw * p * c)
    return out


@njit
def mean_anomaly(t, tc, p, e, w):
    """
    Calculate the Mean Anomaly at time t, bounded between [0, 2pi].

    Parameters
    ----------
    t : float
        Observation time.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period in the same units as t.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    m : float
        Mean Anomaly in radians, wrapped to the interval [0, 2pi].
    """
    offset = mean_anomaly_at_transit(e, w)
    return mod(TWO_PI * (t - (tc - offset * p / TWO_PI)) / p, TWO_PI)


@njit(fastmath=True)
def mean_anomaly_with_derivatives(t, tc, p, ecc, w):
    """
    Calculate the Mean Anomaly and its partial derivatives w.r.t. tc, p, e, and w.

    Parameters
    ----------
    t : float
        Observation time.
    tc : float
        Time of primary transit center.
    p : float
        Orbital period in the same units as t.
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    m : float
        Mean Anomaly in radians.
    dm_dtc : float
        Partial derivative of m w.r.t. transit center time.
    dm_dp : float
        Partial derivative of m w.r.t. orbital period.
    dm_de : float
        Partial derivative of m w.r.t. eccentricity.
    dm_dw : float
        Partial derivative of m w.r.t. argument of periastron.
    """
    m_tr, dm_tr_de, dm_tr_dw = mean_anomaly_at_transit_with_derivatives(ecc, w)
    dt = t - tc
    mean_motion = TWO_PI / p
    m = mean_motion * dt + m_tr
    dm_dtc = -mean_motion
    dm_dp  = -TWO_PI * dt / (p**2)
    dm_de  = dm_tr_de
    dm_dw  = dm_tr_dw
    return m, dm_dtc, dm_dp, dm_de, dm_dw


@njit
def z_from_ta(f, a, i, e, w):
    """
    Compute the sky-projected separation.

    Parameters
    ----------
    f : float or NDArray
        True anomaly in radians.
    a : float
        Scaled semi-major axis (a/R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.

    Returns
    -------
    z : float or NDArray
        Projected separation between the planet and star centers in units
        of stellar radii. Positive values indicate the planet is in
        front of the star (transit), negative values indicate eclipse.
    """
    z  = a * (1.0-e**2) / (1.0 + e * cos(f)) * sqrt(1.0 - sin(w + f) ** 2 * sin(i) ** 2)
    z *= copysign(1.0, sin(w + f))
    return z


@njit
def impact_parameter(a, i):
    """
    Calculate the impact parameter for a circular orbit.

    Parameters
    ----------
    a : float
        Scaled semi-major axis (a/R_star).
    i : float
        Orbital inclination in radians.

    Returns
    -------
    b : float
        Impact parameter (separation at transit center) in units of R_star.
    """
    return a * cos(i)


@njit
def impact_parameter_ec(a, i, e, w, tr_sign):
    """
    Calculate the impact parameter for an eccentric orbit.

    Parameters
    ----------
    a : float
        Scaled semi-major axis (a/R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.
    tr_sign : float
        Sign of the event: 1.0 for primary transit, -1.0 for secondary eclipse.

    Returns
    -------
    b : float
        Impact parameter in units of R_star, corrected for eccentricity.
    """
    return a * cos(i) * ((1.-e**2) / (1.+tr_sign*e*sin(w)))

@njit
def d_from_pkaiews(p, k, a, i, e, w, tr_sign, kind=14):
    """
    Calculate the transit/eclipse duration (T14 or T23).

    Parameters
    ----------
    p : float
        Orbital period in days.
    k : float
        Radius ratio (R_planet / R_star).
    a : float
        Scaled semi-major axis (a / R_star).
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity (0 <= e < 1).
    w : float
        Argument of periastron in radians.
    tr_sign : float
        Sign of the event: 1.0 for transit, -1.0 for eclipse.
    kind : int, optional
        14 for full duration (first to fourth contact),
        23 for total duration (second to third contact). Default is 14.

    Returns
    -------
    d : float
        Duration in days.
    """
    b  = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = sqrt(1.-e**2)/(1.+tr_sign*e*sin(w))
    ds = 1. if kind == 14 else -1.
    return p/pi  * arcsin(sqrt((1.+ds*k)**2-b**2)/(a*sin(i))) * ae