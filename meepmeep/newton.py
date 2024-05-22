from numba import njit
from numpy import cos, sin, zeros

from .utils import mean_anomaly, ta_from_ea_s, ta_from_ea_v, z_from_ta_s, z_from_ta_v, eclipse_phase


@njit
def ea_newton_s(t, t0, p, e, w):
    ma = mean_anomaly(t, t0, p, e, w)
    ea = ma
    err = 0.05
    k = 0
    while abs(err) > 1e-8 and k < 1000:
        err = ea - e*sin(ea) - ma
        ea = ea - err/(1.0-e*cos(ea))
        k += 1
    return ea


@njit
def ea_newton_v(t, t0, p, e, w):
    ea = zeros(t.size)
    for i in range(len(t)):
        ea[i] = ea_newton_s(t[i], t0, p, e, w)
    return ea


@njit
def ta_newton_s(t, t0, p, e, w):
    return ta_from_ea_s(ea_newton_s(t, t0, p, e, w), e)


@njit
def ta_newton_v(t, t0, p, e, w):
    return ta_from_ea_v(ea_newton_v(t, t0, p, e, w), e)


@njit(fastmath=True)
def xy_newton_v(time, t0, p, a, i, e, w):
    """Planet velocity and acceleration at mid-transit in [R_star / day]"""
    f = ta_newton_v(time, t0, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    return x, y


@njit(fastmath=True)
def xyz_newton_v(time, t0, p, a, i, e, w):
    """Planet velocity and acceleration at mid-transit in [R_star / day]"""
    f = ta_newton_v(time, t0, p, e, w)
    r = a * (1. - e ** 2) / (1. + e * cos(f))
    x = -r * cos(w + f)
    y = -r * sin(w + f) * cos(i)
    z =  r * sin(w + f) * sin(i)
    return x, y, z


@njit
def z_newton_s(time, t0, p, a, i, e, w):
    """Normalized projected distance for scalar time.
    """
    ta = ta_newton_s(time, t0, p, e, w)
    return z_from_ta_s(ta, a, i, e, w)


@njit
def z_newton_v(time, t0, p, a, i, e, w):
    """Normalized projected distance for an array of times.
    """
    ta = ta_newton_v(time, t0, p, e, w)
    return z_from_ta_v(ta, a, i, e, w)


@njit
def rv_newton_v(times, k, t0, p, e, w):
    ta_n = ta_newton_v(times, t0, p, e, w)
    return k * (cos(w + ta_n) + e * cos(w))


@njit
def eclipse_light_travel_time(p: float, a: float, i: float, e: float, w: float, rstar: float):
    """
    Calculate the light travel time difference between the transit and the secondary eclipse of an exoplanet.

    This function computes the difference in light travel time caused by the displacement of the planet between
    its transit across the star and its secondary eclipse (when the planet passes behind the star as viewed from Earth).

    Parameters
    ----------
    p : float
        Orbital period in days.
    a : float
        Semi-major axis of the planet's orbit in host star radii.
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity.
    w : float
        Argument of periastron in radians.
    rstar : float
        Radius of the star in solar radii.

    Returns
    -------
    float
        The light travel time difference in days between the transit and secondary eclipse of the exoplanet.
    """
    s = 2.685885891543453e-05  # Light travel time for a distance of one solar radius in days

    ae = a * (1. - e ** 2)
    si = sin(i)

    f = ta_newton_s(0.0, 0.0, p, e, w)
    r = ae / (1. + e * cos(f))
    ztr = r * sin(w + f) * si

    f = ta_newton_s(eclipse_phase(p, i, e, w), 0.0, p, e, w)
    r = ae / (1. + e * cos(f))
    zec = r * sin(w + f) * si

    return (ztr - zec) * rstar * s