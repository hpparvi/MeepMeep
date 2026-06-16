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

"""Single-expansion-point stellar radial-velocity evaluators (from the planet's line-of-sight velocity)."""

from numba import njit
from numpy import pi, sqrt, sin
from numpy.typing import NDArray

from .zvelocity import zvel_c, zvel


@njit(inline='always')
def rv_c(time: float | NDArray, k: float, p: float, a: float, i: float, e: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the stellar radial velocity induced by the planet at an expansion-point-centered time.

    Converts the planet's centered line-of-sight velocity into the
    physical radial velocity of the host star, scaled by the
    user-supplied semi-amplitude `k`. The conversion follows
    Perryman (2018) Eq. 2.23.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    k : float
        Radial-velocity semi-amplitude of the star, in physical
        velocity units (e.g. m/s). The function output inherits these
        units.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis in units of stellar radii.
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read by the inner `zvel_c`.

    Returns
    -------
    rv : float or NDArray
        Stellar radial velocity in the same units as `k`. The sign
        convention matches the underlying z-axis: positive when the
        planet (and therefore the reflex motion of the star) is
        directed toward the observer.

    Notes
    -----
    The normalisation factor
    `n = (2*pi/p) * (a * sin(i)) / sqrt(1 - e^2)`
    has units of inverse time times stellar radii, exactly cancelling
    the units of the scaled `vz` so that `vz / n` is dimensionless and
    the final multiplication by `k` carries the physical units.
    """
    n = 2 * pi / p * (a * sin(i)) / sqrt(1 - e ** 2)  # Perryman (2018) Eq. 2.23
    return zvel_c(time, c) / n * k


@njit(inline='always')
def rv(time: float | NDArray, k: float, tc: float, p: float, a: float, i: float, e: float, c: NDArray,
       te: float = 0.0) -> float | NDArray:
    """
    Evaluate the stellar radial velocity induced by the planet at an absolute time.

    Direct counterpart of `rv_c`: epoch-folds the absolute time via
    `zvel` and applies the same Perryman (2018) Eq. 2.23 conversion
    to physical units.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tc` and `p`.
    k : float
        Radial-velocity semi-amplitude of the star, in physical
        velocity units (e.g. m/s). The function output inherits these
        units.
    tc : float
        Transit-centre time (time of inferior conjunction), on the same
        time axis as `time`.
    p : float
        Orbital period.
    a : float
        Scaled semi-major axis in units of stellar radii.
    i : float
        Orbital inclination in radians.
    e : float
        Orbital eccentricity.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        is read.
    te : float, optional
        Expansion-point offset from the transit centre [days] - the same value that
        was passed to `solve3d`. Defaults to 0.0, the expansion point at the
        transit centre.

    Returns
    -------
    rv : float or NDArray
        Stellar radial velocity in the same units as `k`. Positive
        when the planet is moving toward the observer (the star's
        reflex motion has the same sign convention).
    """
    n = 2 * pi / p * (a * sin(i)) / sqrt(1 - e ** 2)
    return zvel(time, tc, p, c, te) / n * k
