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
from numpy import floor, pi, sqrt, sin
from numpy.typing import NDArray


@njit(fastmath=True, inline='always')
def vel_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (vx, vy, vz) velocity at a knot-centered time.

    Centered velocity companion to `position3d.pos_c`. Each velocity
    component is obtained by analytically differentiating the
    corresponding 5th-order position polynomial; the resulting
    polynomial is 4th-order in `time` and is evaluated using Horner's
    scheme.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point, i.e.
        `time = tc - (tk + epoch*p)`. Must lie within the knot's
        region of validity for the truncation error to remain small.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Row 0 holds
        the x-direction coefficients, row 1 the y-direction, and row 2
        the z-direction, ordered as
        [position, velocity, acceleration/2, jerk/6, snap/24]
        (i.e. pre-scaled by the factorial of the Taylor order).

    Returns
    -------
    vx : float or NDArray
        Sky-plane x velocity in stellar radii per unit time.
    vy : float or NDArray
        Sky-plane y velocity in stellar radii per unit time.
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.

    Notes
    -----
    The pre-factors `1, 2, 3, 4` in front of `c[d, 1..4]` are the
    chain-rule factors from differentiating `c[d, n] * time^n` with
    respect to `time`. Because the polynomial loses one order under
    differentiation, the velocity is a 4th-order Taylor approximation
    even though the underlying position expansion is 5th order.
    """
    vx = c[0, 1] + time * (2.0 * c[0, 2] + time * (3.0 * c[0, 3] + time * 4.0 * c[0, 4]))
    vy = c[1, 1] + time * (2.0 * c[1, 2] + time * (3.0 * c[1, 3] + time * 4.0 * c[1, 4]))
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))
    return vx, vy, vz


@njit(fastmath=True, inline='always')
def zvel_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight velocity component at a knot-centered time.

    Centered counterpart of `zvel`. Only the z-direction coefficients
    (row 2 of `c`) are read, making this the cheapest velocity
    evaluator in the module. The polynomial is the analytic derivative
    of the position z-polynomial, 4th-order in `time`.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.

    Notes
    -----
    Useful for radial-velocity computations, where only the
    line-of-sight component matters; see `rv_c` / `rv`.
    """
    return c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))


@njit(fastmath=True, inline='always')
def zvel(time: float | NDArray, tk: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight velocity component at an absolute time.

    Direct counterpart of `zvel_c`: accepts an absolute observation
    time `time`, folds it back into a single orbital epoch around the
    expansion point `tk`, and delegates to `zvel_c`.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tk` and `p`.
    tk : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period, used for epoch folding.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return zvel_c(time - (tk + epoch * p), c)


@njit(inline='always')
def rv_c(time: float | NDArray, k: float, p: float, a: float, i: float, e: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the stellar radial velocity induced by the planet at a knot-centered time.

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
def rv(time: float | NDArray, k: float, tk: float, p: float, a: float, i: float, e: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the stellar radial velocity induced by the planet at an absolute time.

    Direct counterpart of `rv_c`: epoch-folds the absolute time via
    `zvel` and applies the same Perryman (2018) Eq. 2.23 conversion
    to physical units.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `tk` and `p`.
    k : float
        Radial-velocity semi-amplitude of the star, in physical
        velocity units (e.g. m/s). The function output inherits these
        units.
    tk : float
        Taylor series expansion time (knot time).
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

    Returns
    -------
    rv : float or NDArray
        Stellar radial velocity in the same units as `k`. Positive
        when the planet is moving toward the observer (the star's
        reflex motion has the same sign convention).
    """
    n = 2 * pi / p * (a * sin(i)) / sqrt(1 - e ** 2)
    return zvel(time, tk, p, c) / n * k
