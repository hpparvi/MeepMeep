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
from numpy import floor, sqrt, sin, cos, pi, zeros
from numpy.typing import NDArray


@njit(fastmath=True)
def vel_cd(time: float | NDArray, c: NDArray, dc: NDArray):
    """
    Evaluate the (vx, vy, vz) velocity and its orbital-parameter derivatives at a knot-centered time.

    Centered velocity companion to `position3dd.pos_cd`. The velocity
    components are obtained by analytically differentiating the
    5th-order position polynomials, yielding 4th-order polynomials in
    `time` that are evaluated using Horner's scheme. The same
    differentiation is applied to the parameter-derivative
    coefficients so the result is the velocity together with its seven
    partial derivatives with respect to `(tc, p, a, i, e, w, lan)`.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Rows index
        the spatial dimensions (x, y, z) and columns the Taylor order
        from position through snap (pre-scaled by the factorial of the
        order).
    dc : NDArray
        A (7, 3, 5) tensor of parameter-derivative coefficients
        produced by `solve3d_d`. The leading axis enumerates the seven
        Keplerian parameters in the canonical order `(tc, p, a, i, e, w, lan)`;
        the remaining axes mirror the layout of `c`.

    Returns
    -------
    vx : float or NDArray
        Sky-plane x velocity in stellar radii per unit time.
    vy : float or NDArray
        Sky-plane y velocity in stellar radii per unit time.
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    dvx : NDArray
        Shape (7,) partial derivatives of `vx` w.r.t. `(tc, p, a, i, e, w, lan)`.
    dvy : NDArray
        Shape (7,) partial derivatives of `vy` w.r.t. `(tc, p, a, i, e, w, lan)`.
    dvz : NDArray
        Shape (7,) partial derivatives of `vz` w.r.t. `(tc, p, a, i, e, w, lan)`.

    Notes
    -----
    The pre-factors `1, 2, 3, 4` in front of `c[d, 1..4]` and
    `dc[k, d, 1..4]` are the chain-rule factors from differentiating
    `c[d, n] * time^n` (and likewise the derivative coefficients) with
    respect to `time`. Differentiation drops the truncation order by
    one, so the velocity polynomials are 4th order even though the
    underlying position expansion is 5th order.
    """
    vx = c[0, 1] + time * (2.0 * c[0, 2] + time * (3.0 * c[0, 3] + time * 4.0 * c[0, 4]))
    vy = c[1, 1] + time * (2.0 * c[1, 2] + time * (3.0 * c[1, 3] + time * 4.0 * c[1, 4]))
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))

    dvx = zeros(7)
    dvy = zeros(7)
    dvz = zeros(7)
    for k in range(7):
        dvx[k] = dc[k, 0, 1] + time * (2.0 * dc[k, 0, 2] + time * (3.0 * dc[k, 0, 3] + time * 4.0 * dc[k, 0, 4]))
        dvy[k] = dc[k, 1, 1] + time * (2.0 * dc[k, 1, 2] + time * (3.0 * dc[k, 1, 3] + time * 4.0 * dc[k, 1, 4]))
        dvz[k] = dc[k, 2, 1] + time * (2.0 * dc[k, 2, 2] + time * (3.0 * dc[k, 2, 3] + time * 4.0 * dc[k, 2, 4]))

    return vx, vy, vz, dvx, dvy, dvz


@njit(fastmath=True)
def zvel_cd(time: float | NDArray, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight velocity and its parameter derivatives at a knot-centered time.

    Centered companion to `velocity3d.zvel_c` that additionally
    returns the partial derivatives of the line-of-sight velocity
    with respect to each of the seven orbital parameters. Only the
    z-direction polynomials are evaluated; the x and y rows of `c`
    and `dc` are not read.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`, with the leading axis ordered as
        `(tc, p, a, i, e, w, lan)`. Only the slice `dc[:, 2, :]` is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    dvz : NDArray
        Shape (7,) partial derivatives of `vz` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))
    dvz = zeros(7)
    for k in range(7):
        dvz[k] = dc[k, 2, 1] + time * (2.0 * dc[k, 2, 2] + time * (3.0 * dc[k, 2, 3] + time * 4.0 * dc[k, 2, 4]))
    return vz, dvz


@njit(fastmath=True)
def zvel_d(time: float | NDArray, tk: float, p: float, c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the line-of-sight velocity and its parameter derivatives at an absolute time.

    Direct counterpart of `zvel_cd`: epoch-folds the absolute time
    `time` around the expansion point `tk` and delegates to
    `zvel_cd`.

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
        is read.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`. Only the slice `dc[:, 2, :]` is read.

    Returns
    -------
    vz : float or NDArray
        Line-of-sight z velocity in stellar radii per unit time.
        Positive values indicate motion toward the observer.
    dvz : NDArray
        Shape (7,) partial derivatives of `vz` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return zvel_cd(time - (tk + epoch * p), c, dc)


@njit(fastmath=True)
def rv_cd(time: float | NDArray, k: float, p: float, a: float, i: float, e: float,
          c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the stellar radial velocity and its parameter derivatives at a knot-centered time.

    Converts the planet's centered line-of-sight velocity into the
    physical radial velocity of the host star, scaled by the
    semi-amplitude `k`, following Perryman (2018) Eq. 2.23. The same
    chain rule is propagated to give the seven partial derivatives of
    the radial velocity with respect to the orbital parameters.

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
        is read by the inner `zvel_cd`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`, with the leading axis ordered as
        `(tc, p, a, i, e, w, lan)`.

    Returns
    -------
    rv : float or NDArray
        Stellar radial velocity in the same units as `k`. Positive
        when the planet is moving toward the observer.
    drv : NDArray
        Shape (7,) partial derivatives of `rv` with respect to
        `(tc, p, a, i, e, w, lan)`.

    Notes
    -----
    Let `s = k / n` with `n = (2*pi/p) * (a*sin(i)) / sqrt(1 - e^2)`.
    Then `rv = s * vz`, and the chain rule gives
    `d(rv)/dtheta = s * d(vz)/dtheta + vz * ds/dtheta`. The factor `s`
    depends only on `(p, a, i, e)`; its derivatives w.r.t. `tc` and
    `w` are zero. The non-trivial derivatives are
    `ds/dp = s/p`, `ds/da = -s/a`, `ds/di = -s*cot(i)`, and
    `ds/de = -s*e/(1 - e^2)`.
    """
    n = 2.0 * pi / p * (a * sin(i)) / sqrt(1.0 - e ** 2)
    s = k / n

    vz, dvz = zvel_cd(time, c, dc)
    rv_val = s * vz

    # ds/dtheta for each parameter: tc, p, a, i, e, w, lan
    drv = zeros(7)
    ds = zeros(7)
    ds[1] = s / p       # ds/dp
    ds[2] = -s / a      # ds/da
    ds[3] = -s * cos(i) / sin(i)  # ds/di
    ds[4] = -s * e / (1.0 - e ** 2)  # ds/de

    for j in range(7):
        drv[j] = s * dvz[j] + vz * ds[j]

    return rv_val, drv


@njit(fastmath=True)
def rv_d(time: float | NDArray, k: float, tk: float, p: float, a: float, i: float, e: float,
         c: NDArray, dc: NDArray) -> tuple[float | NDArray, NDArray]:
    """
    Evaluate the stellar radial velocity and its parameter derivatives at an absolute time.

    Direct counterpart of `rv_cd`: epoch-folds the absolute time
    `time` around the expansion point `tk` and delegates to `rv_cd`.

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
        A (3, 5) coefficient matrix produced by `solve3d`.
    dc : NDArray
        A (7, 3, 5) parameter-derivative tensor produced by
        `solve3d_d`.

    Returns
    -------
    rv : float or NDArray
        Stellar radial velocity in the same units as `k`.
    drv : NDArray
        Shape (7,) partial derivatives of `rv` with respect to
        `(tc, p, a, i, e, w, lan)`.
    """
    epoch = floor((time - tk + 0.5 * p) / p)
    return rv_cd(time - (tk + epoch * p), k, p, a, i, e, c, dc)
