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
from numpy import floor, sqrt
from numpy.typing import NDArray


@njit(fastmath=True, inline='always')
def pos_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position at a knot-centered time.

    This is the "centered" variant of `pos`: it assumes the caller has
    already subtracted the expansion time `t0` (and any epoch offset) so
    that `time` is a small displacement around the knot. Each spatial
    coordinate is evaluated as a 5th-order polynomial using Horner's
    scheme.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point, i.e.
        `time = tc - (t0 + epoch*p)`. Must lie within the knot's region
        of validity for the truncation error to remain small.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. See `pos` for
        the row/column ordering convention.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.

    Notes
    -----
    This is the fastest 3D position evaluator in the module since it
    skips the epoch-folding arithmetic. Prefer it whenever the knot
    index and centered time are already known (e.g. inside multi-knot
    dispatch loops in `orbit3d.py`).
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    pz = c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))
    return px, py, pz


@njit(fastmath=True, inline='always')
def pos(time: float | NDArray, t0: float, p: float, c: NDArray) -> tuple[
    float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position at an absolute time using a 3D Taylor expansion.

    This is the "direct" variant of the 3D position evaluator: it
    accepts an absolute observation time `time`, folds it back into a
    single orbital epoch around the expansion point `t0`, and then
    evaluates the 5th-order Taylor polynomial stored in `c` using
    Horner's scheme via `pos_c`.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s) in the same units as `t0` and `p`
        (typically days). Scalar or array inputs are both accepted; the
        return type matches.
    t0 : float
        Time at which the Taylor series was expanded (the knot time).
    p : float
        Orbital period, used to fold `time` into the interval
        `[t0 - p/2, t0 + p/2)`.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Row 0 holds
        the x-direction coefficients, row 1 the y-direction, and row 2
        the z-direction, ordered as
        [position, velocity, acceleration/2, jerk/6, snap/24]
        (i.e. already pre-scaled by the factorial of the Taylor order).

    Returns
    -------
    px : float or NDArray
        Sky-plane x position(s) in units of stellar radii.
    py : float or NDArray
        Sky-plane y position(s) in units of stellar radii.
    pz : float or NDArray
        Line-of-sight z position(s) in units of stellar radii. Positive
        values point toward the observer.

    """
    epoch = floor((time - t0 + 0.5 * p) / p)
    return pos_c(time - (t0 + epoch * p), c)


@njit(fastmath=True, inline='always')
def sep_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation in units of stellar radii at a knot-centered time.

    Centered counterpart of `sep`: assumes `time` has already been
    shifted to be relative to the expansion point. Only the x and y
    Taylor polynomials are evaluated; the z polynomial is skipped
    because the sky projection drops the line-of-sight component.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only rows 0
        and 1 (x and y) are read.

    Returns
    -------
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
        Always non-negative.

    Notes
    -----
    Unlike the 2D analogue, which delegates to `pos_c`, this routine
    inlines the px/py Horner evaluations to avoid the wasted work of
    computing the z coefficient that `pos_c` would also evaluate.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True, inline='always')
def sep(time: float | NDArray, t0: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the sky-projected planet-star separation at an absolute time.

    Folds the absolute observation time back to a knot-centered offset
    and delegates to `sep_c`. This is the quantity most commonly used
    by transit light-curve models, where it represents the sky-projected
    separation between the centers of the star and planet in units of
    the stellar radius.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    t0 : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
        Always non-negative; the sign of the line-of-sight depth
        (transit vs. eclipse) is not encoded here. Use `pz` or `pz_c`
        if the transit/eclipse branch is needed.
    """
    epoch = floor((time - t0 + 0.5 * p) / p)
    return sep_c(time - (t0 + epoch * p), c)


@njit(fastmath=True, inline='always')
def pos_and_sep_c(time: float | NDArray, c: NDArray) -> tuple[
    float | NDArray, float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position and the sky-projected separation jointly.

    Returns the sky-plane coordinates, the line-of-sight coordinate,
    and the sky-projected distance `sqrt(x^2 + y^2)` from a single
    Horner-scheme pass, saving the redundant polynomial work that
    would occur if `pos_c` and `sep_c` were called separately.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
        Computed from `px` and `py` only.
    """
    px = c[0, 0] + time * (c[0, 1] + time * (c[0, 2] + time * (c[0, 3] + time * c[0, 4])))
    py = c[1, 0] + time * (c[1, 1] + time * (c[1, 2] + time * (c[1, 3] + time * c[1, 4])))
    pz = c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))
    return px, py, pz, sqrt(px ** 2 + py ** 2)


@njit(fastmath=True, inline='always')
def pos_and_sep(time: float | NDArray, t0: float, p: float, c: NDArray) -> tuple[
    float | NDArray, float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (x, y, z) position and the sky-projected separation at an absolute time.

    Direct counterpart of `pos_and_sep_c`: epoch-folds the absolute
    time and delegates to the centered evaluator.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    t0 : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    px : float or NDArray
        Sky-plane x position in units of stellar radii.
    py : float or NDArray
        Sky-plane y position in units of stellar radii.
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer.
    d : float or NDArray
        Sky-projected planet-star separation in units of stellar radii.
    """
    epoch = floor((time - t0 + 0.5 * p) / p)
    return pos_and_sep_c(time - (t0 + epoch * p), c)


@njit(fastmath=True, inline='always')
def pz_c(time: float | NDArray, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight z position at a knot-centered time.

    Centered counterpart of `pz`: evaluates only the z-direction Taylor
    polynomial (row 2 of `c`), skipping the x and y rows. This is the
    cheapest 3D evaluator in the module and is the right choice when
    only the transit/eclipse branch is needed (e.g. to discriminate
    primary from secondary eclipse via the sign of z).

    Parameters
    ----------
    time : float or NDArray
        Time relative to the Taylor series expansion point.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`. Only row 2
        (the z-direction coefficients) is read.

    Returns
    -------
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
    """
    return c[2, 0] + time * (c[2, 1] + time * (c[2, 2] + time * (c[2, 3] + time * c[2, 4])))


@njit(fastmath=True, inline='always')
def pz(time: float | NDArray, t0: float, p: float, c: NDArray) -> float | NDArray:
    """
    Evaluate the planet's line-of-sight z position at an absolute time.

    Folds the absolute observation time back to a knot-centered offset
    and delegates to `pz_c`.

    Parameters
    ----------
    time : float or NDArray
        Absolute observation time(s).
    t0 : float
        Taylor series expansion time (knot time).
    p : float
        Orbital period.
    c : NDArray
        A (3, 5) coefficient matrix produced by `solve3d`.

    Returns
    -------
    pz : float or NDArray
        Line-of-sight z position in units of stellar radii. Positive
        values point toward the observer; negative values point away.
        The sign distinguishes the transit (positive z) and eclipse
        (negative z) branches of the orbit.
    """
    epoch = floor((time - t0 + 0.5 * p) / p)
    return pz_c(time - (t0 + epoch * p), c)
