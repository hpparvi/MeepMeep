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

"""Single-knot stellar radial-velocity evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import floor, sqrt, sin, cos, pi, zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array
from .zvelocity import _zvel_cd_w


@njit(fastmath=True, inline='always')
def _rv_scale(k, p, a, i, e):
    """RV scale factor ``s = k / n`` and its non-zero parameter derivatives.

    Returns ``(s, ds/dp, ds/da, ds/di, ds/de)``; the derivatives w.r.t.
    ``tc``, ``w``, and ``lan`` are identically zero. Hoist this out of
    vector loops: the factor depends only on the orbital parameters.
    """
    n = 2.0 * pi / p * (a * sin(i)) / sqrt(1.0 - e ** 2)
    s = k / n
    return s, s / p, -s / a, -s * cos(i) / sin(i), -s * e / (1.0 - e ** 2)


@njit(fastmath=True, inline='always')
def _rv_cd_w(time, s, dsp, dsa, dsi, dse, c, dc, drv, dvz):
    """Write-into kernel shared by the scalar and vector evaluators.

    Writes the seven-parameter gradient into the caller-provided ``(7,)``
    buffer ``drv`` and returns the radial velocity. ``dvz`` is a ``(7,)``
    scratch buffer for the z-velocity gradient; vector loops allocate it
    once and reuse it. The scale factor ``s`` and its derivatives come
    from :func:`_rv_scale`.
    """
    vz = _zvel_cd_w(time, c, dc, dvz)
    rv_val = s * vz
    for j in range(7):
        drv[j] = s * dvz[j]
    drv[1] += vz * dsp
    drv[2] += vz * dsa
    drv[3] += vz * dsi
    drv[4] += vz * dse
    return rv_val


@njit(fastmath=True)
def _rv_cd_s(time, k, p, a, i, e, c, dc):
    """Scalar kernel for :func:`rv_cd`. See that function for documentation."""
    s, dsp, dsa, dsi, dse = _rv_scale(k, p, a, i, e)
    drv = zeros(7)
    dvz = zeros(7)
    rv_val = _rv_cd_w(time, s, dsp, dsa, dsi, dse, c, dc, drv, dvz)
    return rv_val, drv


@njit(fastmath=True)
def _rv_cd_v(time, k, p, a, i, e, c, dc):
    """Vector kernel for :func:`rv_cd`. See that function for documentation."""
    nt = time.size
    rv_val = zeros(nt)
    drv = zeros((nt, 7))
    s, dsp, dsa, dsi, dse = _rv_scale(k, p, a, i, e)
    dvz = zeros(7)
    for j in range(nt):
        rv_val[j] = _rv_cd_w(time[j], s, dsp, dsa, dsi, dse, c, dc, drv[j], dvz)
    return rv_val, drv


def rv_cd(time: float | NDArray, k: float, p: float, a: float, i: float, e: float,
          c: NDArray, dc: NDArray):
    """
    Evaluate the stellar radial velocity and its parameter derivatives at a knot-centered time.

    Converts the planet's centered line-of-sight velocity into the
    physical radial velocity of the host star, scaled by the
    semi-amplitude `k`, following Perryman (2018) Eq. 2.23. The same
    chain rule is propagated to give the seven partial derivatives of
    the radial velocity with respect to the orbital parameters.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `radial_velocity.rv_c`.

    Parameters
    ----------
    time : float or ndarray
        Time(s) relative to the Taylor series expansion point.
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
    rv : float or ndarray
        Stellar radial velocity in the same units as `k`. Positive
        when the planet is moving toward the observer. Shape (N,) for an
        array `time`.
    drv : NDArray
        Partial derivatives of `rv` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.

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
    if isinstance(time, ndarray):
        return _rv_cd_v(time, k, p, a, i, e, c, dc)
    return _rv_cd_s(time, k, p, a, i, e, c, dc)


@overload(rv_cd, jit_options={'fastmath': True})
def _rv_cd_overload(time, k, p, a, i, e, c, dc):
    if _is_1d_array(time):
        def impl(time, k, p, a, i, e, c, dc):
            return _rv_cd_v(time, k, p, a, i, e, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, k, p, a, i, e, c, dc):
            return _rv_cd_s(time, k, p, a, i, e, c, dc)
        return impl
    return None


@njit(fastmath=True)
def _rv_d_s(time, k, tk, p, a, i, e, c, dc):
    """Scalar kernel for :func:`rv_d`. See that function for documentation."""
    epoch = floor((time - tk + 0.5 * p) / p)
    return _rv_cd_s(time - (tk + epoch * p), k, p, a, i, e, c, dc)


@njit(fastmath=True)
def _rv_d_v(time, k, tk, p, a, i, e, c, dc):
    """Vector kernel for :func:`rv_d`. See that function for documentation."""
    nt = time.size
    rv_val = zeros(nt)
    drv = zeros((nt, 7))
    s, dsp, dsa, dsi, dse = _rv_scale(k, p, a, i, e)
    dvz = zeros(7)
    for j in range(nt):
        epoch = floor((time[j] - tk + 0.5 * p) / p)
        rv_val[j] = _rv_cd_w(time[j] - (tk + epoch * p), s, dsp, dsa, dsi, dse, c, dc, drv[j], dvz)
    return rv_val, drv


def rv_d(time: float | NDArray, k: float, tk: float, p: float, a: float, i: float, e: float,
         c: NDArray, dc: NDArray):
    """
    Evaluate the stellar radial velocity and its parameter derivatives at an absolute time.

    Direct counterpart of `rv_cd`: epoch-folds the absolute time
    `time` around the expansion point `tk` and delegates to `rv_cd`.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python), mirroring the value-only `radial_velocity.rv`.

    Parameters
    ----------
    time : float or ndarray
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
    rv : float or ndarray
        Stellar radial velocity in the same units as `k`. Shape (N,) for
        an array `time`.
    drv : NDArray
        Partial derivatives of `rv` with respect to `(tc, p, a, i, e, w, lan)`.
        Shape (7,) for a scalar `time`, (N, 7) for an array `time`.
    """
    if isinstance(time, ndarray):
        return _rv_d_v(time, k, tk, p, a, i, e, c, dc)
    return _rv_d_s(time, k, tk, p, a, i, e, c, dc)


@overload(rv_d, jit_options={'fastmath': True})
def _rv_d_overload(time, k, tk, p, a, i, e, c, dc):
    if _is_1d_array(time):
        def impl(time, k, tk, p, a, i, e, c, dc):
            return _rv_d_v(time, k, tk, p, a, i, e, c, dc)
        return impl
    if isinstance(time, types.Float):
        def impl(time, k, tk, p, a, i, e, c, dc):
            return _rv_d_s(time, k, tk, p, a, i, e, c, dc)
        return impl
    return None
