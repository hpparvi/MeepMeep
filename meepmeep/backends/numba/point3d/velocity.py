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

"""Single-knot 3D planet (vx, vy, vz) velocity evaluators."""

from numba import njit, prange, types
from numba.extending import overload
from numpy import zeros, ndarray
from numpy.typing import NDArray

from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _vel_c_s(time, c):
    """Scalar kernel for :func:`vel_c`. See that function for documentation."""
    vx = c[0, 1] + time * (2.0 * c[0, 2] + time * (3.0 * c[0, 3] + time * 4.0 * c[0, 4]))
    vy = c[1, 1] + time * (2.0 * c[1, 2] + time * (3.0 * c[1, 3] + time * 4.0 * c[1, 4]))
    vz = c[2, 1] + time * (2.0 * c[2, 2] + time * (3.0 * c[2, 3] + time * 4.0 * c[2, 4]))
    return vx, vy, vz


def _vel_c_v_body(time, c):
    """Vector-kernel body for :func:`vel_c`; see that function for documentation.

    Compiled twice: ``_vel_c_v`` is the serial kernel (``prange`` compiles
    as a plain ``range`` without ``parallel=True``) and ``_vel_c_vp`` the
    parallel twin. The loop writes only into per-sample output elements,
    so no per-thread scratch is needed.
    """
    n = time.size
    vx = zeros(n)
    vy = zeros(n)
    vz = zeros(n)
    for j in prange(n):
        vx[j], vy[j], vz[j] = _vel_c_s(time[j], c)
    return vx, vy, vz


_vel_c_v = njit(fastmath=True)(_vel_c_v_body)
_vel_c_vp = njit(fastmath=True, parallel=True)(_vel_c_v_body)


def vel_c(time: float | NDArray, c: NDArray) -> tuple[float | NDArray, float | NDArray, float | NDArray]:
    """
    Evaluate the planet's (vx, vy, vz) velocity at a knot-centered time.

    Centered velocity companion to `position.pos_c`. Each velocity
    component is obtained by analytically differentiating the
    corresponding 5th-order position polynomial; the resulting
    polynomial is 4th-order in `time` and is evaluated using Horner's
    scheme.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    appropriate kernel at compile time (inside ``@njit``) or at call time
    (pure Python).

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
    if isinstance(time, ndarray):
        return _vel_c_v(time, c)
    return _vel_c_s(time, c)


@overload(vel_c, jit_options={'fastmath': True}, inline='always')
def _vel_c_overload(time, c):
    if _is_1d_array(time):
        def impl(time, c):
            return _vel_c_v(time, c)
        return impl
    if isinstance(time, types.Float):
        def impl(time, c):
            return _vel_c_s(time, c)
        return impl
    return None
