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

"""Taylor coefficient solvers.

JAX ports of ``meepmeep.backends.numba.point2d.solve.solve2d``,
``meepmeep.backends.numba.point3d.solve.solve3d`` and
``meepmeep.backends.numba.orbit3d.solve3d_orbit``. There are no
``_d`` variants: differentiate the solver (or, better, the whole model that
uses its coefficients) with ``jax.jacfwd``/``jax.grad``. Because the
evaluators compute a polynomial whose coefficients these functions return,
autodiff gives the exact gradient of the polynomial actually evaluated,
which is also what the numba ``_d`` kernels compute analytically.

The solver is written element-wise in ``te``, so a batch of expansion times
gives a stack of coefficient matrices without ``vmap``.
"""

import jax.numpy as jnp

from ._common import TWO_PI, require_x64
from .newton import ea_from_ma
from .utils import mean_anomaly_at_transit


def _solve(te, p, a, i, e, w, lan, ndim):
    """Coefficient matrices of shape ``te.shape + (ndim, 5)``; see :func:`solve3d`."""
    require_x64()
    te = jnp.asarray(te, dtype=float)

    # Analytic differentiation of Keplerian motion
    # --------------------------------------------
    n = TWO_PI / p
    mu = n ** 2 * a ** 3  # Standard gravitational parameter [R_star^3 / day^2]

    sqe2 = jnp.sqrt(1.0 - e ** 2)
    ci = jnp.cos(i)
    si = jnp.sin(i)
    cw = jnp.cos(w)
    sw = jnp.sin(w)

    # 1. Mean and eccentric anomaly
    offset = mean_anomaly_at_transit(e, w)
    ma = (TWO_PI * (te - (-offset * p / TWO_PI)) / p) % TWO_PI
    ea = ea_from_ma(ma, e)
    sea = jnp.sin(ea)
    cea = jnp.cos(ea)

    # 2. Orbital-plane position and velocity
    r_val = a * (1.0 - e * cea)
    xi = a * (cea - e)
    eta = a * sqe2 * sea
    ea_dot = n * a / r_val
    v_xi = -a * sea * ea_dot
    v_eta = a * sqe2 * cea * ea_dot

    # 3. Acceleration, jerk and snap
    r2 = r_val ** 2
    v2 = v_xi ** 2 + v_eta ** 2
    rv = xi * v_xi + eta * v_eta

    inv_r3 = 1.0 / (r2 * r_val)
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2

    u = -mu * inv_r3
    u_dot = 3.0 * mu * rv * inv_r5
    u_ddot = 3.0 * mu * (v2 * inv_r5 - 5.0 * rv ** 2 * inv_r7) - 3.0 * u ** 2

    a_xi = u * xi
    a_eta = u * eta
    j_xi = u_dot * xi + u * v_xi
    j_eta = u_dot * eta + u * v_eta
    s_coeff = u_ddot + u ** 2
    s_xi = s_coeff * xi + 2.0 * u_dot * v_xi
    s_eta = s_coeff * eta + 2.0 * u_dot * v_eta

    # 4. Rotation to the sky frame, pre-scaled by the factorial of the order
    def row(m0, m1):
        return [m0 * xi + m1 * eta,
                m0 * v_xi + m1 * v_eta,
                (m0 * a_xi + m1 * a_eta) * 0.5,
                (m0 * j_xi + m1 * j_eta) / 6.0,
                (m0 * s_xi + m1 * s_eta) / 24.0]

    x = row(-cw, sw)
    y = row(-sw * ci, -cw * ci)

    # 5. Longitude of the ascending node: a constant rotation of the sky-plane
    # (x, y) about the line of sight.
    c_o = jnp.cos(lan)
    s_o = jnp.sin(lan)
    rows = [[c_o * xk - s_o * yk for xk, yk in zip(x, y)],
            [s_o * xk + c_o * yk for xk, yk in zip(x, y)]]
    if ndim == 3:
        rows.append(row(sw * si, cw * si))
    return jnp.stack([jnp.stack(jnp.broadcast_arrays(*r), axis=-1) for r in rows], axis=-2)


def solve2d(te, p, a, i, e, w, lan=0.0):
    """Taylor expansion of the sky-plane (x, y) position around an expansion point.

    Parameters
    ----------
    te : float or NDArray
        Expansion-point time relative to the transit centre [days]; an array
        gives one coefficient matrix per element.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis [R_star].
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    lan : float, optional
        Longitude of the ascending node [radians]. Defaults to 0.0.

    Returns
    -------
    cf : NDArray, shape ``te.shape + (2, 5)``
        Coefficients pre-scaled by the factorial of the Taylor order.
    """
    return _solve(te, p, a, i, e, w, lan, 2)


def solve3d(te, p, a, i, e, w, lan=0.0):
    """Taylor expansion of the (x, y, z) position around an expansion point.

    Parameters
    ----------
    te : float or NDArray
        Expansion-point time relative to the transit centre [days]; an array
        gives one coefficient matrix per element.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis [R_star].
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    lan : float, optional
        Longitude of the ascending node [radians]. A constant rotation of the
        sky-plane (x, y) about the line of sight; z is unaffected. Defaults
        to 0.0.

    Returns
    -------
    cf : NDArray, shape ``te.shape + (3, 5)``
        Rows x, y, z; columns position through snap, pre-scaled by the
        factorial of the Taylor order.
    """
    return _solve(te, p, a, i, e, w, lan, 3)


def solve3d_orbit(ep_times, p, a, i, e, w, lan=0.0):
    """Taylor coefficients at every expansion point of one orbit.

    Parameters
    ----------
    ep_times : NDArray, shape (npt,)
        Normalised expansion-point phases from periastron, with
        ``ep_times[-1] == ep_times[0] + 1`` (the periodic image), as built by
        :func:`~meepmeep.jax3d.create_expansion_points`.
    p, a, i, e, w, lan : float
        Orbital parameters, see :func:`solve3d`.

    Returns
    -------
    coeffs : NDArray, shape (npt, 3, 5)
        Coefficient matrix per expansion point. The last slot is a copy of
        the first.

    Notes
    -----
    Unlike the numba version there is no ``npt`` argument; it is the length
    of ``ep_times``.
    """
    to = mean_anomaly_at_transit(e, w) / TWO_PI * p
    coeffs = solve3d(p * jnp.asarray(ep_times)[:-1] - to, p, a, i, e, w, lan)
    return jnp.concatenate([coeffs, coeffs[:1]], axis=0)
