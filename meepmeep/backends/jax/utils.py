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

"""Orbital-mechanics utilities for the JAX backend.

Element-wise JAX ports of ``meepmeep.backends.numba.utils``. Every
function accepts scalars or broadcastable arrays and is differentiable.
The gradient basis transforms of the numba module (``tc_to_tp_gradient``
and friends) have no counterpart: with autodiff, the basis is set by which
timing parameter the differentiated function takes.
"""

import jax.numpy as jnp
from scipy.constants import G

from ._common import TWO_PI, HALF_PI


def eccentricity_vector(i, e, w, lan=0.0):
    """Eccentricity vector in the sky frame.

    Parameters
    ----------
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    lan : float, optional
        Longitude of the ascending node [radians]. Rotates (ex, ey) about the
        line of sight, as the solvers rotate the positions. Defaults to 0.0.

    Returns
    -------
    ev : NDArray, shape (3,)
        ``(ex, ey, ez)``. Near-circular orbits (``e <= 1e-5``) return the
        sentinel ``(-1, 0, 0)``, which the true-anomaly evaluator
        recognises and replaces with the mean anomaly.

    Notes
    -----
    Pass the same ``lan`` the coefficients were solved with; with a non-zero
    node an unrotated vector puts the true anomaly off by up to the node
    angle.
    """
    circular = e <= 1e-5
    ci = jnp.cos(i)
    si = jnp.sin(i)
    ex0 = -e * jnp.cos(w)
    ey0 = -e * jnp.sin(w) * ci
    c_o = jnp.cos(lan)
    s_o = jnp.sin(lan)
    ex = jnp.where(circular, -1.0, c_o * ex0 - s_o * ey0)
    ey = jnp.where(circular, 0.0, s_o * ex0 + c_o * ey0)
    ez = jnp.where(circular, 0.0, e * jnp.sin(w) * si)
    return jnp.stack(jnp.broadcast_arrays(ex, ey, ez))


def eclipse_time_offset(p, i, e, w):
    """Time from the transit centre to the secondary-eclipse centre [days], in ``(0, p]``."""
    sqe2 = jnp.sqrt(1.0 - e ** 2)
    etr = jnp.arctan2(sqe2 * jnp.sin(HALF_PI - w), e + jnp.cos(HALF_PI - w))
    eec = jnp.arctan2(sqe2 * jnp.sin(HALF_PI + jnp.pi - w), e + jnp.cos(HALF_PI + jnp.pi - w))
    mtr = etr - e * jnp.sin(etr)
    mec = eec - e * jnp.sin(eec)
    offset = (mec - mtr) * p / TWO_PI
    return jnp.where(offset > 0.0, offset, p + offset)


def transit_distance_factor(e, w):
    """Star-planet distance at transit in units of the semi-major axis."""
    return (1.0 - e ** 2) / (1.0 + e * jnp.sin(w))


def i_from_baew(b, a, e, w):
    """Inclination [radians] from the impact parameter, scaled semi-major axis, e and w."""
    return jnp.arccos(b / (a * transit_distance_factor(e, w)))


def as_from_rhop(rho, period):
    """Scaled semi-major axis from the stellar density [g/cm^3] and orbital period [days]."""
    return (G / (3 * jnp.pi)) ** (1 / 3) * ((period * 86400.0) ** 2 * 1000.0 * rho) ** (1 / 3)


def ta_from_ea(ea, ecc):
    """True anomaly [radians] from the eccentric anomaly."""
    sta = jnp.sqrt(1.0 - ecc ** 2) * jnp.sin(ea) / (1.0 - ecc * jnp.cos(ea))
    cta = (jnp.cos(ea) - ecc) / (1.0 - ecc * jnp.cos(ea))
    return jnp.arctan2(sta, cta)


def mean_anomaly_at_transit(ecc, w):
    """Mean anomaly [radians] at the transit centre (inferior conjunction)."""
    ea_tr = jnp.arctan2(jnp.sqrt(1.0 - ecc ** 2) * jnp.sin(HALF_PI - w), ecc + jnp.cos(HALF_PI - w))
    return ea_tr - ecc * jnp.sin(ea_tr)


def mean_anomaly(t, tc, p, e, w):
    """Mean anomaly [radians] at time ``t``, wrapped into ``[0, 2 pi)``."""
    offset = mean_anomaly_at_transit(e, w)
    return jnp.mod(TWO_PI * (t - (tc - offset * p / TWO_PI)) / p, TWO_PI)


def z_from_ta(f, a, i, e, w):
    """Sky-projected separation from the true anomaly, signed by the side of the sky plane."""
    z = a * (1.0 - e ** 2) / (1.0 + e * jnp.cos(f)) * jnp.sqrt(1.0 - jnp.sin(w + f) ** 2 * jnp.sin(i) ** 2)
    return z * jnp.copysign(1.0, jnp.sin(w + f))


def impact_parameter(a, i):
    """Impact parameter of a circular orbit."""
    return a * jnp.cos(i)


def impact_parameter_ec(a, i, e, w, tr_sign):
    """Impact parameter of an eccentric orbit (``tr_sign`` = 1 for transit, -1 for eclipse)."""
    return a * jnp.cos(i) * ((1.0 - e ** 2) / (1.0 + tr_sign * e * jnp.sin(w)))


def d_from_pkaiews(p, k, a, i, e, w, tr_sign, kind=14):
    """Transit (``kind=14``) or full-transit (``kind=23``) duration [days]."""
    b = impact_parameter_ec(a, i, e, w, tr_sign)
    ae = jnp.sqrt(1.0 - e ** 2) / (1.0 + tr_sign * e * jnp.sin(w))
    ds = 1.0 if kind == 14 else -1.0
    return p / jnp.pi * jnp.arcsin(jnp.sqrt((1.0 + ds * k) ** 2 - b ** 2) / (a * jnp.sin(i))) * ae
