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

"""Transit geometry from a single Taylor expansion: contact points, durations, minimum separation.

JAX ports of ``meepmeep.backends.numba.point3d.util`` (identical to the
2D module, since only the x and y rows are used; ``c`` may be ``(2, 5)`` or
``(3, 5)``). The searches replicate the numba bisection and golden-section
loops step by step inside ``lax.while_loop``, so the values agree with
numba to round-off.

A while loop cannot be reverse-differentiated, so the derivatives come from
``custom_jvp`` rules that apply the implicit function theorem at the found
point instead of differentiating the iteration. That makes the contact
times and the durations differentiable w.r.t. ``k`` and the coefficients,
and through them w.r.t. the orbital parameters, which the numba backend
does not offer.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ._common import horner
from .point3d import sep_c


def _contact_setup(point):
    """Search direction and the target separation ``z0 + zk * k`` for a contact point."""
    s = -1.0 if point in (1, 2, 12) else 1.0
    if point in (1, 4):
        return s, 1.0, 1.0
    elif point in (2, 3):
        return s, 1.0, -1.0
    return s, 1.0, 0.0


def _contact_bisect(k, point, c):
    s, z0, zk = _contact_setup(point)
    zt = z0 + zk * k
    speed = jnp.sqrt(c[0, 1] ** 2 + c[1, 1] ** 2)
    t0 = jnp.zeros_like(speed)
    t2 = s * 2.0 / speed
    t1 = 0.5 * t2
    za = sep_c(t0, c) - zt
    zb = sep_c(t1, c) - zt

    def cond(state):
        t0, _, t2, _, _, j = state
        return (jnp.abs(t2 - t0) > 1e-6) & (j < 100)

    def body(state):
        t0, t1, t2, za, zb, j = state
        straddles = za * zb < 0.0
        t1_l, t2_l = 0.5 * (t0 + t1), t1
        t0_r, t1_r = t1, 0.5 * (t1 + t2)
        t0n = jnp.where(straddles, t0, t0_r)
        t1n = jnp.where(straddles, t1_l, t1_r)
        t2n = jnp.where(straddles, t2_l, t2)
        zan = jnp.where(straddles, za, zb)
        return t0n, t1n, t2n, zan, sep_c(t1n, c) - zt, j + 1

    return lax.while_loop(cond, body, (t0, t1, t2, za, zb, 0))[1]


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def find_contact_point(k, point, c):
    """Time of a transit contact point relative to the expansion point.

    Parameters
    ----------
    k : float
        Planet-star radius ratio.
    point : int
        Contact point: 1 and 4 are the outer contacts (projected separation
        ``1 + k``), 2 and 3 the inner contacts (``1 - k``), and 12 the
        ingress midpoint (separation 1). Must be a Python int.
    c : ndarray, shape (2, 5) or (3, 5)
        Taylor coefficients expanded near the transit centre.

    Returns
    -------
    float
        Contact time relative to the expansion point [days], found by
        bisection to a 1e-6 day bracket.

    Notes
    -----
    The derivative is that of the exact root, ``dt = (dzt - dsep) / sep'``,
    evaluated at the bisection result.
    """
    return _contact_bisect(k, point, c)


@find_contact_point.defjvp
def _find_contact_point_jvp(point, primals, tangents):
    k, c = primals
    dk, dc = tangents
    t = _contact_bisect(k, point, c)
    _, _, zk = _contact_setup(point)
    dsep_dt = jax.grad(sep_c, argnums=0)(t, c)
    _, dsep_c = jax.jvp(lambda cc: sep_c(t, cc), (c,), (dc,))
    return t, (zk * dk - dsep_c) / dsep_dt


def bounding_box(k, coeffs):
    """First and fourth contact times ``(t1, t4)`` relative to the expansion point [days]."""
    return find_contact_point(k, 1, coeffs), find_contact_point(k, 4, coeffs)


def t14(k, c):
    """Total transit duration [days]."""
    return find_contact_point(k, 4, c) - find_contact_point(k, 1, c)


def t23(k, c):
    """Full (flat-bottom) transit duration [days]."""
    return find_contact_point(k, 3, c) - find_contact_point(k, 2, c)


def t12(k, c):
    """Ingress duration [days]."""
    return find_contact_point(k, 2, c) - find_contact_point(k, 1, c)


def t34(k, c):
    """Egress duration [days]."""
    return find_contact_point(k, 4, c) - find_contact_point(k, 3, c)


def t1(k, c):
    """First contact time relative to the expansion point [days]."""
    return find_contact_point(k, 1, c)


def t4(k, c):
    """Fourth contact time relative to the expansion point [days]."""
    return find_contact_point(k, 4, c)


def _golden_section(tc, c):
    r = 0.61803399
    cc = 1.0 - r
    tc = jnp.asarray(tc, dtype=float)
    x0, x3 = tc - 0.01, tc + 0.01
    x1 = tc
    x2 = tc + cc * (x3 - tc)

    def cond(state):
        x0, _, _, x3, _, _, j = state
        return (jnp.abs(x3 - x0) > 1e-7) & (j < 100)

    def body(state):
        x0, x1, x2, x3, f1, f2, j = state
        right = f2 < f1
        x0n = jnp.where(right, x1, x0)
        x3n = jnp.where(right, x3, x2)
        x1n = jnp.where(right, x2, r * x1 + cc * x0)
        x2n = jnp.where(right, r * x2 + cc * x3, x1)
        fnew = sep_c(jnp.where(right, x2n, x1n), c)
        f1n = jnp.where(right, f2, fnew)
        f2n = jnp.where(right, fnew, f1)
        return x0n, x1n, x2n, x3n, f1n, f2n, j + 1

    state = (x0, x1, x2, x3, sep_c(x1, c), sep_c(x2, c), 0)
    _, x1, x2, _, f1, f2, _ = lax.while_loop(cond, body, state)
    return jnp.where(f1 < f2, x1, x2), jnp.where(f1 < f2, f1, f2)


def _sep2_c(t, c):
    return horner(t, c, 0) ** 2 + horner(t, c, 1) ** 2


@jax.custom_jvp
def find_z_min(tc, c):
    """Time and value of the minimum projected separation near ``tc``.

    Parameters
    ----------
    tc : float
        Initial guess, relative to the expansion point [days]. The search
        brackets ``tc +- 0.01`` days.
    c : ndarray, shape (2, 5) or (3, 5)
        Taylor coefficients.

    Returns
    -------
    t_min : float
        Time of the minimum relative to the expansion point [days], to a
        1e-7 day bracket.
    z_min : float
        Minimum projected separation [R_star].

    Notes
    -----
    The derivative of ``t_min`` follows from the stationarity of the squared
    separation, which stays smooth at a zero-impact-parameter minimum where
    the separation itself has a kink. The derivative w.r.t. the guess is
    zero. ``z_min`` has a NaN derivative when the minimum separation is
    exactly zero.
    """
    return _golden_section(tc, c)


@find_z_min.defjvp
def _find_z_min_jvp(primals, tangents):
    tc, c = primals
    _, dc = tangents
    t, z = _golden_section(tc, c)
    dh_dt = jax.grad(_sep2_c, argnums=0)
    d2h_dt2 = jax.grad(dh_dt, argnums=0)(t, c)
    _, d2h_dtdc = jax.jvp(lambda cc: dh_dt(t, cc), (c,), (dc,))
    dt = -d2h_dtdc / d2h_dt2
    _, dh_c = jax.jvp(lambda cc: _sep2_c(t, cc), (c,), (dc,))
    dz = (dh_c + dh_dt(t, c) * dt) / (2.0 * z)
    return (t, z), (dt, dz)
