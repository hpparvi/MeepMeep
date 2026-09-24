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

"""Kepler-equation solver and exact Newton-Raphson orbit references.

JAX ports of ``meepmeep.backends.numba.newton.newton``. The numba
module's ``_s`` (scalar) and ``_v`` (vector) variants collapse into one
element-wise function each.

:func:`ea_from_ma` iterates to a tolerance inside ``lax.while_loop``, which
reverse-mode autodiff cannot differentiate. Its derivative comes instead
from a ``custom_jvp`` rule derived by implicit differentiation of Kepler's
equation, so the gradient is exact whatever the iteration did, and the
function works under ``jax.grad`` as well as ``jax.jvp``.
"""

import jax
import jax.numpy as jnp
from jax import lax

from .utils import mean_anomaly, ta_from_ea, eclipse_time_offset, z_from_ta

LTT_DAYS_PER_RSUN = 2.685885891543453e-05


def _ea_newton_loop(ma, ecc):
    """Newton iteration of the numba ``ea_from_ma``: same start, tolerance and iteration cap."""
    ma, ecc = jnp.broadcast_arrays(jnp.asarray(ma, dtype=float), jnp.asarray(ecc, dtype=float))
    ea0 = jnp.where(ecc > 0.8, jnp.pi, ma)

    def cond(state):
        _, dea, j = state
        return jnp.any((jnp.abs(dea) >= 1e-13) & (j < 50))

    def body(state):
        ea, dea, j = state
        active = (jnp.abs(dea) >= 1e-13) & (j < 50)
        step = -(ea - ecc * jnp.sin(ea) - ma) / (1.0 - ecc * jnp.cos(ea))
        return (jnp.where(active, ea + step, ea),
                jnp.where(active, step, dea),
                jnp.where(active, j + 1, j))

    state = (ea0, jnp.full_like(ea0, jnp.inf), jnp.zeros(ea0.shape, jnp.int32))
    return lax.while_loop(cond, body, state)[0]


@jax.custom_jvp
def ea_from_ma(ma, ecc):
    """Solve Kepler's equation for the eccentric anomaly.

    Parameters
    ----------
    ma : float or ndarray
        Mean anomaly [radians].
    ecc : float or ndarray
        Eccentricity, broadcastable against ``ma``.

    Returns
    -------
    ea : float or ndarray
        Eccentric anomaly [radians].

    Notes
    -----
    Every element iterates on its own until its Newton step drops below
    1e-13 or 50 steps have been taken, exactly as the numba solver does; the
    array form only masks the elements that have already converged.
    """
    return _ea_newton_loop(ma, ecc)


@ea_from_ma.defjvp
def _ea_from_ma_jvp(primals, tangents):
    ma, ecc = primals
    dma, decc = tangents
    ea = _ea_newton_loop(ma, ecc)
    denom = 1.0 - ecc * jnp.cos(ea)
    return ea, (dma + jnp.sin(ea) * decc) / denom


def ea_newton(t, tc, p, e, w):
    """Eccentric anomaly [radians] at time ``t``."""
    return ea_from_ma(mean_anomaly(t, tc, p, e, w), e)


def ta_newton(t, tc, p, e, w):
    """True anomaly [radians] at time ``t``, in ``(-pi, pi]``."""
    return ta_from_ea(ea_newton(t, tc, p, e, w), e)


def xy_newton(time, tc, p, a, i, e, w):
    """Exact sky-plane position ``(x, y)`` [R_star]."""
    f = ta_newton(time, tc, p, e, w)
    r = a * (1.0 - e ** 2) / (1.0 + e * jnp.cos(f))
    x = -r * jnp.cos(w + f)
    y = -r * jnp.sin(w + f) * jnp.cos(i)
    return x, y


def xyz_newton(time, tc, p, a, i, e, w):
    """Exact position ``(x, y, z)`` [R_star]."""
    f = ta_newton(time, tc, p, e, w)
    r = a * (1.0 - e ** 2) / (1.0 + e * jnp.cos(f))
    x = -r * jnp.cos(w + f)
    y = -r * jnp.sin(w + f) * jnp.cos(i)
    z = r * jnp.sin(w + f) * jnp.sin(i)
    return x, y, z


def z_newton(time, tc, p, a, i, e, w):
    """Exact signed sky-projected separation [R_star] (see ``meepmeep.backends.jax.utils.z_from_ta``)."""
    return z_from_ta(ta_newton(time, tc, p, e, w), a, i, e, w)


def rv_newton(times, k, tc, p, e, w):
    """Exact radial velocity with semi-amplitude ``k``."""
    ta = ta_newton(times, tc, p, e, w)
    return k * (jnp.cos(w + ta) + e * jnp.cos(w))


def eclipse_light_travel_time(p, a, i, e, w, rstar):
    """Light travel time [days] between the transit and the secondary eclipse.

    Parameters
    ----------
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
    rstar : float
        Stellar radius [R_sun].

    Returns
    -------
    float
        The eclipse is observed this much later than a zero-light-travel-time
        model predicts [days].
    """
    ae = a * (1.0 - e ** 2)
    si = jnp.sin(i)
    f = ta_newton(0.0, 0.0, p, e, w)
    ztr = ae / (1.0 + e * jnp.cos(f)) * jnp.sin(w + f) * si
    f = ta_newton(eclipse_time_offset(p, i, e, w), 0.0, p, e, w)
    zec = ae / (1.0 + e * jnp.cos(f)) * jnp.sin(w + f) * si
    return (ztr - zec) * rstar * LTT_DAYS_PER_RSUN
