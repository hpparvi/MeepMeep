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

"""Shared helpers for the JAX backend.

Holds the constants, the double-precision guard, the Horner evaluators of
the Taylor polynomial rows, and the two epoch-folding schemes (single
expansion point and multi-expansion-point table lookup) that every
evaluator module builds on.

The coefficient helpers index ``c[..., row, col]``, so they accept a single
``(D, 5)`` matrix (broadcast against an array of times) as well as a stack
``(N, D, 5)`` gathered per time by the multi-expansion-point lookup.
"""

import jax
import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi
HALF_PI = 0.5 * jnp.pi


def require_x64():
    """Raise unless JAX runs in double precision.

    Absolute times in transit work are BJDs around 2.4e6, where a float32
    ulp is about a quarter of a day, and the Taylor coefficients themselves
    need double precision to reach the backend's accuracy. The check runs at
    trace time, so it costs nothing inside a jitted function.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("The MeepMeep JAX backend needs double precision. Enable it before any JAX "
                           "computation with `jax.config.update('jax_enable_x64', True)` or by setting "
                           "the environment variable JAX_ENABLE_X64=1.")


def horner(t, c, row):
    """Evaluate one row of the Taylor polynomial (position) at centered time ``t``."""
    return c[..., row, 0] + t * (c[..., row, 1] + t * (c[..., row, 2] + t * (c[..., row, 3] + t * c[..., row, 4])))


def horner_d(t, c, row):
    """Evaluate the time derivative of one row of the Taylor polynomial (velocity)."""
    return c[..., row, 1] + t * (2.0 * c[..., row, 2] + t * (3.0 * c[..., row, 3] + t * 4.0 * c[..., row, 4]))


def fold(time, tc, p, te):
    """Map absolute time to time relative to the nearest image of a single expansion point.

    The expansion point sits at ``tc + te`` on the observation time axis; the
    epoch is chosen so that the returned time lies in ``[-p/2, p/2)``.
    """
    require_x64()
    time = jnp.asarray(time, dtype=float)
    epoch = jnp.floor((time - tc - te + 0.5 * p) / p)
    return time - (tc + te + epoch * p)


def ep_lookup(t, tpa, p, dt, ep_table, ep_times):
    """Fold a time into one period and dispatch it to its expansion point.

    Parameters
    ----------
    t : float or ndarray
        Time(s) [days].
    tpa : float
        Periastron time anchoring the expansion-point grid [days].
    p : float
        Orbital period [days].
    dt : float
        Width of one ``ep_table`` bucket in fraction of the period.
    ep_table : ndarray of int
        Time-to-expansion-point lookup table.
    ep_times : ndarray
        Normalised expansion-point phases.

    Returns
    -------
    tf : float or ndarray
        Time since the most recent periastron passage, in ``[0, p)``.
    tcc : float or ndarray
        Time relative to the selected expansion point.
    ix : int or ndarray of int
        Expansion-point index per time.

    Notes
    -----
    The table index is clipped into range, so a NaN time (whose integer
    cast is implementation-defined) cannot read out of bounds; its value
    comes out NaN through ``tcc`` regardless of the bucket.
    """
    require_x64()
    t = jnp.asarray(t, dtype=float)
    epoch = jnp.floor((t - tpa) / p)
    tf = t - tpa - epoch * p
    bucket = jnp.floor(tf / (dt * p)).astype(jnp.int32)
    ix = jnp.take(ep_table, bucket, mode='clip')
    return tf, tf - jnp.take(ep_times, ix) * p, ix
