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

"""Expansion-point placement, traceable with a traced eccentricity.

JAX port of
``meepmeep.backends.numba.expansion_points.create_expansion_points``.
The numba version finds the ``'ea'`` and ``'ta'`` placements with scipy's
``brentq``, which cannot run inside a trace. The root-finding is not
needed: the placements are points uniform in eccentric (or true) anomaly,
mapped to time, and that map is closed-form, ``t = (E - e sin E) / 2 pi``
(with ``E`` from the true anomaly by the half-angle formula). So the grid
can be built inside ``jit`` from a traced ``e``, and it agrees with the
numba grid to the ``brentq`` tolerance (~1e-12 in phase).

The time-to-expansion-point table is built with the same sequential rule
as numba (the bucket index advances by at most one per bin), so the two
tables are identical, including numba's behaviour when adjacent boundaries
fall within one bin of each other.
"""

import jax.numpy as jnp
from jax import lax

from ._common import TWO_PI


def _time_from_anomaly(v, e, quantity):
    """Phase from periastron at which the eccentric (``'ea'``) or true (``'ta'``) anomaly equals ``v``."""
    if quantity == 'ta':
        v = 2.0 * jnp.arctan2(jnp.sqrt(1.0 - e) * jnp.sin(0.5 * v), jnp.sqrt(1.0 + e) * jnp.cos(0.5 * v))
    return (v - e * jnp.sin(v)) / TWO_PI


def create_expansion_points(n_ep: int, e, quantity: str = 'ea', tres: int = 200):
    """Place expansion points along one orbital period and build the time-to-expansion-point table.

    Parameters
    ----------
    n_ep : int
        Number of expansion points, including the periodic-image slot. Must be
        odd. Static (a Python int).
    e : float
        Orbital eccentricity; may be a traced value.
    quantity : {'mm', 'ea', 'ta'}, optional
        Placement strategy: uniform in mean motion (time), eccentric anomaly
        (default), or true anomaly. Static.
    tres : int, optional
        Number of time-to-expansion-point table bins per period. Static.

    Returns
    -------
    ep_times : ndarray, shape (n_ep,)
        Expansion-point phases from periastron in ``[0, 1]``.
    change_times : ndarray, shape (n_ep - 1,)
        Phases at which the dispatch switches to the next expansion point.
    dt : float
        Table bin width, ``1 / tres``.
    ep_table : ndarray of int, shape (tres,)
        Expansion-point index per table bin (int32).

    Raises
    ------
    ValueError
        If ``quantity`` is unknown or ``n_ep`` is even.

    Notes
    -----
    The placement depends on ``e`` smoothly, so it is differentiable.
    Moving the expansion points shifts values only at the level of the
    truncation error, and the table index jumps as boundaries cross
    sample times; wrap ``e`` in ``jax.lax.stop_gradient`` for the numba
    convention of a fixed grid.
    """
    if quantity not in ('mm', 'ea', 'ta'):
        raise ValueError("Quantity needs to be either 'mm' for mean motion, 'ea' for eccentric anomaly, "
                         "or 'ta' for true anomaly.")
    if n_ep % 2 != 1:
        raise ValueError("Number of expansion points should be odd.")

    half = n_ep // 2
    if quantity == 'mm':
        ep_times = jnp.linspace(0.0, 1.0, n_ep)
        change_times = 0.5 * (ep_times[:-1] + ep_times[1:])
    else:
        e = jnp.asarray(e, dtype=float)
        ep_sep = TWO_PI / n_ep
        lower = _time_from_anomaly(jnp.arange(1, half) * ep_sep, e, quantity)
        ep_times = jnp.concatenate([jnp.zeros(1), lower, jnp.full(1, 0.5), 1.0 - lower[::-1], jnp.ones(1)])
        lower_ct = _time_from_anomaly((jnp.arange(half) + 0.5) * ep_sep, e, quantity)
        change_times = jnp.concatenate([lower_ct, 1.0 - lower_ct[::-1]])

    dt = 1.0 / tres

    def step(ik, i):
        advance = (ik < n_ep - 1) & (i * dt > change_times[jnp.minimum(ik, n_ep - 2)])
        ik = ik + advance.astype(jnp.int32)
        return ik, ik

    _, ep_table = lax.scan(step, jnp.int32(0), jnp.arange(tres))
    return ep_times, change_times, dt, ep_table
