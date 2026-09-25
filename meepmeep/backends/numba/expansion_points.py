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

"""Expansion-point placement and the time-to-expansion-point lookup table."""


from numba import njit
from numpy import pi, linspace, zeros, arange, arctan2, sqrt, sin, cos, concatenate, diff, searchsorted, ceil
from scipy.optimize import root_scalar

from .newton.newton import ea_newton_s, ta_newton_s


@njit
def eccentric_anomaly(t, e):
    """Eccentric anomaly at a normalised time since periastron.

    Placement helper for :func:`~meepmeep.numba3d.create_expansion_points`, not part of the
    public API.

    Parameters
    ----------
    t : float
        Time since periastron passage as a fraction of the orbital period.
    e : float
        Orbital eccentricity.

    Returns
    -------
    ea : float
        Eccentric anomaly [radians].
    """
    return ea_newton_s(t, 0.0, 1.0, e, 0.5*pi)

@njit
def true_anomaly(t, e):
    """True anomaly at a normalised time since periastron, wrapped into ``[0, 2 pi)``.

    Placement helper for :func:`~meepmeep.numba3d.create_expansion_points`, not part of the
    public API.

    Parameters
    ----------
    t : float
        Time since periastron passage as a fraction of the orbital period.
    e : float
        Orbital eccentricity.

    Returns
    -------
    f : float
        True anomaly [radians] in ``[0, 2 pi)``.
    """
    f = ta_newton_s(t, 0.0, 1.0, e, 0.5*pi)
    if f < 0.0:
        f += 2*pi
    return f


# The time-to-expansion-point table is sized so that its bins are TABLE_BINS_PER_REGION
# times narrower than the narrowest expansion-point region, and never has fewer
# than TABLE_MIN_SIZE bins. The size is computed for max(e, TABLE_SIZING_E), so it
# does not depend on e below that: the JAX backend needs a static table shape and
# can then match these tables bin for bin even when e is traced. TABLE_MAX_SIZE
# caps the table (8 MB of int64) for extreme eccentricities.
TABLE_MIN_SIZE = 200
TABLE_BINS_PER_REGION = 8
TABLE_SIZING_E = 0.9
TABLE_MAX_SIZE = 2 ** 20


def _phase_from_anomaly(v, e, quantity):
    """Phase from periastron at which the eccentric ('ea') or true ('ta') anomaly equals ``v``.

    The closed-form inverse of the anomalies the root finder in
    :func:`create_expansion_points` solves for.
    """
    if quantity == 'ta':
        v = 2.0 * arctan2(sqrt(1.0 - e) * sin(0.5 * v), sqrt(1.0 + e) * cos(0.5 * v))
    return (v - e * sin(v)) / (2.0 * pi)


def expansion_table_size(n_ep: int, e: float, quantity: str = 'ea') -> int:
    """Default number of time-to-expansion-point table bins for a placement.

    Parameters
    ----------
    n_ep : int
        Number of expansion points, including the periodic-image slot.
    e : float
        Orbital eccentricity the grid is placed for.
    quantity : {'mm', 'ea', 'ta'}, optional
        Placement strategy. Defaults to ``'ea'``.

    Returns
    -------
    tres : int
        ``max(200, ceil(8 / w))``, where ``w`` is the narrowest
        expansion-point region, in phase, of the placement for
        ``max(e, 0.9)``, capped at 2**20 bins. The size is therefore the
        same for every eccentricity up to 0.9.

    Notes
    -----
    Each bin of the table maps to one expansion point, so the bins must be
    narrower than the regions they resolve. Near periastron at high
    eccentricity the ``'ea'`` and ``'ta'`` regions shrink to a fraction of a
    percent of the period; eight bins per narrowest region keep the
    table's contribution to the error below the Taylor truncation error.
    """
    if quantity == 'mm':
        w_min = 0.5 / (n_ep - 1)
    else:
        es = max(e, TABLE_SIZING_E)
        half = n_ep // 2
        lower = _phase_from_anomaly((arange(half) + 0.5) * 2.0 * pi / n_ep, es, quantity)
        widths = diff(concatenate(([0.0], lower, 1.0 - lower[::-1], [1.0])))
        w_min = widths.min()
    # The relative margin absorbs round-off between this closed form and the
    # placed grid, so the bins stay narrow enough for the actual regions too.
    return min(TABLE_MAX_SIZE, max(TABLE_MIN_SIZE, int(ceil(TABLE_BINS_PER_REGION / w_min * (1.0 + 1e-9)))))


def create_expansion_points(n_ep: int, e: float, quantity: str = 'ea', tres: int | None = None):
    """Place expansion points along one orbital period and build the time-to-expansion-point table.

    An *expansion point* is a point along the orbit that serves as the
    center of a local 5th-order Taylor expansion of the planet's trajectory
    in time. This function distributes ``n_ep`` such expansion centers over
    a single period and records where the dispatch should switch from one
    expansion point to the next.

    The expansion points are the expansion *centers*; the boundaries
    between adjacent expansion points' regions of validity are a separate
    concept, returned as ``change_times``.

    Parameters
    ----------
    n_ep : int
        Number of expansion points. Must be odd so that one expansion point
        lands at the orbit midpoint.
    e : float
        Orbital eccentricity, used by the ``'ea'`` and ``'ta'`` placement
        strategies to cluster expansion points near periastron.
    quantity : {'mm', 'ea', 'ta'}, optional
        Expansion-point placement strategy: ``'mm'`` spaces expansion points
        uniformly in mean motion (time), ``'ea'`` in eccentric anomaly
        (default), ``'ta'`` in true anomaly.
    tres : int, optional
        Resolution of the time-to-expansion-point lookup table (number of
        bins per period). Defaults to :func:`~meepmeep.numba3d.expansion_table_size`, which keeps the
        bins eight times narrower than the narrowest expansion-point region.

    Returns
    -------
    ep_times : NDArray
        Times of the expansion points (expansion centers), as fractions of
        the orbital period in ``[0, 1]``.
    change_times : NDArray
        Boundary times at which the time-to-expansion-point dispatch
        switches from one expansion point to the next, i.e. the edges of
        each expansion point's region of validity (one fewer than
        ``ep_times``).
    dt : float
        Width of a single time-to-expansion-point table bin, ``1 / tres``.
    ep_table : NDArray of int
        Time-to-expansion-point table mapping each of the ``tres`` time bins
        within one folded period to the index of the expansion point whose
        region contains the bin's centre.

    Notes
    -----
    The ``'ea'`` and ``'ta'`` grids are *not* uniform in their anomaly, and
    in particular do not reduce to the ``'mm'`` grid at zero eccentricity.
    The expansion points are spaced ``2*pi/n_ep`` apart in anomaly even
    though the grid holds only ``n_ep - 1`` distinct expansion points (the
    last slot is the periodic image of the first), and the midpoint
    expansion point is pinned at anomaly pi, which is not a multiple of that
    spacing. The leftover space collects as a 1.5x-wide gap on each side of
    the midpoint expansion point, whose region of validity is therefore
    twice as wide as the others'. This is benign for accuracy: the midpoint
    expansion point sits at apoastron, where the planet moves slowest and
    the Taylor truncation error is smallest, while the remaining expansion
    points are spaced tighter than uniform near periastron, where the error
    budget is actually spent.
    """
    if quantity not in ('mm', 'ea', 'ta'):
        raise ValueError("Quantity needs to be either 'mm' for mean motion, 'ea' for eccentric anomaly, or 'ta' for true anomaly.")
    if n_ep % 2 != 1:
        raise ValueError("Number of expansion points should be odd.")

    if quantity == 'mm':
        ep_times = linspace(0.0, 1.0, n_ep)
        change_times = 0.5 * (ep_times[:-1] + ep_times[1:])
    else:
        if quantity == 'ea':
            def cfun(t, e, v):
                return eccentric_anomaly(t, e) - v
        else:  # quantity == 'ta' (validated above)
            def cfun(t, e, v):
                return true_anomaly(t, e) - v

        ep_sep = 2 * pi / n_ep

        ep_times = zeros(n_ep)
        ep_times[n_ep // 2] = 0.5
        t0 = 1e-5
        for i in range(1, n_ep // 2):
            ep_times[i] = root_scalar(cfun, args=(e, i * ep_sep), bracket=(t0, 1.0 - 1e-5)).root
            t0 = ep_times[i]
        ep_times[n_ep // 2 + 1:-1] = 1 - ep_times[n_ep // 2 - 1:0:-1]
        ep_times[-1] = 1.0

        change_times = zeros(n_ep-1)
        t0 = 1e-5
        for i in range(0, n_ep // 2):
            change_times[i] = root_scalar(cfun, args=(e, (i + 0.5) * ep_sep), bracket=(t0, 1.0 - 1e-5)).root
            t0 = change_times[i]
        change_times[n_ep // 2:] = 1 - change_times[n_ep // 2 - 1::-1]

    # Time-to-expansion-point table: each bin maps to the expansion point whose
    # region contains the bin's centre.
    if tres is None:
        tres = expansion_table_size(n_ep, e, quantity)
    dt = 1 / tres
    ep_table = searchsorted(change_times, (arange(tres) + 0.5) * dt, side='left')

    return ep_times, change_times, dt, ep_table
