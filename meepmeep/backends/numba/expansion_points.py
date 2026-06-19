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
from numpy import pi, linspace, zeros
from scipy.optimize import root_scalar

from .newton.newton import ea_newton_s, ta_newton_s


@njit
def eccentric_anomaly(t, e):
    return ea_newton_s(t, 0.0, 1.0, e, 0.5*pi)

@njit
def true_anomaly(t, e):
    f = ta_newton_s(t, 0.0, 1.0, e, 0.5*pi)
    if f < 0.0:
        f += 2*pi
    return f


def create_expansion_points(n_ep: int, e: float, quantity: str = 'ea', tres: int = 200):
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
        bins per period).

    Returns
    -------
    ep_times : ndarray
        Times of the expansion points (expansion centers), as fractions of
        the orbital period in ``[0, 1]``.
    change_times : ndarray
        Boundary times at which the time-to-expansion-point dispatch
        switches from one expansion point to the next, i.e. the edges of
        each expansion point's region of validity (one fewer than
        ``ep_times``).
    dt : float
        Width of a single time-to-expansion-point table bin, ``1 / tres``.
    ep_table : ndarray of int
        Time-to-expansion-point table mapping each of the ``tres`` time bins
        within one folded period to the index of the expansion point that
        should evaluate it.

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

    # Create the time-to-expansion-point table
    dt = 1 / tres
    ep_table = zeros(tres, int)
    ik = 0
    for i in range(tres):
        if i*dt > change_times[ik]:
            ik += 1
        if ik >= n_ep-1:
            ep_table[i:] = ik
            break
        ep_table[i] = ik

    return ep_times, change_times, dt, ep_table
