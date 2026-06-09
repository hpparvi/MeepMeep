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


def create_knots(n_knots: int, e: float, quantity: str = 'ea', tres: int = 200):
    """Place knots along one orbital period and build the time-to-knot table.

    A *knot* is a point along the orbit that serves as the center of a
    local 5th-order Taylor expansion of the planet's trajectory in time.
    This function distributes ``n_knots`` such expansion centers over a
    single period and records where the dispatch should switch from one
    knot to the next.

    Note the distinction from spline interpolation: spline knots are the
    boundaries where neighbouring polynomial pieces join, whereas these
    knots are the expansion *centers*. The boundaries between adjacent
    knots' regions of validity are returned separately as ``change_times``.

    Parameters
    ----------
    n_knots : int
        Number of knots. Must be odd so that one knot lands at the orbit
        midpoint.
    e : float
        Orbital eccentricity, used by the ``'ea'`` and ``'ta'`` placement
        strategies to cluster knots near periastron.
    quantity : {'mm', 'ea', 'ta'}, optional
        Knot placement strategy: ``'mm'`` spaces knots uniformly in mean
        motion (time), ``'ea'`` in eccentric anomaly (default), ``'ta'``
        in true anomaly.
    tres : int, optional
        Resolution of the time-to-knot lookup table (number of bins per
        period).

    Returns
    -------
    knot_times : ndarray
        Times of the knots (expansion centers), as fractions of the
        orbital period in ``[0, 1]``.
    change_times : ndarray
        Boundary times at which the time-to-knot dispatch switches from
        one knot to the next, i.e. the edges of each knot's region of
        validity (one fewer than ``knot_times``).
    dt : float
        Width of a single time-to-knot table bin, ``1 / tres``.
    tktable : ndarray of int
        Time-to-knot table mapping each of the ``tres`` time bins within
        one folded period to the index of the knot that should evaluate
        it.
    """
    if quantity not in ('mm', 'ea', 'ta'):
        raise ValueError("Quantity needs to be either 'mm' for mean motion, 'ea' for eccentric anomaly, or 'ta' for true anomaly.")
    if n_knots % 2 != 1:
        raise ValueError("Number of knots should be odd.")

    if quantity == 'mm':
        knot_times = linspace(0.0, 1.0, n_knots)
        change_times = 0.5 * (knot_times[:-1] + knot_times[1:])
    else:
        if quantity == 'ea':
            def cfun(t, e, v):
                return eccentric_anomaly(t, e) - v
        elif quantity == 'ta':
            def cfun(t, e, v):
                return true_anomaly(t, e) - v
        else:
            raise NotImplementedError

        knot_sep = 2 * pi / n_knots

        knot_times = zeros(n_knots)
        knot_times[n_knots // 2] = 0.5
        t0 = 1e-5
        for i in range(1, n_knots // 2):
            knot_times[i] = root_scalar(cfun, args=(e, i * knot_sep), bracket=(t0, 1.0 - 1e-5)).root
            t0 = knot_times[i]
        knot_times[n_knots // 2 + 1:-1] = 1 - knot_times[n_knots // 2 - 1:0:-1]
        knot_times[-1] = 1.0

        change_times = zeros(n_knots-1)
        t0 = 1e-5
        for i in range(0, n_knots // 2):
            change_times[i] = root_scalar(cfun, args=(e, (i + 0.5) * knot_sep), bracket=(t0, 1.0 - 1e-5)).root
            t0 = change_times[i]
        change_times[n_knots // 2:] = 1 - change_times[n_knots // 2 - 1::-1]

    # Create the time-to-knot table
    dt = 1 / tres
    tktable = zeros(tres, int)
    ik = 0
    for i in range(tres):
        if i*dt > change_times[ik]:
            ik += 1
        if ik >= n_knots-1:
            tktable[i:] = ik
            break
        tktable[i] = ik

    return knot_times, change_times, dt, tktable
