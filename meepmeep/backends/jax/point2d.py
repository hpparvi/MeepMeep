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

"""Single-expansion-point 2D Taylor evaluators.

JAX ports of the ``meepmeep.backends.numba.point2d`` package, for the
``(2, 5)`` coefficient matrices of
:func:`~meepmeep.jax2d.solve2d`. See
``meepmeep.backends.jax.point3d`` for the conventions. Under ``jit``
the 2D surface is mostly a convenience: XLA drops the unused z row of a 3D
solve, so the 3D functions cost about the same.
"""

from ._common import horner, fold
from .point3d import sep_c, sep


def pos_c(time, c):
    """Sky-plane position ``(x, y)`` [R_star] at time relative to the expansion point."""
    return horner(time, c, 0), horner(time, c, 1)


def pos(time, tc, p, c, te=0.0):
    """Sky-plane position ``(x, y)`` [R_star] at absolute time.

    Parameters
    ----------
    time : float or NDArray
        Time [days].
    tc : float
        Transit-centre time [days].
    p : float
        Orbital period [days].
    c : NDArray, shape (2, 5)
        Coefficients from :func:`~meepmeep.jax2d.solve2d`.
    te : float, optional
        Expansion-point offset from the transit centre, the same value given
        to the solver. Defaults to 0.0.
    """
    return pos_c(fold(time, tc, p, te), c)


__all__ = ["pos_c", "pos", "sep_c", "sep"]
