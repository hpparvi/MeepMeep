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

"""Shared helpers for the single-expansion-point 3D value evaluators.

Holds the Numba array-type predicate used by the scalar-or-array
``@overload`` dispatchers (``pos_c`` / ``pos`` / ``zpos_*`` / ``sep_*`` /
``vel_c`` / ``zvel_*``) to route a 1-D time array to their vector kernels
and a scalar time to their scalar kernels.
"""

from numba import types


def _is_1d_array(typ):
    """True for a 1-D Numba array type (any layout)."""
    return isinstance(typ, types.Array) and typ.ndim == 1
