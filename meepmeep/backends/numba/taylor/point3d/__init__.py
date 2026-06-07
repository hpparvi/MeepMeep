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

"""Single-knot 3D Taylor-series evaluators (non-derivative).

The functions in this package evaluate orbital quantities (3D position,
line-of-sight position, sky-projected separation, velocity, line-of-sight
velocity, radial velocity) and transit geometry (contact points, durations,
minimum separation) from a single 5th-order Taylor expansion around one
knot. The code is organised one module per physical quantity to mirror the
``orbit3d`` package layout, plus a ``solve`` module and a ``util`` module.

Coefficient layout: ``c`` is a ``(3, 5)`` matrix produced by ``solve3d``
(rows index the spatial dimensions x, y, z; columns index Taylor order from
position through snap, pre-scaled by the factorial of the order).

This ``__init__`` re-exports the full surface. The multi-knot dispatchers in
``orbit3d`` delegate to the single-knot evaluators defined here.
"""

from .position import pos_c, pos, pos_and_sep_c, pos_and_sep
from .zposition import zpos_c, zpos
from .separation import sep_c, sep
from .velocity import vel_c
from .zvelocity import zvel_c, zvel
from .radial_velocity import rv_c, rv
from .solve import solve3d
from .util import (
    find_contact_point, bounding_box,
    t1, t4, t12, t14, t23, t34,
    find_z_min,
)
