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

"""Single-expansion-point 2D Taylor-series evaluators (non-derivative).

The functions in this package evaluate sky-plane quantities (position,
projected separation) and transit geometry (contact points, durations,
minimum separation) from a single 5th-order Taylor expansion around one
expansion point. They form the 2D counterpart of the single-expansion-point 3D modules
(``position3d``, ``solve3d``, ``util3d``), organised one module per
physical quantity to mirror the ``orbit3d`` package layout.

Coefficient layout: ``c`` is a ``(2, 5)`` matrix produced by ``solve2d``
(rows index the spatial dimensions x, y; columns index Taylor order from
position through snap, pre-scaled by the factorial of the order).

Each physical quantity lives in its own module, holding that quantity's
centered (``X_c``) and direct (``X``) evaluators. The ``solve`` module
builds the coefficient matrix and ``util`` holds the transit-geometry
helpers. This ``__init__`` re-exports the full surface.
"""

from ._common import _is_1d_array
from .position import pos_c, pos, _pos_c_s, _pos_c_v, _pos_c_vp, _pos_s, _pos_v, _pos_vp
from .separation import sep_c, sep, _sep_c_s, _sep_c_v, _sep_c_vp, _sep_s, _sep_v, _sep_vp
from .solve import solve2d
from .util import (
    find_contact_point, bounding_box,
    t1, t4, t12, t14, t23, t34,
    find_z_min,
)
