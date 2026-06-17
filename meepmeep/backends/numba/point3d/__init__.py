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

"""Single-expansion-point 3D Taylor-series evaluators (non-derivative).

The functions in this package evaluate orbital quantities (3D position,
line-of-sight position, sky-projected separation, velocity, line-of-sight
velocity, radial velocity) and transit geometry (contact points, durations,
minimum separation) from a single 5th-order Taylor expansion around one
expansion point. The code is organised one module per physical quantity to mirror the
``orbit3d`` package layout, plus a ``solve`` module and a ``util`` module.

Coefficient layout: ``c`` is a ``(3, 5)`` matrix produced by ``solve3d``
(rows index the spatial dimensions x, y, z; columns index Taylor order from
position through snap, pre-scaled by the factorial of the order).

This ``__init__`` re-exports the full surface. The multi-expansion-point dispatchers in
``orbit3d`` delegate to the single-expansion-point evaluators defined here.
"""

from ._common import _is_1d_array
from .position import pos_c, pos, _pos_c_s, pos_c_v, pos_c_vp, _pos_s, pos_v, pos_vp
from .zposition import zpos_c, zpos, _zpos_c_s, zpos_c_v, zpos_c_vp, _zpos_s, zpos_v, zpos_vp
from .separation import sep_c, sep, _sep_c_s, sep_c_v, sep_c_vp, _sep_s, sep_v, sep_vp
from .velocity import vel_c, _vel_c_s, vel_c_v, vel_c_vp
from .zvelocity import zvel_c, zvel, _zvel_c_s, zvel_c_v, zvel_c_vp, _zvel_s, zvel_v, zvel_vp
from .radial_velocity import rv_c, rv
from .cos_phase_angle import (
    cos_alpha_c, cos_alpha, _cos_alpha_c_s, cos_alpha_c_v, cos_alpha_c_vp,
    _cos_alpha_s, cos_alpha_v, cos_alpha_vp,
)
from .lambert import (
    _lambert_kernel,
    lambert_phase_curve_c, lambert_phase_curve, _lambert_phase_curve_c_s,
    lambert_phase_curve_c_v, lambert_phase_curve_c_vp,
    _lambert_phase_curve_s, lambert_phase_curve_v, lambert_phase_curve_vp,
)
from .ev_signal import (
    ev_signal_c, ev_signal, _ev_signal_c_s,
    ev_signal_c_v, ev_signal_c_vp,
    _ev_signal_s, ev_signal_v, ev_signal_vp,
)
from .solve import solve3d
from .util import (
    find_contact_point, bounding_box,
    t1, t4, t12, t14, t23, t34,
    find_z_min,
)
