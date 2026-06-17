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

"""Single-expansion-point 3D Taylor-series evaluators with parameter derivatives.

Derivative-returning counterpart of the ``point3d`` package: each evaluator
returns its quantity together with the partial derivatives with respect to
the seven orbital parameters ``(tc, p, a, i, e, w, lan)``. The code is
organised one module per physical quantity to mirror the ``orbit3dd``
package layout, plus a ``solve`` module.

Coefficient layout: ``c`` is a ``(3, 5)`` matrix and ``dc`` a ``(7, 3, 5)``
tensor, both produced by ``solve3d_d``. The leading axis of ``dc``
enumerates the seven parameters in the canonical order
``(tc, p, a, i, e, w, lan)``.

This ``__init__`` re-exports the full derivative surface. The multi-expansion-point
gradient dispatchers in ``orbit3dd`` delegate to the single-expansion-point evaluators
defined here.
"""

from ._common import _is_1d_array
from .position import (pos_cd, pos_d, _pos_cd_w, _pos_cd_s, pos_cd_v, pos_cd_vp,
                       _pos_d_s, pos_d_v, pos_d_vp)
from .zposition import (zpos_cd, zpos_d, _zpos_cd_w, _zpos_cd_s, zpos_cd_v, zpos_cd_vp,
                        _zpos_d_s, zpos_d_v, zpos_d_vp)
from .separation import (sep_cd, sep_d, _sep_cd_w, _sep_cd_s, sep_cd_v, sep_cd_vp,
                         _sep_d_s, sep_d_v, sep_d_vp)
from .velocity import vel_cd, _vel_cd_w, _vel_cd_s, vel_cd_v, vel_cd_vp
from .zvelocity import (zvel_cd, zvel_d, _zvel_cd_w, _zvel_cd_s, zvel_cd_v, zvel_cd_vp,
                        _zvel_d_s, zvel_d_v, zvel_d_vp)
from .radial_velocity import (rv_cd, rv_d, _rv_scale, _rv_cd_w, _rv_cd_s, rv_cd_v, rv_cd_vp,
                              _rv_d_s, rv_d_v, rv_d_vp)
from .cos_phase_angle import (cos_alpha_cd, cos_alpha_d, _cos_alpha_cd_w, _cos_alpha_cd_s,
                          cos_alpha_cd_v, cos_alpha_cd_vp, _cos_alpha_d_s, cos_alpha_d_v, cos_alpha_d_vp)
from .lambert import (_lambert_kernel_d, lambert_phase_curve_cd, lambert_phase_curve_d,
                      _lambert_phase_curve_cd_w, _lambert_phase_curve_cd_s,
                      lambert_phase_curve_cd_v, lambert_phase_curve_cd_vp,
                      _lambert_phase_curve_d_s, lambert_phase_curve_d_v, lambert_phase_curve_d_vp)
from .solve import solve3d_d
