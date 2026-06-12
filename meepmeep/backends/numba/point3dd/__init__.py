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

"""Single-knot 3D Taylor-series evaluators with parameter derivatives.

Derivative-returning counterpart of the ``point3d`` package: each evaluator
returns its quantity together with the partial derivatives with respect to
the seven orbital parameters ``(tc, p, a, i, e, w, lan)``. The code is
organised one module per physical quantity to mirror the ``orbit3dd``
package layout, plus a ``solve`` module.

Coefficient layout: ``c`` is a ``(3, 5)`` matrix and ``dc`` a ``(7, 3, 5)``
tensor, both produced by ``solve3d_d``. The leading axis of ``dc``
enumerates the seven parameters in the canonical order
``(tc, p, a, i, e, w, lan)``.

This ``__init__`` re-exports the full derivative surface. The multi-knot
gradient dispatchers in ``orbit3dd`` delegate to the single-knot evaluators
defined here.
"""

from ._common import _is_1d_array
from .position import (pos_cd, pos_d, _pos_cd_w, _pos_cd_s, _pos_cd_v, _pos_cd_vp,
                       _pos_d_s, _pos_d_v, _pos_d_vp)
from .zposition import (zpos_cd, zpos_d, _zpos_cd_w, _zpos_cd_s, _zpos_cd_v, _zpos_cd_vp,
                        _zpos_d_s, _zpos_d_v, _zpos_d_vp)
from .separation import (sep_cd, sep_d, _sep_cd_w, _sep_cd_s, _sep_cd_v, _sep_cd_vp,
                         _sep_d_s, _sep_d_v, _sep_d_vp)
from .velocity import vel_cd, _vel_cd_w, _vel_cd_s, _vel_cd_v, _vel_cd_vp
from .zvelocity import (zvel_cd, zvel_d, _zvel_cd_w, _zvel_cd_s, _zvel_cd_v, _zvel_cd_vp,
                        _zvel_d_s, _zvel_d_v, _zvel_d_vp)
from .radial_velocity import (rv_cd, rv_d, _rv_scale, _rv_cd_w, _rv_cd_s, _rv_cd_v, _rv_cd_vp,
                              _rv_d_s, _rv_d_v, _rv_d_vp)
from .solve import solve3d_d
