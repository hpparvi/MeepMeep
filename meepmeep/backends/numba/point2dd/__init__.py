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

"""Single-expansion-point 2D Taylor-series evaluators with parameter derivatives.

Derivative-returning counterpart of the ``point2d`` package: each evaluator
returns the sky-plane quantity together with its partial derivatives with
respect to the seven orbital parameters ``(tc, p, a, i, e, w, lan)``. This
is the 2D counterpart of the single-expansion-point 3D derivative modules
(``position3dd``, ``solve3dd``), organised one module per physical quantity
to mirror the ``orbit3dd`` package layout.

Coefficient layout: ``c`` is a ``(2, 5)`` matrix and ``dc`` a ``(7, 2, 5)``
tensor, both produced by ``solve2d_d``. The leading axis of ``dc``
enumerates the seven parameters in the canonical order
``(tc, p, a, i, e, w, lan)``.

This ``__init__`` re-exports the full derivative surface.
"""

from ._common import _is_1d_array
from .position import (pos_cd, pos_d, _pos_cd_w, _pos_cd_s, pos_cd_v, pos_cd_vp,
                       _pos_d_s, pos_d_v, pos_d_vp)
from .separation import (sep_cd, sep_d, _sep_cd_w, _sep_cd_s, sep_cd_v, sep_cd_vp,
                         _sep_d_s, sep_d_v, sep_d_vp)
from .solve import solve2d_d
