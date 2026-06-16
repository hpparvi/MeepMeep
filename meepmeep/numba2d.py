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
"""Public low-level 2D Taylor API.

This module is the canonical public entry point to MeepMeep's
2D Numba-jitted primitives: position, sky-projected separation, the
Taylor coefficient solver, transit contact-point utilities, and their
parameter-derivative variants. All names are re-exported verbatim from
``meepmeep.backends.numba.*`` so user ``@njit`` kernels and
direct callers can import them from one place.

Each scalar-or-array dispatcher (``pos``, ``sep``, ``pos_d``, ...) also
has explicit vector (``pos_v``, ``sep_v``, ``sep_d_v``, ...) and parallel
(``pos_vp``, ``sep_vp``, ...) kernels. Call these directly to commit to
the array path and skip the dispatcher's scalar-or-array type check; the
parallel twins multi-thread the sample loop and pay off only for large
time grids (see :class:`meepmeep.expansion2d.Expansion2D`'s ``parallel``
flag for the size thresholds).

For 3D Taylor primitives and multi-expansion-point orbit-spanning
routines see :mod:`meepmeep.numba3d`. Dimension-agnostic primitives
(expansion points, Newton solvers, orbital-mechanics utilities) are
re-exported only from :mod:`meepmeep.numba3d`; 2D users who need them
should import from ``meepmeep.backends.numba.*`` directly.
"""

from .backends.numba.point2d import (
    pos_c, pos, sep_c, sep,
    pos_c_v, pos_c_vp, pos_v, pos_vp,
    sep_c_v, sep_c_vp, sep_v, sep_vp,
    solve2d,
    find_contact_point, bounding_box,
    t1, t12, t14, t23, t34, t4,
    find_z_min,
)
from .backends.numba.point2dd import (
    pos_cd, pos_d, sep_cd, sep_d,
    pos_cd_v, pos_cd_vp, pos_d_v, pos_d_vp,
    sep_cd_v, sep_cd_vp, sep_d_v, sep_d_vp,
    solve2d_d,
)

__all__ = [
    "bounding_box",
    "find_contact_point",
    "find_z_min",
    "pos",
    "pos_c",
    "pos_c_v",
    "pos_c_vp",
    "pos_cd",
    "pos_cd_v",
    "pos_cd_vp",
    "pos_d",
    "pos_d_v",
    "pos_d_vp",
    "pos_v",
    "pos_vp",
    "sep",
    "sep_c",
    "sep_c_v",
    "sep_c_vp",
    "sep_cd",
    "sep_cd_v",
    "sep_cd_vp",
    "sep_d",
    "sep_d_v",
    "sep_d_vp",
    "sep_v",
    "sep_vp",
    "solve2d",
    "solve2d_d",
    "t1",
    "t12",
    "t14",
    "t23",
    "t34",
    "t4",
]
