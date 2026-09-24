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

"""Public low-level 2D Taylor API, JAX backend.

The JAX counterpart of :mod:`meepmeep.numba2d`: the 2D coefficient
solver, sky-plane position, projected separation and transit-geometry
utilities. The functions are element-wise over arrays of times and fully
traceable (``jit``, ``vmap``, ``grad``). There are no gradient (``_d``,
``_cd``) or vector (``_v``, ``_vp``) variants; differentiate a function
that calls :func:`solve2d` and an evaluator instead.

Requires double precision: call ``jax.config.update('jax_enable_x64', True)``
before any JAX computation. For 3D and multi-expansion-point routines see
:mod:`meepmeep.jax3d`.
"""

from .backends.jax.point2d import pos_c, pos, sep_c, sep
from .backends.jax.solve import solve2d
from .backends.jax.util import (
    find_contact_point, bounding_box,
    t1, t12, t14, t23, t34, t4,
    find_z_min,
)

__all__ = [
    "bounding_box",
    "find_contact_point",
    "find_z_min",
    "pos",
    "pos_c",
    "sep",
    "sep_c",
    "solve2d",
    "t1",
    "t12",
    "t14",
    "t23",
    "t34",
    "t4",
]
