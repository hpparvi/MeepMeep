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

"""JAX backend.

Element-wise JAX ports of the Numba backend's value surface: the Taylor
coefficient solvers, the single- and multi-expansion-point evaluators, the
transit-geometry utilities, expansion-point placement, and the exact
Newton-Raphson references. Parameter gradients come from JAX autodiff
rather than from hand-derived ``_d`` kernels.

Import the public surface from :mod:`meepmeep.jax2d` and
:mod:`meepmeep.jax3d`; the module layout here is implementation detail.
The backend needs ``jax`` (the optional ``jax`` dependency group) and
double precision (``jax.config.update('jax_enable_x64', True)``).
"""
