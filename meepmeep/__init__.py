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

"""MeepMeep: fast Keplerian orbits for exoplanet modelling.

The top-level package exports the high-level classes (:class:`Orbit`,
:class:`Expansion2D`, :class:`Expansion3D`) and
:func:`eclipse_light_travel_time`. The low-level Taylor primitives live in
:mod:`meepmeep.numba2d` / :mod:`meepmeep.numba3d` and their JAX twins
:mod:`meepmeep.jax2d` / :mod:`meepmeep.jax3d`.
"""


from .version import __version__
from .orbit import Orbit
from .expansion2d import Expansion2D
from .expansion3d import Expansion3D
from .backends.numba.newton.newton import eclipse_light_travel_time

__all__ = ["Orbit", "Expansion2D", "Expansion3D", "eclipse_light_travel_time", "__version__"]