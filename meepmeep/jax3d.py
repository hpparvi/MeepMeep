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

"""Public low-level 3D Taylor API, JAX backend.

The JAX counterpart of :mod:`meepmeep.numba3d`, bundling in one flat
namespace:

* Single-expansion-point 3D evaluators (``pos``, ``sep``, ``zpos``,
  ``vel``, ``zvel``, ``rv``, ``cos_alpha`` and the phase curves, each with a
  centered ``X_c`` form), the coefficient solver :func:`solve3d`, and the
  transit-geometry utilities.
* Multi-expansion-point, orbit-spanning evaluators (``X_o``) with the same
  signatures as their numba twins, plus :func:`solve3d_orbit` and
  :func:`ep_ix`.
* Expansion-point placement (:func:`create_expansion_points`, traceable in
  the eccentricity), the helpers needed to build models
  (:func:`mean_anomaly_at_transit`, :func:`eccentricity_vector`,
  :func:`ea_from_ma`), and the high-level pytree orbit :class:`JaxOrbit`.

All functions are element-wise over arrays of times and fully traceable.
Parameter gradients come from autodiff: there are no ``_d``/``_od``
variants, no vector/parallel kernels and no gradient-basis transforms. The
basis follows the parameters of the function being differentiated.

Requires double precision: call ``jax.config.update('jax_enable_x64', True)``
before any JAX computation.
"""

from .backends.jax.point3d import (
    pos_c, pos, sep_c, sep, zpos_c, zpos,
    vel_c, vel, zvel_c, zvel, rv_c, rv,
    cos_alpha_c, cos_alpha,
    lambert_phase_curve_c, lambert_phase_curve,
    ev_signal_c, ev_signal,
    emission_phase_curve_c, emission_phase_curve,
)
from .backends.jax.solve import solve3d, solve3d_orbit
from .backends.jax.util import (
    find_contact_point, bounding_box,
    t1, t12, t14, t23, t34, t4,
    find_z_min,
)
from .backends.jax.orbit3d import (
    ep_ix, pos_o, zpos_o, sep_o, vel_o, zvel_o, rv_o, cos_alpha_o,
    star_planet_distance_o, cos_v_p_angle_o, lambert_phase_curve_o,
    ev_signal_o, emission_phase_curve_o, light_travel_time_o, true_anomaly_o,
)
from .backends.jax.expansion_points import create_expansion_points, expansion_table_size
from .backends.jax.newton import ea_from_ma
from .backends.jax.utils import mean_anomaly_at_transit, eccentricity_vector, eclipse_time_offset
from .backends.jax.orbit import JaxOrbit

__all__ = [
    "JaxOrbit",
    "bounding_box",
    "cos_alpha",
    "cos_alpha_c",
    "cos_alpha_o",
    "cos_v_p_angle_o",
    "create_expansion_points",
    "ea_from_ma",
    "eccentricity_vector",
    "eclipse_time_offset",
    "emission_phase_curve",
    "emission_phase_curve_c",
    "emission_phase_curve_o",
    "ep_ix",
    "ev_signal",
    "ev_signal_c",
    "ev_signal_o",
    "expansion_table_size",
    "find_contact_point",
    "find_z_min",
    "lambert_phase_curve",
    "lambert_phase_curve_c",
    "lambert_phase_curve_o",
    "light_travel_time_o",
    "mean_anomaly_at_transit",
    "pos",
    "pos_c",
    "pos_o",
    "rv",
    "rv_c",
    "rv_o",
    "sep",
    "sep_c",
    "sep_o",
    "solve3d",
    "solve3d_orbit",
    "star_planet_distance_o",
    "t1",
    "t12",
    "t14",
    "t23",
    "t34",
    "t4",
    "true_anomaly_o",
    "vel",
    "vel_c",
    "vel_o",
    "zpos",
    "zpos_c",
    "zpos_o",
    "zvel",
    "zvel_c",
    "zvel_o",
]
