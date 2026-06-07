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
"""Public low-level 3D Taylor API.

This module is the canonical public entry point to MeepMeep's 3D
Numba-jitted primitives. It bundles three layers in one flat namespace:

* Single-knot 3D Taylor evaluators from the
  ``meepmeep.backends.numba.point3d`` package (per-quantity
  ``position`` / ``zposition`` / ``separation`` / ``velocity`` /
  ``zvelocity`` / ``radial_velocity`` modules plus ``solve`` and
  ``util``) and their parameter-derivative counterparts in the
  ``point3dd`` package.
* Multi-knot orbit-spanning evaluators from
  ``meepmeep.backends.numba.orbit3d`` and ``orbit3dd`` — exposed
  as unified ``*_o`` (forward) and ``*_od`` (with gradients)
  dispatchers that accept either a scalar time or a 1-D float64 array
  of times and route at compile time to the appropriate
  scalar/vector kernel.
* Dimension-agnostic primitives from
  ``meepmeep.backends.numba.knots`` / ``newton.newton`` / ``utils``.
  Anomaly utilities in ``knots`` (``eccentric_anomaly``,
  ``true_anomaly``) are intentionally not re-exported here to avoid
  visual confusion with the ``true_anomaly_o`` evaluator family; use
  ``meepmeep.backends.numba.knots`` if you need them.

For the 2D surface see :mod:`meepmeep.numba2d`.
"""

# --- 3D single-knot Taylor primitives ---------------------------------
from .backends.numba.point3d import (
    pos_c, pos, sep_c, sep, pos_and_sep_c, pos_and_sep, zpos_c, zpos,
    vel_c, zvel_c, zvel, rv_c, rv,
    solve3d,
    find_contact_point, bounding_box,
    t1, t12, t14, t23, t34, t4,
    find_z_min,
)
from .backends.numba.point3dd import (
    pos_cd, pos_d, sep_cd, sep_d, zpos_cd, zpos_d,
    vel_cd, zvel_cd, zvel_d, rv_cd, rv_d,
    solve3d_d,
)

# --- Multi-knot orbit-spanning evaluators -----------------------------
from .backends.numba.orbit3d import (
    solve3d_orbit, knot_ix,
    pos_o, zpos_o, sep_o,
    vel_o, zvel_o,
    true_anomaly_o, cos_v_p_angle_o, cos_alpha_o,
    star_planet_distance_o,
    lambert_phase_curve_o, lambert_and_emission_o,
    ev_signal_o, rv_o, light_travel_time_o,
)
from .backends.numba.orbit3dd import (
    solve3d_orbit_d,
    pos_od, zpos_od, sep_od,
    vel_od, zvel_od,
    cos_alpha_od, star_planet_distance_od,
    cos_v_p_angle_od, true_anomaly_od,
    lambert_phase_curve_od, lambert_and_emission_od,
    ev_signal_od, rv_od, light_travel_time_od,
)

# --- Dimension-agnostic primitives ------------------------------------
from .backends.numba.knots import create_knots
from .backends.numba.utils import tc_to_tp_gradient

__all__ = [
    "bounding_box",
    "cos_alpha_o",
    "cos_alpha_od",
    "cos_v_p_angle_o",
    "cos_v_p_angle_od",
    "create_knots",
    "ev_signal_o",
    "ev_signal_od",
    "find_contact_point",
    "find_z_min",
    "knot_ix",
    "lambert_and_emission_o",
    "lambert_and_emission_od",
    "lambert_phase_curve_o",
    "lambert_phase_curve_od",
    "light_travel_time_o",
    "light_travel_time_od",
    "pos",
    "pos_and_sep",
    "pos_and_sep_c",
    "pos_c",
    "pos_cd",
    "pos_d",
    "pos_o",
    "pos_od",
    "rv",
    "rv_c",
    "rv_cd",
    "rv_d",
    "rv_o",
    "rv_od",
    "sep",
    "sep_c",
    "sep_cd",
    "sep_d",
    "sep_o",
    "sep_od",
    "solve3d",
    "solve3d_d",
    "solve3d_orbit",
    "solve3d_orbit_d",
    "star_planet_distance_o",
    "star_planet_distance_od",
    "t1",
    "t12",
    "t14",
    "t23",
    "t34",
    "t4",
    "tc_to_tp_gradient",
    "true_anomaly_o",
    "true_anomaly_od",
    "vel_c",
    "vel_cd",
    "vel_o",
    "vel_od",
    "zpos",
    "zpos_c",
    "zpos_cd",
    "zpos_d",
    "zpos_o",
    "zpos_od",
    "zvel",
    "zvel_c",
    "zvel_cd",
    "zvel_d",
    "zvel_o",
    "zvel_od",
]
