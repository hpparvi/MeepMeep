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

* Single-knot 3D Taylor evaluators from
  ``meepmeep.backends.numba.taylor.position3d`` /
  ``velocity3d`` / ``solve3d`` / ``util3d`` and their
  parameter-derivative counterparts in ``*dd`` modules.
* Multi-knot orbit-spanning evaluators from
  ``meepmeep.backends.numba.taylor.orbit3d`` and ``orbit3dd``
  (the ``*_os`` / ``*_ov`` / ``*_osd`` / ``*_ovd`` family).
* Dimension-agnostic primitives from
  ``meepmeep.backends.numba.knots`` / ``newton.newton`` / ``utils``.
  Anomaly utilities in ``knots`` (``eccentric_anomaly``,
  ``true_anomaly``) are intentionally not re-exported here to avoid
  visual confusion with the ``true_anomaly_ov`` evaluator family; use
  ``meepmeep.backends.numba.knots`` if you need them.

For the 2D surface see :mod:`meepmeep.numba2d`.
"""

# --- 3D single-knot Taylor primitives ---------------------------------
from .backends.numba.taylor.position3d import (
    pos_c, pos, sep_c, sep, pos_and_sep_c, pos_and_sep, pz_c, pz,
)
from .backends.numba.taylor.position3dd import (
    pos_cd, pos_d, sep_cd, sep_d, pz_cd, pz_d,
)
from .backends.numba.taylor.velocity3d import (
    vel_c, zvel_c, zvel, rv_c, rv,
)
from .backends.numba.taylor.velocity3dd import (
    vel_cd, zvel_cd, zvel_d, rv_cd, rv_d,
)
from .backends.numba.taylor.solve3d import solve3d
from .backends.numba.taylor.solve3dd import solve3d_d
from .backends.numba.taylor.util3d import (
    find_contact_point, bounding_box,
    t1, t12, t14, t23, t34, t4,
    find_z_min,
)

# --- Multi-knot orbit-spanning evaluators -----------------------------
from .backends.numba.taylor.orbit3d import (
    solve3d_orbit, knot_ix,
    pos_os, pos_ov,
    zpos_os, zpos_ov,
    sep_os,
    vel_os, vel_ov,
    zvel_os, zvel_ov,
    true_anomaly_ov,
    cos_v_p_angle_ov,
    cos_alpha_os, cos_alpha_ov,
    star_planet_distance_ov,
    lambert_phase_curve_os, lambert_phase_curve_ov,
    lambert_and_emission_ov,
    ev_signal_ov,
    rv_ov,
    light_travel_time_os, light_travel_time_ov,
)
from .backends.numba.taylor.orbit3dd import (
    solve3d_orbit_d,
    pos_osd, pos_ovd,
    zpos_osd, zpos_ovd,
    sep_osd,
    vel_osd, vel_ovd,
    zvel_osd, zvel_ovd,
    true_anomaly_ovd,
    cos_v_p_angle_ovd,
    cos_alpha_osd, cos_alpha_ovd,
    star_planet_distance_ovd,
    lambert_phase_curve_osd, lambert_phase_curve_ovd,
    lambert_and_emission_ovd,
    ev_signal_ovd,
    rv_ovd,
    light_travel_time_osd, light_travel_time_ovd,
)

# --- Dimension-agnostic primitives ------------------------------------
from .backends.numba.knots import create_knots
from .backends.numba.newton.newton import (
    ea_from_ma,
    ea_newton_s, ea_newton_v,
    ta_newton_s, ta_newton_v,
    xy_newton_v, xyz_newton_v,
    z_newton_s, z_newton_v,
    rv_newton_v,
    eclipse_light_travel_time,
)
from .backends.numba.utils import (
    TWO_PI,
    eccentricity_vector, eclipse_time_offset, transit_distance_factor,
    i_from_baew, as_from_rhop,
    ta_from_ea, z_from_ta,
    mean_anomaly, mean_anomaly_with_derivatives,
    mean_anomaly_at_transit, mean_anomaly_at_transit_with_derivatives,
    impact_parameter, impact_parameter_ec,
    d_from_pkaiews,
)

__all__ = [
    "TWO_PI",
    "as_from_rhop",
    "bounding_box",
    "cos_alpha_os",
    "cos_alpha_osd",
    "cos_alpha_ov",
    "cos_alpha_ovd",
    "cos_v_p_angle_ov",
    "cos_v_p_angle_ovd",
    "create_knots",
    "d_from_pkaiews",
    "ea_from_ma",
    "ea_newton_s",
    "ea_newton_v",
    "eccentricity_vector",
    "eclipse_light_travel_time",
    "eclipse_time_offset",
    "ev_signal_ov",
    "ev_signal_ovd",
    "find_contact_point",
    "find_z_min",
    "i_from_baew",
    "impact_parameter",
    "impact_parameter_ec",
    "knot_ix",
    "lambert_and_emission_ov",
    "lambert_and_emission_ovd",
    "lambert_phase_curve_os",
    "lambert_phase_curve_osd",
    "lambert_phase_curve_ov",
    "lambert_phase_curve_ovd",
    "light_travel_time_os",
    "light_travel_time_osd",
    "light_travel_time_ov",
    "light_travel_time_ovd",
    "mean_anomaly",
    "mean_anomaly_at_transit",
    "mean_anomaly_at_transit_with_derivatives",
    "mean_anomaly_with_derivatives",
    "pos",
    "pos_and_sep",
    "pos_and_sep_c",
    "pos_c",
    "pos_cd",
    "pos_d",
    "pos_os",
    "pos_osd",
    "pos_ov",
    "pos_ovd",
    "pz",
    "pz_c",
    "pz_cd",
    "pz_d",
    "rv",
    "rv_c",
    "rv_cd",
    "rv_d",
    "rv_newton_v",
    "rv_ov",
    "rv_ovd",
    "sep",
    "sep_c",
    "sep_cd",
    "sep_d",
    "sep_os",
    "sep_osd",
    "solve3d",
    "solve3d_d",
    "solve3d_orbit",
    "solve3d_orbit_d",
    "star_planet_distance_ov",
    "star_planet_distance_ovd",
    "t1",
    "t12",
    "t14",
    "t23",
    "t34",
    "t4",
    "ta_from_ea",
    "ta_newton_s",
    "ta_newton_v",
    "transit_distance_factor",
    "true_anomaly_ov",
    "true_anomaly_ovd",
    "vel_c",
    "vel_cd",
    "vel_os",
    "vel_osd",
    "vel_ov",
    "vel_ovd",
    "xy_newton_v",
    "xyz_newton_v",
    "z_from_ta",
    "z_newton_s",
    "z_newton_v",
    "zpos_os",
    "zpos_osd",
    "zpos_ov",
    "zpos_ovd",
    "zvel",
    "zvel_c",
    "zvel_cd",
    "zvel_d",
    "zvel_os",
    "zvel_osd",
    "zvel_ov",
    "zvel_ovd",
]
