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

* Single-expansion-point 3D Taylor evaluators from the
  ``meepmeep.backends.numba.point3d`` package (per-quantity
  ``position`` / ``zposition`` / ``separation`` / ``velocity`` /
  ``zvelocity`` / ``radial_velocity`` modules plus ``solve`` and
  ``util``) and their parameter-derivative counterparts in the
  ``point3dd`` package.
* Multi-expansion-point orbit-spanning evaluators from
  ``meepmeep.backends.numba.orbit3d`` and ``orbit3dd`` — exposed
  as unified ``*_o`` (forward) and ``*_od`` (with gradients)
  dispatchers that accept either a scalar time or a 1-D float64 array
  of times and route at compile time to the appropriate
  scalar/vector kernel.
* Dimension-agnostic primitives from
  ``meepmeep.backends.numba.expansion_points`` / ``newton.newton`` /
  ``utils``. Anomaly utilities in ``expansion_points``
  (``eccentric_anomaly``, ``true_anomaly``) are intentionally not
  re-exported here to avoid visual confusion with the
  ``true_anomaly_o`` evaluator family; use
  ``meepmeep.backends.numba.expansion_points`` if you need them.

Both the single-expansion-point dispatchers (``pos``, ``sep``, ``rv_d``, ...) and the
multi-expansion-point dispatchers (``pos_o``, ``sep_od``, ...) also expose their
underlying vector and parallel kernels: ``X_v`` / ``X_vp`` (single
expansion point) and ``X_ov`` / ``X_ovp`` (multi-expansion-point values),
``X_ovd`` / ``X_ovdp`` (multi-expansion-point gradients). Call these directly
to commit to the array path and skip the dispatcher's scalar-or-array
type check; the parallel twins multi-thread the sample loop and pay off
only for large time grids (see :class:`meepmeep.orbit.Orbit`'s
``parallel`` flag for the size thresholds). The non-derivative 3D radial
velocity (``rv_c`` / ``rv``) is scalar-inline only and has no single-expansion-point
vector kernel.

For the 2D surface see :mod:`meepmeep.numba2d`.
"""

# --- 3D single-expansion-point Taylor primitives ---------------------------------
from .backends.numba.point3d import (
    pos_c, pos, sep_c, sep, zpos_c, zpos,
    vel_c, zvel_c, zvel, rv_c, rv,
    pos_c_v, pos_c_vp, pos_v, pos_vp,
    sep_c_v, sep_c_vp, sep_v, sep_vp,
    zpos_c_v, zpos_c_vp, zpos_v, zpos_vp,
    vel_c_v, vel_c_vp,
    zvel_c_v, zvel_c_vp, zvel_v, zvel_vp,
    solve3d,
    find_contact_point, bounding_box,
    t1, t12, t14, t23, t34, t4,
    find_z_min,
)
from .backends.numba.point3dd import (
    pos_cd, pos_d, sep_cd, sep_d, zpos_cd, zpos_d,
    vel_cd, zvel_cd, zvel_d, rv_cd, rv_d,
    pos_cd_v, pos_cd_vp, pos_d_v, pos_d_vp,
    sep_cd_v, sep_cd_vp, sep_d_v, sep_d_vp,
    zpos_cd_v, zpos_cd_vp, zpos_d_v, zpos_d_vp,
    vel_cd_v, vel_cd_vp,
    zvel_cd_v, zvel_cd_vp, zvel_d_v, zvel_d_vp,
    rv_cd_v, rv_cd_vp, rv_d_v, rv_d_vp,
    solve3d_d,
)

# --- Multi-expansion-point orbit-spanning evaluators ------------------
from .backends.numba.orbit3d import (
    solve3d_orbit, ep_ix,
    pos_o, zpos_o, sep_o,
    vel_o, zvel_o,
    true_anomaly_o, cos_v_p_angle_o, cos_alpha_o,
    star_planet_distance_o,
    lambert_phase_curve_o,
    ev_signal_o, rv_o, light_travel_time_o,
    pos_ov, pos_ovp, zpos_ov, zpos_ovp, sep_ov, sep_ovp,
    vel_ov, vel_ovp, zvel_ov, zvel_ovp,
    true_anomaly_ov, true_anomaly_ovp,
    cos_v_p_angle_ov, cos_v_p_angle_ovp, cos_alpha_ov, cos_alpha_ovp,
    star_planet_distance_ov, star_planet_distance_ovp,
    lambert_phase_curve_ov, lambert_phase_curve_ovp,
    ev_signal_ov, ev_signal_ovp, rv_ov, rv_ovp,
    light_travel_time_ov, light_travel_time_ovp,
)
from .backends.numba.orbit3dd import (
    solve3d_orbit_d,
    pos_od, zpos_od, sep_od,
    vel_od, zvel_od,
    cos_alpha_od, star_planet_distance_od,
    cos_v_p_angle_od, true_anomaly_od,
    lambert_phase_curve_od,
    ev_signal_od, rv_od, light_travel_time_od,
    pos_ovd, pos_ovdp, zpos_ovd, zpos_ovdp, sep_ovd, sep_ovdp,
    vel_ovd, vel_ovdp, zvel_ovd, zvel_ovdp,
    true_anomaly_ovd, true_anomaly_ovdp,
    cos_v_p_angle_ovd, cos_v_p_angle_ovdp, cos_alpha_ovd, cos_alpha_ovdp,
    star_planet_distance_ovd, star_planet_distance_ovdp,
    lambert_phase_curve_ovd, lambert_phase_curve_ovdp,
    ev_signal_ovd, ev_signal_ovdp, rv_ovd, rv_ovdp,
    light_travel_time_ovd, light_travel_time_ovdp,
)

# --- Dimension-agnostic primitives ------------------------------------
from .backends.numba.expansion_points import create_expansion_points
from .backends.numba.utils import tc_to_tp_gradient

__all__ = [
    "bounding_box",
    "cos_alpha_o",
    "cos_alpha_od",
    "cos_alpha_ov",
    "cos_alpha_ovd",
    "cos_alpha_ovdp",
    "cos_alpha_ovp",
    "cos_v_p_angle_o",
    "cos_v_p_angle_od",
    "cos_v_p_angle_ov",
    "cos_v_p_angle_ovd",
    "cos_v_p_angle_ovdp",
    "cos_v_p_angle_ovp",
    "create_expansion_points",
    "ep_ix",
    "ev_signal_o",
    "ev_signal_od",
    "ev_signal_ov",
    "ev_signal_ovd",
    "ev_signal_ovdp",
    "ev_signal_ovp",
    "find_contact_point",
    "find_z_min",
    "lambert_phase_curve_o",
    "lambert_phase_curve_od",
    "lambert_phase_curve_ov",
    "lambert_phase_curve_ovd",
    "lambert_phase_curve_ovdp",
    "lambert_phase_curve_ovp",
    "light_travel_time_o",
    "light_travel_time_od",
    "light_travel_time_ov",
    "light_travel_time_ovd",
    "light_travel_time_ovdp",
    "light_travel_time_ovp",
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
    "pos_o",
    "pos_od",
    "pos_ov",
    "pos_ovd",
    "pos_ovdp",
    "pos_ovp",
    "pos_v",
    "pos_vp",
    "rv",
    "rv_c",
    "rv_cd",
    "rv_cd_v",
    "rv_cd_vp",
    "rv_d",
    "rv_d_v",
    "rv_d_vp",
    "rv_o",
    "rv_od",
    "rv_ov",
    "rv_ovd",
    "rv_ovdp",
    "rv_ovp",
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
    "sep_o",
    "sep_od",
    "sep_ov",
    "sep_ovd",
    "sep_ovdp",
    "sep_ovp",
    "sep_v",
    "sep_vp",
    "solve3d",
    "solve3d_d",
    "solve3d_orbit",
    "solve3d_orbit_d",
    "star_planet_distance_o",
    "star_planet_distance_od",
    "star_planet_distance_ov",
    "star_planet_distance_ovd",
    "star_planet_distance_ovdp",
    "star_planet_distance_ovp",
    "t1",
    "t12",
    "t14",
    "t23",
    "t34",
    "t4",
    "tc_to_tp_gradient",
    "true_anomaly_o",
    "true_anomaly_od",
    "true_anomaly_ov",
    "true_anomaly_ovd",
    "true_anomaly_ovdp",
    "true_anomaly_ovp",
    "vel_c",
    "vel_c_v",
    "vel_c_vp",
    "vel_cd",
    "vel_cd_v",
    "vel_cd_vp",
    "vel_o",
    "vel_od",
    "vel_ov",
    "vel_ovd",
    "vel_ovdp",
    "vel_ovp",
    "zpos",
    "zpos_c",
    "zpos_c_v",
    "zpos_c_vp",
    "zpos_cd",
    "zpos_cd_v",
    "zpos_cd_vp",
    "zpos_d",
    "zpos_d_v",
    "zpos_d_vp",
    "zpos_o",
    "zpos_od",
    "zpos_ov",
    "zpos_ovd",
    "zpos_ovdp",
    "zpos_ovp",
    "zpos_v",
    "zpos_vp",
    "zvel",
    "zvel_c",
    "zvel_c_v",
    "zvel_c_vp",
    "zvel_cd",
    "zvel_cd_v",
    "zvel_cd_vp",
    "zvel_d",
    "zvel_d_v",
    "zvel_d_vp",
    "zvel_o",
    "zvel_od",
    "zvel_ov",
    "zvel_ovd",
    "zvel_ovdp",
    "zvel_ovp",
    "zvel_v",
    "zvel_vp",
]
