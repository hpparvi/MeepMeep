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

"""Multi-knot Taylor-series evaluators with parameter derivatives.

Derivative-returning counterparts of the routines in the ``orbit3d``
package. Every function returns both the value and its partial derivatives
with respect to the orbital parameters ``(tc, p, a, i, e, w, lan)`` and any
extra physical inputs the routine takes (appended to the orbital block in
argument order).

Coefficient layout:
- ``coeffs`` : ``(npt, 3, 5)`` — Taylor coefficients, as in ``orbit3d``.
- ``dcoeffs`` : ``(npt, 7, 3, 5)`` — derivatives of the Taylor coefficients
  w.r.t. the 6 orbital parameters, produced by ``solve3d_orbit_d``.

Vector evaluators (``*_ovd``) return per-coordinate derivative arrays of
shape ``(N, ndp)`` where ``ndp`` is ``7`` for orbital-only routines and
``7 + n_extra`` for routines with extra physical inputs.

Each physical quantity lives in its own module, holding that quantity's
scalar kernel (``_X_osd``), vector kernel (``_X_ovd``), public dispatcher
(``X_od``), and Numba ``@overload`` registration. Shared infrastructure
(``_is_1d_array``, ``solve3d_orbit_d``) lives in ``_common``. This
``__init__`` re-exports the full surface so the historical
``...taylor.orbit3dd`` import path keeps working unchanged.
"""

from ._common import _is_1d_array, solve3d_orbit_d
from .position import pos_od, _pos_osd, _pos_ovd
from .zposition import zpos_od, _zpos_osd, _zpos_ovd
from .separation import sep_od, _sep_osd, _sep_ovd
from .velocity import vel_od, _vel_osd, _vel_ovd
from .zvelocity import zvel_od, _zvel_osd, _zvel_ovd
from .phase_angle import cos_alpha_od, _cos_alpha_osd, _cos_alpha_ovd
from .star_planet_distance import (
    star_planet_distance_od, _star_planet_distance_osd, _star_planet_distance_ovd,
)
from .projected_angle import cos_v_p_angle_od, _cos_v_p_angle_osd, _cos_v_p_angle_ovd
from .true_anomaly import true_anomaly_od, _true_anomaly_osd, _true_anomaly_ovd
from .lambert import (
    _lambert_kernel_d,
    lambert_phase_curve_od, _lambert_phase_curve_osd, _lambert_phase_curve_ovd,
    lambert_and_emission_od, _lambert_and_emission_osd, _lambert_and_emission_ovd,
)
from .ev_signal import ev_signal_od, _ev_signal_osd, _ev_signal_ovd
from .radial_velocity import rv_od, _rv_osd, _rv_ovd
from .light_travel_time import (
    LTT_DAYS_PER_RSUN,
    light_travel_time_od, _light_travel_time_osd, _light_travel_time_ovd,
)
