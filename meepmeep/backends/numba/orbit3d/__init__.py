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

"""Multi-knot Taylor-series evaluators over a full orbit.

The functions in this package evaluate orbit-spanning quantities (position,
velocity, projected separation, phase angle, radial velocity, etc.) at
arbitrary times by looking up the appropriate knot via ``pktable`` and then
delegating to the single-knot evaluators in ``position3d``/``velocity3d``.

Coefficient layout: ``coeffs`` is an ``(npt, 3, 5)`` array as produced by
``solve3d_orbit`` — ``coeffs[ix]`` is the ``(3, 5)`` matrix consumed by
``pos_c``, ``vel_c``, ``zvel_c``, ``sep_c``, and ``zpos_c``.

Each physical quantity lives in its own module, holding that quantity's
scalar kernel (``_X_os``), vector kernel (``_X_ov``), public dispatcher
(``X_o``), and Numba ``@overload`` registration. Shared infrastructure
(``_is_1d_array``, ``knot_ix``, ``solve3d_orbit``) lives in ``_common``.
This ``__init__`` re-exports the full surface as the package's public API.
"""

from ._common import _is_1d_array, solve3d_orbit, knot_ix
from .position import pos_o, _pos_os, _pos_ov
from .zposition import zpos_o, _zpos_os, _zpos_ov
from .separation import sep_o, _sep_os, _sep_ov
from .velocity import vel_o, _vel_os, _vel_ov
from .zvelocity import zvel_o, _zvel_os, _zvel_ov
from .true_anomaly import true_anomaly_o, _true_anomaly_os, _true_anomaly_ov
from .projected_angle import cos_v_p_angle_o, _cos_v_p_angle_os, _cos_v_p_angle_ov
from .phase_angle import cos_alpha_o, _cos_alpha_os, _cos_alpha_ov
from .star_planet_distance import (
    star_planet_distance_o, _star_planet_distance_os, _star_planet_distance_ov,
)
from .lambert import (
    _lambert_kernel,
    lambert_phase_curve_o, _lambert_phase_curve_os, _lambert_phase_curve_ov,
    lambert_and_emission_o, _lambert_and_emission_os, _lambert_and_emission_ov,
)
from .ev_signal import ev_signal_o, _ev_signal_os, _ev_signal_ov
from .radial_velocity import rv_o, _rv_os, _rv_ov
from .light_travel_time import (
    LTT_DAYS_PER_RSUN,
    light_travel_time_o, _light_travel_time_os, _light_travel_time_ov,
)
