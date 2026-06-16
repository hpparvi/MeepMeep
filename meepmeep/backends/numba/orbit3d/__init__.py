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

"""Multi-expansion-point Taylor-series evaluators over a full orbit.

The functions in this package evaluate orbit-spanning quantities (position,
velocity, projected separation, phase angle, radial velocity, etc.) at
arbitrary times by looking up the appropriate expansion point via ``ep_table`` and then
delegating to the single-expansion-point evaluators in ``position3d``/``velocity3d``.

Coefficient layout: ``coeffs`` is an ``(npt, 3, 5)`` array as produced by
``solve3d_orbit`` — ``coeffs[ix]`` is the ``(3, 5)`` matrix consumed by
``pos_c``, ``vel_c``, ``zvel_c``, ``sep_c``, and ``zpos_c``.

Each physical quantity lives in its own module, holding that quantity's
scalar kernel (``_X_os``), vector kernel (``_X_ov``), parallel vector twin
(``_X_ovp``), public dispatcher (``X_o``), and Numba ``@overload``
registration. Shared infrastructure (``_is_1d_array``, ``ep_ix``,
``solve3d_orbit``) lives in ``_common``. This ``__init__`` re-exports the
full surface as the package's public API.

The ``_X_ovp`` twins are compiled with ``parallel=True`` and a ``prange``
sample loop but are otherwise identical to the serial vector kernels. The
public dispatchers always route to the serial kernels; the twins are
opt-in via ``Orbit(parallel=True)``, which uses them only above a minimum
array size (the parallel-region launch costs tens of microseconds, so for
the value kernels the break-even is around 5e4 samples).
"""

from ._common import _is_1d_array, solve3d_orbit, ep_ix
from .position import pos_o, _pos_os, pos_ov, pos_ovp
from .zposition import zpos_o, _zpos_os, zpos_ov, zpos_ovp
from .separation import sep_o, _sep_os, sep_ov, sep_ovp
from .velocity import vel_o, _vel_os, vel_ov, vel_ovp
from .zvelocity import zvel_o, _zvel_os, zvel_ov, zvel_ovp
from .true_anomaly import true_anomaly_o, _true_anomaly_os, true_anomaly_ov, true_anomaly_ovp
from .projected_angle import cos_v_p_angle_o, _cos_v_p_angle_os, cos_v_p_angle_ov, cos_v_p_angle_ovp
from .phase_angle import cos_alpha_o, _cos_alpha_os, cos_alpha_ov, cos_alpha_ovp
from .star_planet_distance import (
    star_planet_distance_o, _star_planet_distance_os, star_planet_distance_ov,
    star_planet_distance_ovp,
)
from .lambert import (
    _lambert_kernel,
    lambert_phase_curve_o, _lambert_phase_curve_os, lambert_phase_curve_ov,
    lambert_phase_curve_ovp,
)
from .ev_signal import ev_signal_o, _ev_signal_os, ev_signal_ov, ev_signal_ovp
from .radial_velocity import rv_o, _rv_os, rv_ov, rv_ovp
from .light_travel_time import (
    LTT_DAYS_PER_RSUN,
    light_travel_time_o, _light_travel_time_os, light_travel_time_ov,
    light_travel_time_ovp,
)
