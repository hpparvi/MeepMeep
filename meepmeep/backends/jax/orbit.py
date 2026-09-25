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

"""High-level JAX orbit: an immutable pytree holding the parameters, grid and coefficients.

:class:`JaxOrbit` is the functional counterpart of :class:`meepmeep.Orbit`.
Instead of binding parameters and data with ``set_pars``/``set_data``, an
orbit is built by a pure constructor and every evaluator takes the times
explicitly, so the whole model traces, jits and differentiates::

    def loglike(theta):
        tc, p, a, i, e, w = theta
        orbit = JaxOrbit.from_tc(tc, p, a, i, e, w)
        return -0.5 * jnp.sum((orbit.projected_separation(times) - z_obs) ** 2)

    jax.jit(jax.grad(loglike))(theta)

Gradients come out in the basis of the constructor: ``from_tc`` gives the
transit-centre basis, ``from_tp`` the periastron basis.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from ._common import TWO_PI
from .expansion_points import create_expansion_points
from .solve import solve3d_orbit
from .utils import mean_anomaly_at_transit, eccentricity_vector
from . import orbit3d as o3

EP_GRID_E_FLOOR = 0.2


@dataclass(frozen=True)
class JaxOrbit:
    """Taylor-series Keplerian orbit as a JAX pytree.

    Build instances with ``from_tc`` or ``from_tp``. All fields are
    pytree leaves, so an orbit can be passed through ``jit``, ``vmap``
    (e.g. over a batch of parameter sets) and ``grad``.

    Attributes
    ----------
    tc, tp : Array
        Transit-centre and periastron times [days].
    p, a, i, e, w, lan : Array
        Period [days], scaled semi-major axis [R_star], inclination,
        eccentricity, argument of periastron and longitude of the ascending
        node [radians].
    ep_times, dt, ep_table : Array
        Expansion-point grid from
        :func:`~meepmeep.jax3d.create_expansion_points`.
    coeffs : Array, shape (npt, 3, 5)
        Taylor coefficients at every expansion point.
    """
    tc: Array
    tp: Array
    p: Array
    a: Array
    i: Array
    e: Array
    w: Array
    lan: Array
    ep_times: Array
    dt: Array
    ep_table: Array
    coeffs: Array

    @classmethod
    def _build(cls, tc, tp, p, a, i, e, w, lan, npt, ep_placement, tres, grid):
        if grid is None:
            # Like meepmeep.Orbit, place the grid for max(e, 0.2): near-circular orbits
            # keep a mild periastron clustering. The placement is held fixed under
            # differentiation, so the expansion points sit at fixed phases as in numba.
            e_grid = jax.lax.stop_gradient(jnp.maximum(e, EP_GRID_E_FLOOR))
            grid = create_expansion_points(npt, e_grid, ep_placement, tres)
        ep_times, _, dt, ep_table = grid
        coeffs = solve3d_orbit(ep_times, p, a, i, e, w, lan)
        return cls(*[jnp.asarray(v, dtype=float) for v in (tc, tp, p, a, i, e, w, lan, ep_times, dt)],
                   jnp.asarray(ep_table), coeffs)

    @classmethod
    def from_tc(cls, tc, p, a, i, e, w, lan=0.0, *, npt: int = 15, ep_placement: str = 'ea',
                tres: int = 200, grid=None):
        """Build an orbit anchored at the transit centre.

        Parameters
        ----------
        tc : float
            Time of inferior conjunction (transit centre) [days].
        p : float
            Orbital period [days].
        a : float
            Scaled semi-major axis [R_star].
        i : float
            Inclination [radians].
        e : float
            Eccentricity.
        w : float
            Argument of periastron [radians].
        lan : float, optional
            Longitude of the ascending node [radians]. Defaults to 0.0.
        npt : int, optional
            Number of expansion points (odd, including the periodic image). Static.
        ep_placement : {'ea', 'ta', 'mm'}, optional
            Expansion-point placement strategy. Static.
        tres : int, optional
            Time-to-expansion-point table resolution. Static.
        grid : tuple, optional
            A precomputed ``(ep_times, change_times, dt, ep_table)`` grid, for
            example from the numba ``create_expansion_points``. By default the
            grid is built for ``max(e, 0.2)`` and held fixed under
            differentiation.
        """
        tp = tc - mean_anomaly_at_transit(e, w) / TWO_PI * p
        return cls._build(tc, tp, p, a, i, e, w, lan, npt, ep_placement, tres, grid)

    @classmethod
    def from_tp(cls, tp, p, a, i, e, w, lan=0.0, *, npt: int = 15, ep_placement: str = 'ea',
                tres: int = 200, grid=None):
        """Build an orbit anchored at the periastron passage ``tp`` [days].

        See ``from_tc`` for the other arguments.
        """
        tc = tp + mean_anomaly_at_transit(e, w) / TWO_PI * p
        return cls._build(tc, tp, p, a, i, e, w, lan, npt, ep_placement, tres, grid)

    def _grid(self):
        return self.tp, self.p, self.dt, self.ep_table, self.ep_times, self.coeffs

    def mean_anomaly(self, times):
        """Mean anomaly [radians] in ``[0, 2 pi)``, computed analytically."""
        offset = mean_anomaly_at_transit(self.e, self.w)
        return jnp.mod(TWO_PI * (times - (self.tc - offset * self.p / TWO_PI)) / self.p, TWO_PI)

    def true_anomaly(self, times):
        """True anomaly [radians] in ``[0, 2 pi)``.

        The eccentricity vector is differentiated too, so the gradient is the
        full derivative of the true anomaly, as in
        :meth:`meepmeep.orbit.Orbit.true_anomaly`. The eccentricity vector is
        rotated by ``lan`` together with the positions.
        """
        ex, ey, ez = eccentricity_vector(self.i, self.e, self.w, self.lan)
        tp, p, dt, ep_table, ep_times, coeffs = self._grid()
        return o3.true_anomaly_o(times, tp, p, ex, ey, ez, self.w, dt, ep_table, ep_times, coeffs)

    def xyz(self, times):
        """Planet position ``(x, y, z)`` [R_star]."""
        return o3.pos_o(times, *self._grid())

    def vxyz(self, times):
        """Planet velocity ``(vx, vy, vz)`` [R_star / day]."""
        return o3.vel_o(times, *self._grid())

    def projected_separation(self, times):
        """Sky-projected separation between the star and planet centers [R_star]."""
        return o3.sep_o(times, *self._grid())

    def cos_phase(self, times):
        """Cosine of the star-planet-observer phase angle."""
        return o3.cos_alpha_o(times, *self._grid())

    def phase(self, times):
        """Phase angle [radians]; zero at full phase (secondary eclipse)."""
        return jnp.arccos(jnp.clip(self.cos_phase(times), -1.0 + 1e-15, 1.0 - 1e-15))

    def theta(self, times):
        """Supplement of the phase angle, ``pi - phase`` [radians]."""
        return jnp.arccos(-jnp.clip(self.cos_phase(times), -1.0 + 1e-15, 1.0 - 1e-15))

    def star_planet_distance(self, times):
        """3D star-planet distance [R_star]."""
        return o3.star_planet_distance_o(times, *self._grid())

    def light_travel_time(self, times, rstar):
        """Light travel time relative to the transit centre [days]; ``rstar`` in R_sun."""
        tp, p, dt, ep_table, ep_times, coeffs = self._grid()
        return o3.light_travel_time_o(times, tp, p, self.e, self.w, rstar, dt, ep_table, ep_times, coeffs)

    def radial_velocity(self, times, k):
        """Radial velocity with semi-amplitude ``k``."""
        tp, p, dt, ep_table, ep_times, coeffs = self._grid()
        return o3.rv_o(times, k, tp, p, self.a, self.i, self.e, dt, ep_table, ep_times, coeffs)

    def lambert_phase_curve(self, times, k, ag):
        """Lambert-sphere reflected-light phase curve for radius ratio ``k`` and geometric albedo ``ag``."""
        return o3.lambert_phase_curve_o(times, ag, k, *self._grid())

    def emission_phase_curve(self, times, k, fratio, offset):
        """Thermal-emission phase curve with surface brightness ratio ``fratio`` and hotspot ``offset``."""
        return o3.emission_phase_curve_o(times, k, fratio, offset, *self._grid())

    def ellipsoidal_variation(self, times, alpha, mass_ratio):
        """Ellipsoidal-variation signal with amplitude coefficient ``alpha`` and mass ratio ``mass_ratio``."""
        return o3.ev_signal_o(alpha, mass_ratio, self.i, times, *self._grid())


jax.tree_util.register_dataclass(
    JaxOrbit,
    data_fields=['tc', 'tp', 'p', 'a', 'i', 'e', 'w', 'lan', 'ep_times', 'dt', 'ep_table', 'coeffs'],
    meta_fields=[],
)
