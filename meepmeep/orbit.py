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

from typing import Optional

from matplotlib.patches import Circle, Wedge
from matplotlib.pyplot import subplots, setp
from numpy import arccos, ndarray, mod, argmin, degrees, linspace, clip, sqrt

from .backends.numba.knots import create_knots
from .backends.numba.newton.newton import xyz_newton_v, ta_newton_v
from .backends.numba.utils import mean_anomaly_at_transit, TWO_PI, eccentricity_vector
from .backends.numba.taylor.orbit3d import (solve3d_orbit as solve_xyz_o5s, xyz_o5v, cos_alpha_o5v, vxyz_o5v,
                                            true_anomaly_o5v, rv_o5v, star_planet_distance_o5v, ev_signal_o5v,
                                            lambert_phase_curve_o5v, lambert_and_emission_o5v, light_travel_time_o5v, )
from .backends.numba.taylor.orbit3dd import (solve3d_orbit_d, xyz_o5v_d, cos_alpha_o5v_d, vxyz_o5v_d,
                                             true_anomaly_o5v_d, rv_o5v_d, star_planet_distance_o5v_d, ev_signal_o5v_d,
                                             lambert_phase_curve_o5v_d, lambert_and_emission_o5v_d,
                                             light_travel_time_o5v_d, )


class Orbit:
    """Multi-knot Taylor-series orbit evaluator.

    Parameters
    ----------
    npt : int
        Number of knots used by the Taylor expansion.
    knot_placement : str
        Knot placement strategy ('mm', 'ea', 'ta').
    derivatives : bool
        If ``True``, every evaluator method also returns parameter
        derivatives in addition to its value(s). The derivative ordering is
        ``(phase, p, a, i, e, w)`` followed by per-method physical extras
        (e.g. ``k`` for ``radial_velocity``; ``ag, k`` for
        ``lambert_phase_curve``; ``alpha, mass_ratio, inc`` for
        ``ellipsoidal_variation``; see the underlying ``*_o5v_d`` routines in
        ``meepmeep.backends.numba.taylor.orbit3dd`` for full details).

        With ``derivatives=True``, multi-coordinate returns are extended
        with derivative arrays (e.g. ``xyz()`` returns
        ``(xs, ys, zs, dxs, dys, dzs)`` with each ``d*s`` of shape
        ``(N, 6)``); single-value returns become ``(value, dvalue)``.

        ``rstar`` derivatives for ``light_travel_time`` are *not* returned
        (per package spec); only the 6 orbital derivatives.
    """

    def __init__(self, npt: int = 15, knot_placement: str = "ea", derivatives: bool = False):
        self.npt: int = npt
        self.times: Optional[ndarray] = None

        self._dt: Optional[float] = None
        self._points: Optional[float] = None
        self._coeffs: Optional[ndarray] = None
        self._dcoeffs: Optional[ndarray] = None
        self._t0: Optional[float] = None
        self._p: Optional[float] = None
        self._a: Optional[float] = None
        self._i: Optional[float] = None
        self._e: Optional[float] = None
        self._w: Optional[float] = None
        self._derivatives: bool = derivatives

        self._points, self._change_times, self._dt, self._tptable = create_knots(npt, 0.2, knot_placement)

    def set_data(self, times):
        self.times = times

    def set_pars(self, t0, p, a, i, e, w):
        self._t0 = t0
        self._p = p
        self._a = a
        self._i = i
        self._e = e
        self._w = w
        self._tc = t0 - mean_anomaly_at_transit(e, w) / TWO_PI * p
        if self._derivatives:
            self._coeffs, self._dcoeffs = solve3d_orbit_d(self._points, p, a, i, e, w, self.npt)
        else:
            self._coeffs = solve_xyz_o5s(self._points, p, a, i, e, w, self.npt)

    def mean_anomaly(self):
        offset = mean_anomaly_at_transit(self._e, self._w)
        return mod(TWO_PI * (self.times - (self._t0 - offset * self._p / TWO_PI)) / self._p, TWO_PI)

    def true_anomaly(self, exact: bool = False):
        if exact and self._derivatives:
            raise NotImplementedError("exact=True is incompatible with derivatives — Newton-Raphson "
                                      "reference does not provide parameter derivatives.")
        if exact:
            return ta_newton_v(self.times, self._t0, self._p, self._e, self._w)
        ev = eccentricity_vector(self._i, self._e, self._w)
        if self._derivatives:
            return true_anomaly_o5v_d(self.times, self._tc, self._p, ev[0], ev[1], ev[2], self._w, self._dt,
                self._tptable, self._points, self._coeffs, self._dcoeffs, )
        return true_anomaly_o5v(self.times, self._tc, self._p, ev[0], ev[1], ev[2], self._w, self._dt, self._tptable,
            self._points, self._coeffs, )

    def xyz(self, times: Optional[ndarray] = None):
        times = times if times is not None else self.times
        if self._derivatives:
            return xyz_o5v_d(times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs,
                self._dcoeffs, )
        return xyz_o5v(times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def _xyz_error(self):
        # Diagnostic against Newton-Raphson; uses value-only path.
        if self._derivatives:
            x, y, z, _, _, _ = self.xyz()
        else:
            x, y, z = self.xyz()
        xt, yt, zt = xyz_newton_v(self.times, self._t0, self._p, self._a, self._i, self._e, self._w)
        return x - xt, y - yt, z - zt

    def vxyz(self):
        if self._derivatives:
            return vxyz_o5v_d(self.times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs,
                self._dcoeffs, )
        return vxyz_o5v(self.times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def cos_phase(self):
        if self._derivatives:
            return cos_alpha_o5v_d(self.times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs,
                self._dcoeffs, )
        return cos_alpha_o5v(self.times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def _cos_phase_error(self):
        ta = ta_newton_v(self.times, self._t0, self._p, self._e, self._w)
        cos_alpha_t = ta
        if self._derivatives:
            ca, _ = self.cos_phase()
        else:
            ca = self.cos_phase()
        return ca - cos_alpha_t

    def phase(self):
        """Phase angle ``arccos(cos_alpha)``.

        Notes
        -----
        When ``derivatives=True``, the derivative is computed via the arccos
        chain rule ``dphase/dθ = -dcα/dθ / sqrt(1 - cα²)``. At the
        transit/eclipse extrema ``|cα| → 1`` the derivative diverges; the
        implementation clamps ``cα`` slightly inside ``(-1, 1)`` so the
        returned gradient stays finite but loses physical meaning at the
        clamped points.
        """
        if self._derivatives:
            ca, dca = cos_alpha_o5v_d(self.times, self._tc, self._p, self._dt, self._tptable, self._points,
                self._coeffs, self._dcoeffs, )
            ca_c = clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
            ph = arccos(ca_c)
            inv_s = -1.0 / sqrt(1.0 - ca_c * ca_c)
            dph = inv_s[:, None] * dca
            return ph, dph
        return arccos(cos_alpha_o5v(self.times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs))

    def theta(self):
        """Angle ``arccos(-cos_alpha)``.

        Notes
        -----
        See :meth:`phase` for the derivative-clamping caveat near the
        ``|cα| = 1`` extrema.
        """
        if self._derivatives:
            ca, dca = cos_alpha_o5v_d(self.times, self._tc, self._p, self._dt, self._tptable, self._points,
                self._coeffs, self._dcoeffs, )
            ca_c = clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
            th = arccos(-ca_c)
            # d(arccos(-c))/dθ = +dc/dθ / sqrt(1 - c²)
            inv_s = 1.0 / sqrt(1.0 - ca_c * ca_c)
            dth = inv_s[:, None] * dca
            return th, dth
        return arccos(
            -cos_alpha_o5v(self.times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs))

    def star_planet_distance(self, times: Optional[ndarray] = None):
        times = times if times is not None else self.times
        if self._derivatives:
            return star_planet_distance_o5v_d(times, self._tc, self._p, self._dt, self._tptable, self._points,
                self._coeffs, self._dcoeffs, )
        return star_planet_distance_o5v(times, self._tc, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def light_travel_time(self, rstar: float):
        """Light-travel-time correction, referenced to primary transit.

        Returns ``ltt`` of shape ``(N,)`` in non-derivative mode and
        ``(ltt, dltt)`` with ``dltt`` of shape ``(N, 6)`` in derivative
        mode. The derivative w.r.t. ``rstar`` is intentionally not
        returned (per spec); only the 6 orbital parameter derivatives.
        """
        if self._derivatives:
            return light_travel_time_o5v_d(self.times, self._tc, self._p, self._e, self._w, rstar, self._dt,
                self._tptable, self._points, self._coeffs, self._dcoeffs, )
        return light_travel_time_o5v(self.times, self._tc, self._p, self._e, self._w, rstar, self._dt, self._tptable,
            self._points, self._coeffs, )

    def radial_velocity(self, k: float):
        if self._derivatives:
            return rv_o5v_d(self.times, k, self._tc, self._p, self._a, self._i, self._e, self._dt, self._tptable,
                self._points, self._coeffs, self._dcoeffs, )
        return rv_o5v(self.times, k, self._tc, self._p, self._a, self._i, self._e, self._dt, self._tptable,
            self._points, self._coeffs, )

    def lambert_phase_curve(self, k: float, ag: float, times: ndarray | None = None):
        times = times if times is not None else self.times
        if self._derivatives:
            return lambert_phase_curve_o5v_d(times, ag, self._a, k, self._tc, self._p, self._dt, self._tptable,
                self._points, self._coeffs, self._dcoeffs, )
        return lambert_phase_curve_o5v(times, ag, self._a, k, self._tc, self._p, self._dt, self._tptable, self._points,
            self._coeffs, )

    def lambert_and_emission(self, k: float, ag: float, fr_night, fr_day, times: ndarray | None = None):
        times = times if times is not None else self.times
        if self._derivatives:
            return lambert_and_emission_o5v_d(times, ag, fr_night, fr_day, 0.0, self._a, k, self._tc, self._p, self._dt,
                self._tptable, self._points, self._coeffs, self._dcoeffs, )
        return lambert_and_emission_o5v(times, ag, fr_night, fr_day, 0.0, self._a, k, self._tc, self._p, self._dt,
            self._tptable, self._points, self._coeffs, )

    def ellipsoidal_variation(self, alpha: float, mass_ratio: float, times: Optional[ndarray] = None):
        """Ellipsoidal variation signal.

        NOTES: See Eqs. 6-10 in Lillo-Box al. (2014).
        """
        times = times if times is not None else self.times
        if self._derivatives:
            return ev_signal_o5v_d(alpha, mass_ratio, self._i, times, self._tc, self._p, self._dt, self._tptable,
                self._points, self._coeffs, self._dcoeffs, )
        return ev_signal_o5v(alpha, mass_ratio, self._i, times, self._tc, self._p, self._dt, self._tptable,
            self._points, self._coeffs, )

    def plot(self, figsize: Optional[tuple] = None, show_exact: bool = False, sr: float = 1.0, pr: float = 0.5, pc="k",
            npt: int = 1000, ):
        tcur = self.times
        self.set_data(linspace(0, self._p, npt))

        # Use value-only path for plotting regardless of derivative mode.
        if self._derivatives:
            x, y, z, _, _, _ = self.xyz()
        else:
            x, y, z = self.xyz()
        xl, yl, zl = 1.1 * abs(x).max(), 1.1 * abs(y).max(), 1.1 * abs(z).max()
        al = max([xl, yl, zl])

        fig, axs = subplots(1, 3, figsize=figsize)
        axs[0].plot(x, y, zorder=0)
        axs[0].add_patch(Circle((self._coeffs[0, 0, 0], self._coeffs[0, 1, 0]), pr, fc=pc, ec="k", zorder=10))
        axs[1].plot(x, z, zorder=1)
        axs[1].add_patch(Circle((self._coeffs[0, 0, 0], self._coeffs[0, 2, 0]), pr, fc=pc, ec="k", zorder=11))
        axs[1].add_patch(Wedge((0, 0), 1.3 * sr, 180 - degrees(self._w), 180, fc=pc, ec="k", zorder=-10))

        axs[2].plot(z, y, zorder=2)
        axs[2].add_patch(Circle((self._coeffs[0, 2, 0], self._coeffs[0, 1, 0]), pr, fc=pc, ec="k", zorder=12))

        di = self.times.size // 6
        for i in range(6):
            axs[1].arrow(x[i * di], z[i * di], x[i * di + 1] - x[i * di], z[i * di + 1] - z[i * di], shape="full", lw=6,
                length_includes_head=True, head_width=0.1, color="k", )

        m = x < 0.0
        axs[1].plot((0, x[m][argmin(abs(z[m]))]), (0, 0), "k", zorder=-10, ls="--")
        omega_ix = argmin(x ** 2 + y ** 2 + z ** 2)
        axs[1].plot((0, x[omega_ix]), (0, z[omega_ix]), "k", zorder=-10, ls="--")

        if show_exact:
            xt, yt, zt = xyz_newton_v(self.times, self._t0, self._p, self._a, self._i, self._e, self._w)
            axs[0].plot(xt, yt, "k--")
            axs[1].plot(xt, zt, "k--")
            axs[2].plot(zt, yt, "k--")

        [ax.add_patch(Circle((0, 0), sr, fc="y", ec="k")) for ax in axs]
        [ax.set_aspect(1) for ax in axs]
        setp(axs, xlim=(-al, al), ylim=(-al, al))
        setp(axs[0], xlabel="X", ylabel="Y", title="Front")
        setp(axs[1], xlabel="X", ylabel="Z", ylim=(al, -al), title="Top")
        setp(axs[2], xlabel="Z", ylabel="Y", title="Side")
        fig.tight_layout()
        self.set_data(tcur)
