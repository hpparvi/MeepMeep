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

"""User-facing high-level orbit API.

This module exposes :class:`Orbit`, a thin object wrapper around the
multi-knot Taylor-series dispatchers in
:mod:`meepmeep.backends.numba.taylor.orbit3d` and
:mod:`meepmeep.backends.numba.taylor.orbit3dd`. It handles the bookkeeping
that the low-level dispatchers expect — building the knot grid, evaluating
Taylor coefficients (and their parameter derivatives, on request), and
threading the ``points`` / ``pktable`` / ``coeffs`` / ``dcoeffs`` arrays
through every observable.

Parameter conventions at this layer match CLAUDE.md: ``t0`` is the time of
inferior conjunction (transit center), ``p`` is the orbital period in days,
``a`` is the scaled semi-major axis :math:`a/R_\\star`, ``i`` is the
inclination in radians, ``e`` is the eccentricity, and ``w`` is the
argument of periastron in radians. The orbit3d/orbit3dd backend uses a
periastron-anchored time argument internally; the conversion happens once
inside :meth:`Orbit.set_pars`, which stores the periastron-anchor time as
``self._tpa``.
"""

from typing import Optional

from matplotlib.patches import Circle, Wedge
from matplotlib.pyplot import subplots, setp
from numpy import arccos, ndarray, mod, argmin, degrees, linspace, clip, sqrt

from .backends.numba.knots import create_knots
from .backends.numba.newton.newton import xyz_newton_v, ta_newton_v
from .backends.numba.utils import mean_anomaly_at_transit, TWO_PI, eccentricity_vector
from .backends.numba.taylor.orbit3d import (solve3d_orbit as solve_xyz_o5s, pos_ov, cos_alpha_ov, vel_ov,
                                            true_anomaly_ov, rv_ov, star_planet_distance_ov, ev_signal_ov,
                                            lambert_phase_curve_ov, lambert_and_emission_ov, light_travel_time_ov, )
from .backends.numba.taylor.orbit3dd import (solve3d_orbit_d, pos_ovd, cos_alpha_ovd, vel_ovd,
                                             true_anomaly_ovd, rv_ovd, star_planet_distance_ovd, ev_signal_ovd,
                                             lambert_phase_curve_ovd, lambert_and_emission_ovd,
                                             light_travel_time_ovd, )


class Orbit:
    """Multi-knot Taylor-series orbit evaluator.

    `Orbit` builds a knot grid once at construction time, computes Taylor
    coefficients (and, on request, parameter-derivative coefficients) when
    you bind orbital parameters via :meth:`set_pars`, and then evaluates any
    of the registered observables (position, velocity, projected separation,
    phase angle, radial velocity, phase curves, ellipsoidal variation, light
    travel time, …) at arbitrary times. In ``derivatives=True`` mode every
    observable additionally returns analytic gradients w.r.t. the six
    orbital parameters and any method-specific physical extras.

    Workflow:
    ``Orbit(npt, knot_placement, derivatives) → set_pars(t0, p, a, i, e, w)
    → set_data(times) → call any observable``.

    Parameters
    ----------
    npt : int
        Number of knots used by the multi-knot Taylor expansion (default
        15). Includes the periodic-image slot; raise this when the orbit is
        eccentric enough that 15 knots no longer cover periastron with
        adequate per-knot accuracy.
    knot_placement : str
        Knot placement strategy: ``'mm'`` (uniform in mean motion),
        ``'ea'`` (uniform in eccentric anomaly; default and preferred for
        eccentric orbits), or ``'ta'`` (uniform in true anomaly).
    derivatives : bool
        If ``True``, every evaluator method also returns parameter
        derivatives in addition to its value(s). The derivative ordering is
        ``(phase, p, a, i, e, w)`` followed by per-method physical extras
        (e.g. ``k`` for :meth:`radial_velocity`; ``ag, k`` for
        :meth:`lambert_phase_curve`; ``alpha, mass_ratio, inc`` for
        :meth:`ellipsoidal_variation`; see the underlying ``*_ovd`` routines
        in :mod:`meepmeep.backends.numba.taylor.orbit3dd` for the full
        signatures).

        With ``derivatives=True``, multi-coordinate returns are extended
        with derivative arrays (e.g. :meth:`xyz` returns
        ``(xs, ys, zs, dxs, dys, dzs)`` with each ``d*s`` of shape
        ``(N, 6)``); single-value returns become ``(value, dvalue)``.

        ``rstar`` derivatives for :meth:`light_travel_time` are *not*
        returned (per package spec); only the 6 orbital derivatives.

    Attributes
    ----------
    npt : int
        Number of knots, set at construction.
    times : ndarray or None
        Time grid bound via :meth:`set_data`. ``None`` until set.
    _tpa : float
        Periastron-anchor time, derived from the user-facing ``t0`` in
        :meth:`set_pars`. Passed straight through to every orbit3d /
        orbit3dd dispatcher as their ``tpa`` argument.
    _coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every knot, built in :meth:`set_pars`.
    _dcoeffs : ndarray, shape (npt, 6, 3, 5) or None
        Parameter-derivative coefficient tensors at every knot. ``None``
        unless the instance was constructed with ``derivatives=True``.
    _points : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]`` from
        :func:`~meepmeep.backends.numba.knots.create_knots`.
    _dt : float
        Width of one ``_tptable`` bucket in fraction of the period.
    _tptable : ndarray of int
        Time-to-knot lookup table.

    Notes
    -----
    Convention bridge: this class accepts the user-facing ``t0`` =
    inferior-conjunction (transit-center) time, but the underlying
    orbit3d / orbit3dd dispatchers anchor their knot grid at periastron.
    :meth:`set_pars` converts once via
    ``self._tpa = t0 - mean_anomaly_at_transit(e, w) / (2π) · p`` and
    every subsequent dispatcher call uses ``self._tpa``. The raw
    ``self._t0`` is kept around only for the Newton-Raphson diagnostic
    paths (:meth:`_xyz_error`, :meth:`_cos_phase_error`,
    :meth:`mean_anomaly`, :meth:`true_anomaly(exact=True)`) and for
    :meth:`plot(show_exact=True)`.
    """

    def __init__(self, npt: int = 15, knot_placement: str = "ea", derivatives: bool = False):
        """Construct a new ``Orbit``. See the class docstring for argument semantics."""
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
        """Bind a time grid to the instance.

        The bound grid is used as the default ``times`` argument by every
        evaluator that takes a ``times=None`` parameter, and unconditionally
        by methods that don't expose ``times`` at all
        (:meth:`vxyz`, :meth:`cos_phase`, :meth:`phase`, :meth:`theta`,
        :meth:`light_travel_time`, :meth:`radial_velocity`).

        Parameters
        ----------
        times : ndarray, shape (N,)
            Times at which to evaluate the orbit [days].
        """
        self.times = times

    def set_pars(self, t0, p, a, i, e, w):
        """Bind orbital parameters and (re-)compute the per-knot Taylor coefficients.

        Stores the user-facing parameters on the instance, derives the
        periastron-anchor time ``self._tpa`` from ``t0`` via
        ``t_pa = t0 - mean_anomaly_at_transit(e, w) / (2π) · p``, and calls
        :func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit` (or
        :func:`~meepmeep.backends.numba.taylor.orbit3dd.solve3d_orbit_d` in
        derivative mode) to fill ``self._coeffs`` (and ``self._dcoeffs``).

        Parameters
        ----------
        t0 : float
            Time of inferior conjunction (transit center) [days].
        p : float
            Orbital period [days].
        a : float
            Scaled semi-major axis :math:`a/R_\\star`.
        i : float
            Inclination [radians]. :math:`i = \\pi/2` is edge-on.
        e : float
            Eccentricity, :math:`0 \\le e < 1`.
        w : float
            Argument of periastron [radians].
        """
        self._t0 = t0
        self._p = p
        self._a = a
        self._i = i
        self._e = e
        self._w = w
        self._tpa = t0 - mean_anomaly_at_transit(e, w) / TWO_PI * p
        if self._derivatives:
            self._coeffs, self._dcoeffs = solve3d_orbit_d(self._points, p, a, i, e, w, self.npt)
        else:
            self._coeffs = solve_xyz_o5s(self._points, p, a, i, e, w, self.npt)

    def mean_anomaly(self):
        """Mean anomaly at every bound time, wrapped into :math:`[0, 2\\pi)`.

        Computed analytically from ``self._t0`` (transit-center time) via the
        mean-anomaly-at-transit offset, so this method does not use the
        Taylor backend at all.

        Returns
        -------
        m : ndarray, shape (N,)
            Mean anomaly per time in radians, in :math:`[0, 2\\pi)`.
        """
        offset = mean_anomaly_at_transit(self._e, self._w)
        return mod(TWO_PI * (self.times - (self._t0 - offset * self._p / TWO_PI)) / self._p, TWO_PI)

    def true_anomaly(self, exact: bool = False):
        """True anomaly at every bound time.

        Parameters
        ----------
        exact : bool, default False
            If ``True``, use the Newton-Raphson reference solver
            (:func:`~meepmeep.backends.numba.newton.newton.ta_newton_v`)
            instead of the Taylor backend. The exact path is meant for
            validation; it does not support parameter derivatives.

        Returns
        -------
        f : ndarray, shape (N,)
            True anomaly per time in radians, in :math:`[0, 2\\pi)`.
        df : ndarray, shape (N, 6)
            Gradient w.r.t. ``(phase, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True`` (and ``exact`` is ``False``).

        Raises
        ------
        NotImplementedError
            If ``exact=True`` is combined with ``derivatives=True``; the
            Newton-Raphson reference does not produce parameter gradients.
        """
        if exact and self._derivatives:
            raise NotImplementedError("exact=True is incompatible with derivatives — Newton-Raphson "
                                      "reference does not provide parameter derivatives.")
        if exact:
            return ta_newton_v(self.times, self._t0, self._p, self._e, self._w)
        ev = eccentricity_vector(self._i, self._e, self._w)
        if self._derivatives:
            return true_anomaly_ovd(self.times, self._tpa, self._p, ev[0], ev[1], ev[2], self._w, self._dt,
                                    self._tptable, self._points, self._coeffs, self._dcoeffs, )
        return true_anomaly_ov(self.times, self._tpa, self._p, ev[0], ev[1], ev[2], self._w, self._dt, self._tptable,
                               self._points, self._coeffs, )

    def xyz(self, times: Optional[ndarray] = None):
        """Planet (x, y, z) position in the sky frame.

        Parameters
        ----------
        times : ndarray or None
            Times at which to evaluate the position. If ``None``, uses the
            grid bound via :meth:`set_data`.

        Returns
        -------
        xs, ys, zs : ndarray, shape (N,)
            Position components in units of the stellar radius. ``xs``,
            ``ys`` are the sky-plane coordinates; ``zs`` is the
            line-of-sight depth (positive toward the observer).
        dxs, dys, dzs : ndarray, shape (N, 6)
            Gradients w.r.t. ``(phase, p, a, i, e, w)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            return pos_ovd(times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs,
                           self._dcoeffs, )
        return pos_ov(times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def _xyz_error(self):
        """Per-time position residuals against the Newton-Raphson reference.

        Diagnostic helper. Returns ``(dx, dy, dz)`` arrays of the difference
        between the Taylor backend's ``xyz`` and the exact Newton-Raphson
        ``xyz_newton_v``. Uses the value-only path even when the instance
        was constructed with ``derivatives=True``.
        """
        if self._derivatives:
            x, y, z, _, _, _ = self.xyz()
        else:
            x, y, z = self.xyz()
        xt, yt, zt = xyz_newton_v(self.times, self._t0, self._p, self._a, self._i, self._e, self._w)
        return x - xt, y - yt, z - zt

    def vxyz(self):
        """Planet (vx, vy, vz) velocity in the sky frame.

        Returns
        -------
        vxs, vys, vzs : ndarray, shape (N,)
            Velocity components in :math:`R_\\star/\\mathrm{day}`.
        dvxs, dvys, dvzs : ndarray, shape (N, 6)
            Gradients w.r.t. ``(phase, p, a, i, e, w)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        if self._derivatives:
            return vel_ovd(self.times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs,
                           self._dcoeffs, )
        return vel_ov(self.times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def cos_phase(self):
        """Cosine of the phase angle, :math:`\\cos\\alpha = -z/r`.

        Equals ``+1`` at superior conjunction (full phase, planet behind
        star) and ``-1`` at inferior conjunction (new phase, planet in
        front).

        Returns
        -------
        ca : ndarray, shape (N,)
            Cosine of the phase angle per time, in :math:`[-1, 1]`.
        dca : ndarray, shape (N, 6)
            Gradient w.r.t. ``(phase, p, a, i, e, w)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        if self._derivatives:
            return cos_alpha_ovd(self.times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs,
                                 self._dcoeffs, )
        return cos_alpha_ov(self.times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def _cos_phase_error(self):
        """Per-time phase-angle-cosine residuals against the Newton-Raphson reference.

        Diagnostic helper. Currently compares the Taylor backend's
        ``cos_phase`` against the *true anomaly* from
        :func:`~meepmeep.backends.numba.newton.newton.ta_newton_v` — i.e.
        a coarse sanity check, not a tight residual.
        """
        ta = ta_newton_v(self.times, self._t0, self._p, self._e, self._w)
        cos_alpha_t = ta
        if self._derivatives:
            ca, _ = self.cos_phase()
        else:
            ca = self.cos_phase()
        return ca - cos_alpha_t

    def phase(self):
        """Phase angle :math:`\\alpha = \\arccos(\\cos\\alpha)`.

        Returns
        -------
        ph : ndarray, shape (N,)
            Phase angle per time in radians, in :math:`[0, \\pi]`. Zero at
            superior conjunction, :math:`\\pi` at inferior conjunction.
        dph : ndarray, shape (N, 6)
            Gradient w.r.t. ``(phase, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``.

        Notes
        -----
        When ``derivatives=True``, the gradient is computed via the
        :math:`\\arccos` chain rule
        :math:`d\\alpha/d\\theta = -d\\cos\\alpha/d\\theta / \\sqrt{1 - \\cos^2\\alpha}`.
        At the transit/eclipse extrema :math:`|\\cos\\alpha| \\to 1` the
        derivative diverges; the implementation clamps :math:`\\cos\\alpha`
        slightly inside ``(-1, 1)`` so the returned gradient stays finite
        but loses physical meaning at the clamped points.
        """
        if self._derivatives:
            ca, dca = cos_alpha_ovd(self.times, self._tpa, self._p, self._dt, self._tptable, self._points,
                                    self._coeffs, self._dcoeffs, )
            ca_c = clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
            ph = arccos(ca_c)
            inv_s = -1.0 / sqrt(1.0 - ca_c * ca_c)
            dph = inv_s[:, None] * dca
            return ph, dph
        return arccos(cos_alpha_ov(self.times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs))

    def theta(self):
        """Supplement angle :math:`\\theta = \\arccos(-\\cos\\alpha) = \\pi - \\alpha`.

        Useful when the natural reference direction is the line from the
        star to the observer rather than the line from the planet to the
        observer.

        Returns
        -------
        th : ndarray, shape (N,)
            Supplement angle per time in radians, in :math:`[0, \\pi]`.
        dth : ndarray, shape (N, 6)
            Gradient w.r.t. ``(phase, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``.

        Notes
        -----
        See :meth:`phase` for the derivative-clamping caveat near
        :math:`|\\cos\\alpha| = 1`.
        """
        if self._derivatives:
            ca, dca = cos_alpha_ovd(self.times, self._tpa, self._p, self._dt, self._tptable, self._points,
                                    self._coeffs, self._dcoeffs, )
            ca_c = clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
            th = arccos(-ca_c)
            # d(arccos(-c))/dθ = +dc/dθ / sqrt(1 - c²)
            inv_s = 1.0 / sqrt(1.0 - ca_c * ca_c)
            dth = inv_s[:, None] * dca
            return th, dth
        return arccos(
            -cos_alpha_ov(self.times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs))

    def star_planet_distance(self, times: Optional[ndarray] = None):
        """3D star-planet separation :math:`r = \\sqrt{x^2 + y^2 + z^2}`.

        Distinct from :meth:`projected separation` (currently only available
        through the low-level ``sep_ov`` dispatcher), which drops the
        line-of-sight component.

        Parameters
        ----------
        times : ndarray or None
            Times at which to evaluate the separation. If ``None``, uses
            the grid bound via :meth:`set_data`.

        Returns
        -------
        r : ndarray, shape (N,)
            3D star-planet separation per time, in stellar radii.
        dr : ndarray, shape (N, 6)
            Gradient w.r.t. ``(phase, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            return star_planet_distance_ovd(times, self._tpa, self._p, self._dt, self._tptable, self._points,
                                            self._coeffs, self._dcoeffs, )
        return star_planet_distance_ov(times, self._tpa, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def light_travel_time(self, rstar: float):
        """Light-travel-time correction, referenced to primary transit.

        The correction is
        :math:`\\mathrm{ltt}(t) = -(z(t) - z(t_\\mathrm{transit})) \\cdot r_\\star \\cdot (R_\\odot/c)`
        with :math:`z` in stellar radii and the result in days. The
        reference is the primary transit, so ``ltt(t_transit) = 0``.

        Parameters
        ----------
        rstar : float
            Stellar radius [R_sun].

        Returns
        -------
        ltt : ndarray, shape (N,)
            Light travel time correction per time [days].
        dltt : ndarray, shape (N, 6)
            Gradient w.r.t. ``(phase, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``. The derivative w.r.t.
            ``rstar`` is intentionally *not* returned (per package spec).
        """
        if self._derivatives:
            return light_travel_time_ovd(self.times, self._tpa, self._p, self._e, self._w, rstar, self._dt,
                                         self._tptable, self._points, self._coeffs, self._dcoeffs, )
        return light_travel_time_ov(self.times, self._tpa, self._p, self._e, self._w, rstar, self._dt, self._tptable,
                                    self._points, self._coeffs, )

    def radial_velocity(self, k: float):
        """Stellar radial velocity (Perryman 2018, Eq. 2.23).

        Scales the line-of-sight planet velocity by the closed-form factor
        :math:`K / [(2\\pi/p)\\,(a\\sin i)/\\sqrt{1-e^2}]` so the result is
        the observed stellar RV in m/s.

        Parameters
        ----------
        k : float
            Radial-velocity semi-amplitude [m s\\ :sup:`-1`].

        Returns
        -------
        rvs : ndarray, shape (N,)
            Radial velocity per time [m s\\ :sup:`-1`].
        drvs : ndarray, shape (N, 7)
            Gradient w.r.t. ``(phase, p, a, i, e, w, k)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        if self._derivatives:
            return rv_ovd(self.times, k, self._tpa, self._p, self._a, self._i, self._e, self._dt, self._tptable,
                          self._points, self._coeffs, self._dcoeffs, )
        return rv_ov(self.times, k, self._tpa, self._p, self._a, self._i, self._e, self._dt, self._tptable,
                     self._points, self._coeffs, )

    def lambert_phase_curve(self, k: float, ag: float, times: ndarray | None = None):
        """Reflected-light phase curve assuming Lambertian scattering.

        Evaluates :math:`F(t) = (k/a)^2\\,A_g\\,f(\\alpha(t))` with the
        Lambert kernel
        :math:`f(\\alpha) = (\\sin\\alpha + (\\pi - \\alpha)\\cos\\alpha)/\\pi`.

        Parameters
        ----------
        k : float
            Planet-to-star radius ratio :math:`R_p/R_\\star`.
        ag : float
            Geometric albedo.
        times : ndarray or None
            Times at which to evaluate the flux. If ``None``, uses the
            grid bound via :meth:`set_data`.

        Returns
        -------
        flux : ndarray, shape (N,)
            Reflected planet-to-star flux ratio per time.
        dflux : ndarray, shape (N, 8)
            Gradient w.r.t. ``(phase, p, a, i, e, w, ag, k)``. Only
            returned when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            return lambert_phase_curve_ovd(times, ag, self._a, k, self._tpa, self._p, self._dt, self._tptable,
                                           self._points, self._coeffs, self._dcoeffs, )
        return lambert_phase_curve_ov(times, ag, self._a, k, self._tpa, self._p, self._dt, self._tptable, self._points,
                                      self._coeffs, )

    def lambert_and_emission(self, k: float, ag: float, fr_night, fr_day, times: ndarray | None = None):
        """Lambertian reflection plus a simple cosine-emission day/night model.

        Returns the reflected and thermal-emission flux ratios separately
        so callers can combine them with their own bolometric weighting.
        The emission model is a smooth interpolation between night-side and
        day-side levels driven by :math:`\\cos\\alpha`. The emission peak
        offset (advection) parameter is fixed at zero here; use the
        low-level :func:`~meepmeep.backends.numba.taylor.orbit3d.lambert_and_emission_ov`
        directly if you need it.

        Parameters
        ----------
        k : float
            Planet-to-star radius ratio :math:`R_p/R_\\star`.
        ag : float
            Geometric albedo (reflected component).
        fr_night : float
            Night-side flux ratio (planet/star).
        fr_day : float
            Day-side flux ratio (planet/star).
        times : ndarray or None
            Times at which to evaluate the flux. If ``None``, uses the
            grid bound via :meth:`set_data`.

        Returns
        -------
        ref : ndarray, shape (N,)
            Reflected (Lambertian) flux ratio per time.
        emi : ndarray, shape (N,)
            Thermal-emission flux ratio per time.
        dref : ndarray, shape (N, 11)
            Gradient of ``ref`` w.r.t.
            ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``.
            Only returned when ``self._derivatives`` is ``True``.
        demi : ndarray, shape (N, 11)
            Gradient of ``emi`` w.r.t. the same parameter block. Only
            returned when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            return lambert_and_emission_ovd(times, ag, fr_night, fr_day, 0.0, self._a, k, self._tpa, self._p, self._dt,
                                            self._tptable, self._points, self._coeffs, self._dcoeffs, )
        return lambert_and_emission_ov(times, ag, fr_night, fr_day, 0.0, self._a, k, self._tpa, self._p, self._dt,
                                       self._tptable, self._points, self._coeffs, )

    def ellipsoidal_variation(self, alpha: float, mass_ratio: float, times: Optional[ndarray] = None):
        """Ellipsoidal variation signal (Lillo-Box et al. 2014, Eqs. 6-10).

        Relative flux variation induced by the tidally distorted primary
        as a function of orbital phase. The amplitude scales with the mass
        ratio, the projected-area factor :math:`\\sin^2 i`, and the
        inverse cube of the instantaneous 3D star-planet separation.

        Parameters
        ----------
        alpha : float
            Gravity-darkening coefficient.
        mass_ratio : float
            Planet-to-star mass ratio :math:`M_p/M_\\star`.
        times : ndarray or None
            Times at which to evaluate the signal. If ``None``, uses the
            grid bound via :meth:`set_data`.

        Returns
        -------
        ev : ndarray, shape (N,)
            Relative flux variation due to ellipsoidal distortion.
        dev : ndarray, shape (N, 9)
            Gradient w.r.t.
            ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)``. Only
            returned when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            return ev_signal_ovd(alpha, mass_ratio, self._i, times, self._tpa, self._p, self._dt, self._tptable,
                                 self._points, self._coeffs, self._dcoeffs, )
        return ev_signal_ov(alpha, mass_ratio, self._i, times, self._tpa, self._p, self._dt, self._tptable,
                            self._points, self._coeffs, )

    def plot(self, figsize: Optional[tuple] = None, show_exact: bool = False, sr: float = 1.0, pr: float = 0.5, pc="k",
            npt: int = 1000, ):
        """Diagnostic three-panel plot of the orbit geometry.

        Renders front (X-Y), top (X-Z), and side (Z-Y) projections with
        the planet drawn at the first knot, the orbit traced over one full
        period, and arrows along the X-Z trace indicating the direction of
        motion. Temporarily rebinds ``self.times`` to a dense grid of
        ``npt`` samples for the plot and restores it on exit.

        Parameters
        ----------
        figsize : tuple or None
            Matplotlib figure size; passed to ``subplots``.
        show_exact : bool, default False
            If ``True``, overlay the Newton-Raphson reference trajectory
            (``xyz_newton_v``) as a black dashed line in each panel.
        sr : float, default 1.0
            Stellar radius drawn at the origin [stellar radii]. Used only
            for the cosmetic stellar disc.
        pr : float, default 0.5
            Planet-marker radius [stellar radii]. Cosmetic only.
        pc : matplotlib color, default ``"k"``
            Planet-marker face colour.
        npt : int, default 1000
            Number of samples used to draw the orbit trace.

        Returns
        -------
        None
            The figure is rendered in-place; the function does not return
            the ``Figure`` / ``Axes`` handles.
        """
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
