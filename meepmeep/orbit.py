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

"""High-level orbit API.

MeepMeep computes Keplerian exoplanet orbits very fast. This module is
the user-facing front door: the :class:`Orbit` class lets you set a few
orbital elements once and then ask for whatever quantity you need at
whatever times you need: planet position, velocity, sky-projected
separation, phase angle, radial velocity, reflected-light and thermal
phase curves, ellipsoidal variation, and light travel time. In
derivative mode every quantity also comes back with analytic gradients
w.r.t. the orbital parameters, ready to feed a gradient-based optimiser
or sampler.

The time anchor is
either ``tc`` (transit center, the time of inferior conjunction) or ``tp``
(time of periastron passage); :meth:`Orbit.set_pars` accepts either and
stores both internally. The remaining elements are ``p`` (orbital period
in days), ``a`` (scaled semi-major axis :math:`a/R_\\star`), ``i``
(inclination in radians), ``e`` (eccentricity), ``w`` (argument of
periastron in radians), and the optional ``lan`` (longitude of the
ascending node in radians, a sky-plane rotation about the line of sight;
defaults to 0). Internally the Taylor backend anchors its knot
grid at periastron, so :meth:`Orbit.set_pars` converts once and stores
the periastron-anchor time as ``self._tp`` and the transit-center time
as ``self._tc``.
"""

from typing import Optional

from matplotlib.patches import Circle, Wedge
from matplotlib.pyplot import subplots, setp
from numpy import arccos, ndarray, mod, argmin, degrees, linspace, clip, sqrt

from .backends.numba.knots import create_knots
from .backends.numba.newton.newton import xyz_newton_v, ta_newton_v
from .backends.numba.utils import mean_anomaly_at_transit, TWO_PI, eccentricity_vector, tc_to_tp_gradient
from .backends.numba.orbit3d import (solve3d_orbit, pos_o, cos_alpha_o, vel_o,
                                            true_anomaly_o, rv_o, star_planet_distance_o, ev_signal_o,
                                            lambert_phase_curve_o, light_travel_time_o, )
from .backends.numba.orbit3dd import (solve3d_orbit_d, pos_od, cos_alpha_od, vel_od, true_anomaly_od, rv_od,
                                             star_planet_distance_od, ev_signal_od, lambert_phase_curve_od,
                                             light_travel_time_od, )
from .backends.numba.orbit3d import (_pos_ovp, _vel_ovp, _cos_alpha_ovp, _true_anomaly_ovp, _rv_ovp,
                                            _star_planet_distance_ovp, _ev_signal_ovp, _lambert_phase_curve_ovp,
                                            _light_travel_time_ovp, )
from .backends.numba.orbit3dd import (_pos_ovdp, _vel_ovdp, _cos_alpha_ovdp, _true_anomaly_ovdp, _rv_ovdp,
                                             _star_planet_distance_ovdp, _ev_signal_ovdp, _lambert_phase_curve_ovdp,
                                             _light_travel_time_ovdp, )


class Orbit:
    """Evaluate Keplerian orbits at arbitrary times, with optional analytic gradients.

    Bind a handful of orbital elements once via :meth:`set_pars`, hand it
    a time grid via :meth:`set_data`, and then ask for any of the
    quantities that exoplanet light-curve and radial-velocity modelling
    typically need: planet position, velocity, sky-projected separation,
    phase angle, radial velocity, reflected-light and thermal phase
    curves, ellipsoidal variation, light travel time. Construct with
    ``derivatives=True`` to additionally receive the analytic gradient of
    each quantity w.r.t. the seven orbital parameters (and any
    method-specific extras such as ``k`` or ``ag``), which is what a
    gradient-based optimiser or HMC sampler wants.

    Workflow:
    ``Orbit(npt, knot_placement, derivatives) → set_pars(tc=..., p=..., a=...,
    i=..., e=..., w=...) → set_data(times) → call any observable``. Pass
    ``tp=...`` instead of ``tc=...`` to anchor the orbit at periastron
    passage instead of transit center.

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
        ``(tc, p, a, i, e, w, lan)`` followed by per-method physical extras
        (e.g. ``k`` for :meth:`radial_velocity`; ``ag, k`` for
        :meth:`lambert_phase_curve`; ``alpha, mass_ratio, inc`` for
        :meth:`ellipsoidal_variation`; see the underlying ``*_od`` routines
        in :mod:`meepmeep.backends.numba.orbit3dd` for the full
        signatures).

        The timing slot and the shape derivatives follow the timing
        parameter bound via :meth:`set_pars`. Bind ``tc`` (the default) and
        slot 0 is :math:`\\partial/\\partial t_c` with the ``e``, ``w``,
        ``p`` derivatives taken at constant ``tc``; bind ``tp`` and the
        gradient is returned in the periastron basis
        ``(tp, p, a, i, e, w, lan)`` (constant ``tp``). See the
        "Transit-centre vs periastron parametrisation" section of the
        derivatives documentation for the exact relationship.

        With ``derivatives=True``, multi-coordinate returns are extended
        with derivative arrays (e.g. :meth:`xyz` returns
        ``(xs, ys, zs, dxs, dys, dzs)`` with each ``d*s`` of shape
        ``(N, 7)``); single-value returns become ``(value, dvalue)``.

        ``rstar`` derivatives for :meth:`light_travel_time` are *not*
        returned (per package spec); only the seven orbital derivatives.
    parallel : bool
        If ``True``, evaluations over time arrays with at least
        ``_PARALLEL_NMIN_GRAD`` (gradient mode) or ``_PARALLEL_NMIN_VALUE``
        (value mode) samples route to multi-threaded ``prange`` kernel
        twins; smaller arrays always take the serial kernels, which are
        faster below those sizes. The results are identical either way.
        Defaults to ``False``: leave it off when the surrounding
        application already parallelises at a higher level (e.g. one
        process per MCMC chain), where nested thread pools oversubscribe
        the machine.

    Attributes
    ----------
    npt : int
        Number of knots, set at construction.
    times : ndarray or None
        Time grid bound via :meth:`set_data`. ``None`` until set.
    _tc : float
        Transit-center time, set by :meth:`set_pars` (directly if you
        pass ``tc``, or derived from ``tp`` otherwise). Used by the
        Newton-Raphson diagnostic paths and by
        :meth:`plot(show_exact=True)`.
    _tp : float
        Periastron-anchor time. Passed straight through to every orbit3d /
        orbit3dd dispatcher as their ``tpa`` argument. Set by
        :meth:`set_pars` (directly if you pass ``tp``, or derived from
        ``tc`` otherwise).
    _timing : str
        Which timing convention was last bound via :meth:`set_pars`,
        ``"tc"`` (transit centre) or ``"tp"`` (periastron passage). In
        derivative mode it selects the gradient basis: ``"tp"`` triggers a
        reparametrisation of ``_dcoeffs`` into the periastron basis.
    _coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every knot, built in :meth:`set_pars`.
    _dcoeffs : ndarray, shape (npt, 7, 3, 5) or None
        Parameter-derivative coefficient tensors at every knot. ``None``
        unless the instance was constructed with ``derivatives=True``.
    _points : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]`` from
        :func:`~meepmeep.backends.numba.knots.create_knots`. Built at
        construction for ``e = _KNOT_GRID_E_FLOOR`` and rebuilt by
        :meth:`set_pars` when the bound eccentricity drifts more than
        ``_KNOT_GRID_E_TOL`` from the grid's construction eccentricity
        (``'ea'``/``'ta'`` placements only; the ``'mm'`` grid is
        eccentricity-independent).
    _grid_e : float
        Eccentricity the current knot grid was built for,
        ``max(e, _KNOT_GRID_E_FLOOR)`` at the last rebuild.
    _dt : float
        Width of one ``_tptable`` bucket in fraction of the period.
    _tptable : ndarray of int
        Time-to-knot lookup table.

    Notes
    -----
    Convention bridge: this class accepts either ``tc`` (transit-center
    time) or ``tp`` (periastron-passage time) via :meth:`set_pars`. The
    underlying orbit3d / orbit3dd dispatchers anchor their knot grid at
    periastron, so :meth:`set_pars` converts once via
    :math:`t_p = t_c - M_\\mathrm{tr}(e, w) \\cdot p / (2\\pi)` and stores
    both values: ``self._tc`` (transit center) and ``self._tp``
    (periastron anchor). Every dispatcher call uses ``self._tp``;
    ``self._tc`` is used only by the Newton-Raphson diagnostic paths
    (:meth:`_xyz_error`, :meth:`_cos_phase_error`, :meth:`mean_anomaly`,
    :meth:`true_anomaly(exact=True)`) and by :meth:`plot(show_exact=True)`.
    """

    # The knot grid is built for max(e, _KNOT_GRID_E_FLOOR): near-circular
    # grids are essentially uniform, so anything below the floor shares one
    # grid. It is rebuilt in set_pars when the bound eccentricity drifts more
    # than _KNOT_GRID_E_TOL from the grid's construction eccentricity. The
    # tolerance keeps rebuilds rare: create_knots runs scipy root solves in
    # Python (~0.3 ms), and set_pars is the per-likelihood-call hot path in
    # fitting applications.
    _KNOT_GRID_E_FLOOR = 0.2
    _KNOT_GRID_E_TOL = 0.05

    # Minimum time-array sizes for which the prange kernel twins beat the
    # serial kernels (measured on a 16-core machine; the parallel-region
    # launch costs tens of microseconds). Value-only kernels do ~5x less
    # work per sample than gradient kernels, so their break-even is higher.
    _PARALLEL_NMIN_VALUE = 50_000
    _PARALLEL_NMIN_GRAD = 10_000

    def __init__(self, npt: int = 15, knot_placement: str = "ea", derivatives: bool = False,
                 parallel: bool = False):
        """Construct a new ``Orbit``. See the class docstring for argument semantics."""
        self.npt: int = npt
        self.times: Optional[ndarray] = None
        self._parallel: bool = parallel

        self._dt: Optional[float] = None
        self._points: Optional[float] = None
        self._coeffs: Optional[ndarray] = None
        self._dcoeffs: Optional[ndarray] = None
        self._tc: Optional[float] = None
        self._p: Optional[float] = None
        self._a: Optional[float] = None
        self._i: Optional[float] = None
        self._e: Optional[float] = None
        self._w: Optional[float] = None
        self._lan: Optional[float] = None
        self._timing: str = "tc"
        self._derivatives: bool = derivatives
        self._knot_placement: str = knot_placement
        self._grid_e: float = self._KNOT_GRID_E_FLOOR

        self._points, self._change_times, self._dt, self._tptable = \
            create_knots(npt, self._grid_e, knot_placement)

    def _select(self, serial, par, times, nmin):
        """Pick the serial kernel or its prange twin for this evaluation.

        The twin is used only when the instance was constructed with
        ``parallel=True`` and ``times`` is an array with at least ``nmin``
        samples; below that the serial kernel is faster.
        """
        if self._parallel and isinstance(times, ndarray) and times.size >= nmin:
            return par
        return serial

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

    def set_pars(self, *, tc=None, tp=None, p, a, i, e, w, lan=0.0):
        """Bind orbital parameters and (re-)compute the per-knot Taylor coefficients.

        The time anchor is specified by exactly one of ``tc`` (transit
        center) or ``tp`` (periastron passage); the two are related by
        :math:`t_p = t_c - M_\\mathrm{tr}(e, w) \\cdot p / (2\\pi)`.
        Whichever is supplied, the other is derived and stored, so the
        diagnostic paths that need ``t_c`` (e.g. the Newton-Raphson
        reference) and the orbit3d / orbit3dd dispatchers that need
        ``t_pa`` are both satisfied from one call.

        All parameters are keyword-only, so the call site always names the
        convention explicitly.

        Parameters
        ----------
        tc : float, optional
            Time of inferior conjunction (transit center) [days].
        tp : float, optional
            Time of periastron passage [days].
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
        lan : float, optional
            Longitude of the ascending node [radians]. A constant rotation of
            the sky-plane (x, y) coordinates about the line of sight; the
            line-of-sight (z) coordinate is unaffected. Defaults to 0.0. In
            derivative mode the gradient w.r.t. ``lan`` is returned as the
            seventh orbital-parameter column.

        Raises
        ------
        TypeError
            If both ``tc`` and ``tp`` are supplied, or if neither is.

        Notes
        -----
        After this call, ``self._tc`` holds the transit-center time and
        ``self._tp`` holds the periastron-anchor time, regardless of which
        input was used.

        For the ``'ea'`` and ``'ta'`` knot placements the knot grid is
        rebuilt when ``max(e, 0.2)`` differs from the eccentricity the
        current grid was built for by more than ``_KNOT_GRID_E_TOL``, so
        the knots stay clustered near periastron as the bound orbit
        changes. Small eccentricity jitter (e.g. between MCMC steps) does
        not trigger a rebuild.
        """
        if tc is not None and tp is not None:
            raise TypeError("set_pars accepts exactly one of `tc` (transit center) "
                            "or `tp` (periastron passage), not both.")
        if tc is None and tp is None:
            raise TypeError("set_pars requires either `tc` (transit center) "
                            "or `tp` (periastron passage).")
        to = mean_anomaly_at_transit(e, w) / TWO_PI * p
        if tc is not None:
            self._tc = tc
            self._tp = tc - to
            self._timing = "tc"
        else:
            self._tp = tp
            self._tc = tp + to
            self._timing = "tp"
        self._p = p
        self._a = a
        self._i = i
        self._e = e
        self._w = w
        self._lan = lan

        # Rebuild the knot grid if the bound eccentricity has drifted too far
        # from the eccentricity the current grid was built for. The 'mm' grid
        # is eccentricity-independent and never rebuilt.
        if self._knot_placement != "mm":
            e_grid = max(e, self._KNOT_GRID_E_FLOOR)
            if abs(e_grid - self._grid_e) > self._KNOT_GRID_E_TOL:
                self._points, self._change_times, self._dt, self._tptable = \
                    create_knots(self.npt, e_grid, self._knot_placement)
                self._grid_e = e_grid

        if self._derivatives:
            self._coeffs, self._dcoeffs = solve3d_orbit_d(self._points, p, a, i, e, w,
                                                          lan=lan, npt=self.npt)
            # When bound with tp, reparametrise the per-knot gradient from the
            # transit-centre basis (the solver's native basis) to the periastron
            # basis. Each knot slice of _dcoeffs is replaced with its
            # reparametrised copy; because every derivative-returning method
            # reads _dcoeffs, the new basis propagates to all of them.
            if self._timing == "tp":
                for kn in range(self.npt):
                    self._dcoeffs[kn] = tc_to_tp_gradient(self._dcoeffs[kn], p, e, w)
        else:
            self._coeffs = solve3d_orbit(self._points, p, a, i, e, w, lan=lan, npt=self.npt)

    def mean_anomaly(self):
        """Mean anomaly at every bound time, wrapped into :math:`[0, 2\\pi)`.

        Computed analytically from ``self._tc`` (transit-center time) via the
        mean-anomaly-at-transit offset, so this method does not use the
        Taylor backend at all.

        Returns
        -------
        m : ndarray, shape (N,)
            Mean anomaly per time in radians, in :math:`[0, 2\\pi)`.
        """
        offset = mean_anomaly_at_transit(self._e, self._w)
        return mod(TWO_PI * (self.times - (self._tc - offset * self._p / TWO_PI)) / self._p, TWO_PI)

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
        df : ndarray, shape (N, 7)
            Gradient w.r.t. ``(tc, p, a, i, e, w)``. Only returned when
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
            return ta_newton_v(self.times, self._tc, self._p, self._e, self._w)
        ev = eccentricity_vector(self._i, self._e, self._w)
        if self._derivatives:
            fn = self._select(true_anomaly_od, _true_anomaly_ovdp, self.times, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tp, self._p, ev[0], ev[1], ev[2], self._w, self._dt,
                      self._tptable, self._points, self._coeffs, self._dcoeffs, )
        fn = self._select(true_anomaly_o, _true_anomaly_ovp, self.times, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tp, self._p, ev[0], ev[1], ev[2], self._w, self._dt, self._tptable,
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
        dxs, dys, dzs : ndarray, shape (N, 7)
            Gradients w.r.t. ``(tc, p, a, i, e, w)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            fn = self._select(pos_od, _pos_ovdp, times, self._PARALLEL_NMIN_GRAD)
            return fn(times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs,
                      self._dcoeffs, )
        fn = self._select(pos_o, _pos_ovp, times, self._PARALLEL_NMIN_VALUE)
        return fn(times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs)

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
        xt, yt, zt = xyz_newton_v(self.times, self._tc, self._p, self._a, self._i, self._e, self._w)
        return x - xt, y - yt, z - zt

    def vxyz(self):
        """Planet (vx, vy, vz) velocity in the sky frame.

        Returns
        -------
        vxs, vys, vzs : ndarray, shape (N,)
            Velocity components in :math:`R_\\star/\\mathrm{day}`.
        dvxs, dvys, dvzs : ndarray, shape (N, 7)
            Gradients w.r.t. ``(tc, p, a, i, e, w)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        if self._derivatives:
            fn = self._select(vel_od, _vel_ovdp, self.times, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs,
                      self._dcoeffs, )
        fn = self._select(vel_o, _vel_ovp, self.times, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def cos_phase(self):
        """Cosine of the phase angle, :math:`\\cos\\alpha = -z/r`.

        Equals ``+1`` at superior conjunction (full phase, planet behind
        star) and ``-1`` at inferior conjunction (new phase, planet in
        front).

        Returns
        -------
        ca : ndarray, shape (N,)
            Cosine of the phase angle per time, in :math:`[-1, 1]`.
        dca : ndarray, shape (N, 7)
            Gradient w.r.t. ``(tc, p, a, i, e, w)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        if self._derivatives:
            fn = self._select(cos_alpha_od, _cos_alpha_ovdp, self.times, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs,
                      self._dcoeffs, )
        fn = self._select(cos_alpha_o, _cos_alpha_ovp, self.times, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs)

    def _cos_phase_error(self):
        """Per-time phase-angle-cosine residuals against the Newton-Raphson reference.

        Diagnostic helper. Currently compares the Taylor backend's
        ``cos_phase`` against the *true anomaly* from
        :func:`~meepmeep.backends.numba.newton.newton.ta_newton_v` — i.e.
        a coarse sanity check, not a tight residual.
        """
        ta = ta_newton_v(self.times, self._tc, self._p, self._e, self._w)
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
        dph : ndarray, shape (N, 7)
            Gradient w.r.t. ``(tc, p, a, i, e, w)``. Only returned when
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
            ca, dca = self.cos_phase()
            ca_c = clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
            ph = arccos(ca_c)
            inv_s = -1.0 / sqrt(1.0 - ca_c * ca_c)
            dph = inv_s[:, None] * dca
            return ph, dph
        return arccos(self.cos_phase())

    def theta(self):
        """Supplement angle :math:`\\theta = \\arccos(-\\cos\\alpha) = \\pi - \\alpha`.

        Useful when the natural reference direction is the line from the
        star to the observer rather than the line from the planet to the
        observer.

        Returns
        -------
        th : ndarray, shape (N,)
            Supplement angle per time in radians, in :math:`[0, \\pi]`.
        dth : ndarray, shape (N, 7)
            Gradient w.r.t. ``(tc, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``.

        Notes
        -----
        See :meth:`phase` for the derivative-clamping caveat near
        :math:`|\\cos\\alpha| = 1`.
        """
        if self._derivatives:
            ca, dca = self.cos_phase()
            ca_c = clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
            th = arccos(-ca_c)
            # d(arccos(-c))/dθ = +dc/dθ / sqrt(1 - c²)
            inv_s = 1.0 / sqrt(1.0 - ca_c * ca_c)
            dth = inv_s[:, None] * dca
            return th, dth
        return arccos(-self.cos_phase())

    def star_planet_distance(self, times: Optional[ndarray] = None):
        """3D star-planet separation :math:`r = \\sqrt{x^2 + y^2 + z^2}`.

        Distinct from :meth:`projected separation` (currently only available
        through the low-level ``sep_o`` dispatcher), which drops the
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
        dr : ndarray, shape (N, 7)
            Gradient w.r.t. ``(tc, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            fn = self._select(star_planet_distance_od, _star_planet_distance_ovdp, times, self._PARALLEL_NMIN_GRAD)
            return fn(times, self._tp, self._p, self._dt, self._tptable, self._points,
                      self._coeffs, self._dcoeffs, )
        fn = self._select(star_planet_distance_o, _star_planet_distance_ovp, times, self._PARALLEL_NMIN_VALUE)
        return fn(times, self._tp, self._p, self._dt, self._tptable, self._points, self._coeffs)

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
        dltt : ndarray, shape (N, 7)
            Gradient w.r.t. ``(tc, p, a, i, e, w)``. Only returned when
            ``self._derivatives`` is ``True``. The derivative w.r.t.
            ``rstar`` is intentionally *not* returned (per package spec).
        """
        if self._derivatives:
            fn = self._select(light_travel_time_od, _light_travel_time_ovdp, self.times, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tp, self._p, self._e, self._w, rstar, self._dt,
                      self._tptable, self._points, self._coeffs, self._dcoeffs, )
        fn = self._select(light_travel_time_o, _light_travel_time_ovp, self.times, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tp, self._p, self._e, self._w, rstar, self._dt, self._tptable,
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
        drvs : ndarray, shape (N, 8)
            Gradient w.r.t. ``(tc, p, a, i, e, w, k)``. Only returned
            when ``self._derivatives`` is ``True``.
        """
        if self._derivatives:
            fn = self._select(rv_od, _rv_ovdp, self.times, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, k, self._tp, self._p, self._a, self._i, self._e, self._dt, self._tptable,
                          self._points, self._coeffs, self._dcoeffs, )
        fn = self._select(rv_o, _rv_ovp, self.times, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, k, self._tp, self._p, self._a, self._i, self._e, self._dt, self._tptable, self._points,
                     self._coeffs, )

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
        dflux : ndarray, shape (N, 9)
            Gradient w.r.t. ``(tc, p, a, i, e, w, ag, k)``. Only
            returned when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            fn = self._select(lambert_phase_curve_od, _lambert_phase_curve_ovdp, times, self._PARALLEL_NMIN_GRAD)
            return fn(times, ag, self._a, k, self._tp, self._p, self._dt, self._tptable,
                                           self._points, self._coeffs, self._dcoeffs, )
        fn = self._select(lambert_phase_curve_o, _lambert_phase_curve_ovp, times, self._PARALLEL_NMIN_VALUE)
        return fn(times, ag, self._a, k, self._tp, self._p, self._dt, self._tptable, self._points,
                                      self._coeffs, )

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
        dev : ndarray, shape (N, 10)
            Gradient w.r.t.
            ``(tc, p, a, i, e, w, alpha, mass_ratio, inc)``. Only
            returned when ``self._derivatives`` is ``True``.
        """
        times = times if times is not None else self.times
        if self._derivatives:
            fn = self._select(ev_signal_od, _ev_signal_ovdp, times, self._PARALLEL_NMIN_GRAD)
            return fn(alpha, mass_ratio, self._i, times, self._tp, self._p, self._dt, self._tptable,
                                 self._points, self._coeffs, self._dcoeffs, )
        fn = self._select(ev_signal_o, _ev_signal_ovp, times, self._PARALLEL_NMIN_VALUE)
        return fn(alpha, mass_ratio, self._i, times, self._tp, self._p, self._dt, self._tptable, self._points,
                            self._coeffs, )

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
            xt, yt, zt = xyz_newton_v(self.times, self._tc, self._p, self._a, self._i, self._e, self._w)
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
