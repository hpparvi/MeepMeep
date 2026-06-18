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


from numpy import ndarray

from .backends.numba.point3d import (
    solve3d,
    pos, zpos, sep, vel, zvel, rv, cos_alpha,
    lambert_phase_curve, ev_signal, emission_phase_curve,
    pos_vp, zpos_vp, sep_vp, vel_vp, zvel_vp, cos_alpha_vp,
    lambert_phase_curve_vp, ev_signal_vp, emission_phase_curve_vp,
    find_contact_point, bounding_box, find_z_min,
    t12, t14, t23, t34,
)
from .backends.numba.point3dd import (
    solve3d_d,
    pos_d, zpos_d, sep_d, vel_d, zvel_d, rv_d, cos_alpha_d,
    lambert_phase_curve_d, ev_signal_d, emission_phase_curve_d,
    pos_d_vp, zpos_d_vp, sep_d_vp, vel_d_vp, zvel_d_vp, rv_d_vp, cos_alpha_d_vp,
    lambert_phase_curve_d_vp, ev_signal_d_vp, emission_phase_curve_d_vp,
)


class Expansion3D:

    # Minimum time-array sizes for which the prange kernel twins beat the
    # serial kernels (measured on a 16-core machine). The single-expansion-point
    # value kernels run at a few ns per sample, so their break-even is higher
    # than for the gradient kernels. Mirrors Expansion2D.
    _PARALLEL_NMIN_VALUE = 100_000
    _PARALLEL_NMIN_GRAD = 10_000

    def __init__(self, tc: float, p: float, a: float, i: float, e: float, w: float,
                 lan: float = 0.0, te: float = 0.0, derivatives: bool = False,
                 parallel: bool = False):
        """High-level wrapper over the single-expansion-point 3D Taylor evaluators.

        The 3D counterpart of :class:`~meepmeep.expansion2d.Expansion2D`. An
        *expansion point* is a point along the orbit that serves as the center
        of a local 5th-order Taylor expansion of the planet's trajectory in
        time. This class builds one such expansion at a chosen expansion-point
        time ``te`` and exposes the full single-expansion-point 3D surface from
        the ``point3d`` / ``point3dd`` backends:

        * geometry: the 3D ``(x, y, z)`` position, the line-of-sight ``z``
          coordinate, the sky-projected separation between the centers of the
          star and planet (in units of the stellar radius), the velocity
          vector, the line-of-sight velocity, and the cosine of the orbital
          phase angle;
        * observables: the stellar radial velocity, the Lambertian
          reflected-light phase curve, the ellipsoidal-variation signal, and
          the cosine emission phase curve;
        * transit geometry: contact points, durations, and the minimum
          projected separation.

        All quantities share the single ``(3, 5)`` coefficient matrix solved by
        :meth:`set_pars` (and, in derivative mode, the ``(7, 3, 5)`` derivative
        tensor).

        Usage mirrors :class:`~meepmeep.orbit.Orbit` and
        :class:`~meepmeep.expansion2d.Expansion2D`: construct once, then rebind
        orbital elements with :meth:`set_pars` and the observation times with
        :meth:`set_data` as needed. The constructor forwards its orbital
        arguments to :meth:`set_pars`. The evaluation methods read the bound
        time grid directly; the underlying direct evaluators take the
        transit-centre ``tc`` and the expansion-point offset ``te`` and
        epoch-fold around the expansion point internally.

        The ``derivatives`` flag is a once-per-instance switch: when ``True``,
        the evaluation methods return each value together with its
        orbital-parameter derivatives; when ``False`` they return the value
        only.

        Parameters
        ----------
        tc : float
            Time of inferior conjunction (transit centre) [days], in absolute
            time.
        p : float
            Orbital period [days].
        a : float
            Scaled semi-major axis [R_star].
        i : float
            Inclination [rad].
        e : float
            Eccentricity.
        w : float
            Argument of periastron [rad].
        lan : float, optional
            Longitude of the ascending node [rad], a constant rotation of the
            sky plane about the line of sight. Defaults to 0.0.
        te : float, optional
            Expansion-point time [days], measured relative to the transit
            centre (time of inferior conjunction). ``te = 0`` (the default)
            expands the series at the transit centre. The expansion-point time
            is fixed for the lifetime of the instance; rebinding via
            :meth:`set_pars` reuses it.
        derivatives : bool, optional
            If ``True``, the evaluation methods also return parameter
            derivatives. Defaults to ``False``.
        parallel : bool, optional
            If ``True``, evaluation methods route time grids with at least
            ``_PARALLEL_NMIN_GRAD`` (derivative mode) or
            ``_PARALLEL_NMIN_VALUE`` (value mode) samples to multi-threaded
            ``prange`` kernel twins; smaller grids always take the serial
            kernels, which are faster below those sizes. The results are
            identical either way. Defaults to ``False``: leave it off when the
            surrounding application already parallelises at process level (e.g.
            one process per MCMC chain), where nested thread pools
            oversubscribe the machine.
        """
        self._derivatives = derivatives
        self._parallel = parallel
        self.te = te
        self.times = None
        self.set_pars(tc=tc, p=p, a=a, i=i, e=e, w=w, lan=lan)

    def set_pars(self, *, tc: float, p: float, a: float, i: float, e: float, w: float,
                 lan: float = 0.0):
        """Bind orbital elements and (re-)solve the single-expansion-point Taylor coefficients.

        All parameters are keyword-only, so the call site always names the
        elements explicitly. The expansion-point time ``te`` is a
        construction-time constant and is reused on every call.

        Parameters
        ----------
        tc : float
            Time of inferior conjunction (transit centre) [days].
        p : float
            Orbital period [days].
        a : float
            Scaled semi-major axis [R_star].
        i : float
            Inclination [rad].
        e : float
            Eccentricity.
        w : float
            Argument of periastron [rad].
        lan : float, optional
            Longitude of the ascending node [rad]. A constant rotation of the
            sky-plane (x, y) coordinates about the line of sight (the
            line-of-sight z is unaffected). Defaults to 0.0. In derivative mode
            the gradient w.r.t. ``lan`` is the seventh orbital-parameter column.

        Notes
        -----
        After this call, ``self._coeffs`` holds the ``(3, 5)`` coefficient
        matrix (and ``self._dcoeffs`` the ``(7, 3, 5)`` derivative tensor when
        the instance is in derivative mode), and ``self._ep_time`` holds the
        absolute time of the expansion point (``tc + te``).
        """
        self._tc = tc
        self._p = p
        self._a = a
        self._i = i
        self._e = e
        self._w = w
        self._lan = lan

        # Absolute time of the expansion point, used to convert centered
        # contact-point offsets back to absolute times.
        self._ep_time = tc + self.te

        if self._derivatives:
            self._coeffs, self._dcoeffs = solve3d_d(self.te, p, a, i, e, w, lan)
        else:
            self._coeffs = solve3d(self.te, p, a, i, e, w, lan)
            self._dcoeffs = None

    def set_data(self, times):
        """Bind a time grid evaluated by the per-quantity methods.

        Parameters
        ----------
        times : ndarray, shape (N,)
            Absolute observation times [days] at which the evaluation methods
            (:meth:`position`, :meth:`projected_separation`, ...) evaluate the
            orbit.
        """
        self.times = times

    def _select(self, serial, par, nmin):
        """Pick the serial kernel or its prange twin for this evaluation.

        The twin is used only when the instance was constructed with
        ``parallel=True`` and the bound grid is an array with at least
        ``nmin`` samples; below that the serial kernel is faster.
        """
        if self._parallel and isinstance(self.times, ndarray) and self.times.size >= nmin:
            return par
        return serial

    # ------------------------------------------------------------------
    # Geometry on the bound time grid
    # ------------------------------------------------------------------
    def position(self):
        """Planet (x, y, z) position at the times bound via :meth:`set_data`.

        Returns
        -------
        tuple
            ``(xs, ys, zs)`` if the instance was created with
            ``derivatives=False``; ``(xs, ys, zs, dxs, dys, dzs)`` otherwise,
            where the gradients are shape ``(N, 7)`` arrays of partial
            derivatives with respect to ``(tc, p, a, i, e, w, lan)``. ``xs``,
            ``ys`` are the sky-plane coordinates and ``zs`` the line-of-sight
            depth (positive toward the observer); all in units of the stellar
            radius.
        """
        if self._derivatives:
            fn = self._select(pos_d, pos_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(pos, pos_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    def z_position(self):
        """Line-of-sight (z) position at the times bound via :meth:`set_data`.

        Returns
        -------
        zs : ndarray, shape (N,)
            Line-of-sight depth per time (positive toward the observer), in
            units of the stellar radius. Returned alone if
            ``derivatives=False``.
        dzs : ndarray, shape (N, 7)
            Only returned if ``derivatives=True``: partial derivatives of
            ``zs`` with respect to ``(tc, p, a, i, e, w, lan)``.
        """
        if self._derivatives:
            fn = self._select(zpos_d, zpos_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(zpos, zpos_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    def projected_separation(self):
        """Sky-projected star-planet separation at the times bound via :meth:`set_data`.

        The sky-projected separation between the centers of the star and
        planet, in units of the stellar radius.

        Returns
        -------
        d : ndarray, shape (N,)
            Projected separation per time, returned alone if
            ``derivatives=False``.
        dd : ndarray, shape (N, 7)
            Only returned if ``derivatives=True``: partial derivatives of ``d``
            with respect to ``(tc, p, a, i, e, w, lan)``.
        """
        if self._derivatives:
            fn = self._select(sep_d, sep_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(sep, sep_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    def velocity(self):
        """Planet (vx, vy, vz) velocity at the times bound via :meth:`set_data`.

        Returns
        -------
        tuple
            ``(vxs, vys, vzs)`` if ``derivatives=False``; otherwise
            ``(vxs, vys, vzs, dvxs, dvys, dvzs)`` with shape-``(N, 7)``
            gradients with respect to ``(tc, p, a, i, e, w, lan)``. Velocities
            are in units of stellar radii per day.
        """
        if self._derivatives:
            fn = self._select(vel_d, vel_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(vel, vel_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    def z_velocity(self):
        """Line-of-sight (z) velocity at the times bound via :meth:`set_data`.

        Returns
        -------
        vz : ndarray, shape (N,)
            Line-of-sight velocity per time, in units of stellar radii per day.
            Returned alone if ``derivatives=False``.
        dvz : ndarray, shape (N, 7)
            Only returned if ``derivatives=True``: partial derivatives of
            ``vz`` with respect to ``(tc, p, a, i, e, w, lan)``.
        """
        if self._derivatives:
            fn = self._select(zvel_d, zvel_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(zvel, zvel_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    def cos_phase(self):
        """Cosine of the orbital phase angle at the times bound via :meth:`set_data`.

        :math:`\\cos\\alpha = -z / r`, equal to ``+1`` at superior conjunction
        (full phase, planet behind the star) and ``-1`` at inferior conjunction
        (new phase, planet in front).

        Returns
        -------
        ca : ndarray, shape (N,)
            Cosine of the phase angle per time, in :math:`[-1, 1]`. Returned
            alone if ``derivatives=False``.
        dca : ndarray, shape (N, 7)
            Only returned if ``derivatives=True``: partial derivatives of
            ``ca`` with respect to ``(tc, p, a, i, e, w, lan)``.
        """
        if self._derivatives:
            fn = self._select(cos_alpha_d, cos_alpha_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(cos_alpha, cos_alpha_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    # ------------------------------------------------------------------
    # Observables on the bound time grid
    # ------------------------------------------------------------------
    def radial_velocity(self, k: float):
        """Stellar radial velocity at the times bound via :meth:`set_data`.

        Scales the planet's line-of-sight velocity by the closed-form
        Perryman (2018, Eq. 2.23) factor so the result is the observed stellar
        RV.

        Parameters
        ----------
        k : float
            Radial-velocity semi-amplitude of the star, in physical velocity
            units (e.g. m/s); the output inherits these units.

        Returns
        -------
        rvs : ndarray, shape (N,)
            Radial velocity per time, in the units of ``k``. Returned alone if
            ``derivatives=False``.
        drvs : ndarray, shape (N, 7)
            Only returned if ``derivatives=True``: partial derivatives of
            ``rvs`` with respect to ``(tc, p, a, i, e, w, lan)``.

        Notes
        -----
        The value-path 3D radial velocity has no dedicated parallel kernel
        (``rv`` is scalar-inline and routes arrays through the ``zvel`` vector
        kernel), so value-mode evaluation ignores the ``parallel`` flag.
        """
        if self._derivatives:
            fn = self._select(rv_d, rv_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, k, self._tc, self._p, self._a, self._i, self._e,
                      self._coeffs, self._dcoeffs, self.te)
        return rv(self.times, k, self._tc, self._p, self._a, self._i, self._e, self._coeffs, self.te)

    def lambert_phase_curve(self, ag: float, k: float):
        """Reflected-light (Lambertian) phase curve at the times bound via :meth:`set_data`.

        Parameters
        ----------
        ag : float
            Geometric albedo.
        k : float
            Planet-to-star radius ratio :math:`R_p / R_\\star`.

        Returns
        -------
        flux : ndarray, shape (N,)
            Reflected planet-to-star flux ratio per time. Returned alone if
            ``derivatives=False``.
        dflux : ndarray, shape (N, 9)
            Only returned if ``derivatives=True``: partial derivatives of
            ``flux`` with respect to ``(tc, p, a, i, e, w, lan, ag, k)``.
        """
        if self._derivatives:
            fn = self._select(lambert_phase_curve_d, lambert_phase_curve_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, ag, k, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(lambert_phase_curve, lambert_phase_curve_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, ag, k, self._tc, self._p, self._coeffs, self.te)

    def ellipsoidal_variation(self, alpha: float, mass_ratio: float):
        """Ellipsoidal-variation signal at the times bound via :meth:`set_data`.

        Relative flux variation induced by the tidally distorted primary
        (Lillo-Box et al. 2014). The orbital inclination is taken from the
        bound parameters and need not be passed.

        Parameters
        ----------
        alpha : float
            Gravity-darkening coefficient.
        mass_ratio : float
            Planet-to-star mass ratio :math:`M_p / M_\\star`.

        Returns
        -------
        ev : ndarray, shape (N,)
            Relative flux variation per time. Returned alone if
            ``derivatives=False``.
        dev : ndarray, shape (N, 9)
            Only returned if ``derivatives=True``: partial derivatives of
            ``ev`` with respect to ``(tc, p, a, i, e, w, lan, alpha, mass_ratio)``.
        """
        if self._derivatives:
            fn = self._select(ev_signal_d, ev_signal_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, alpha, mass_ratio, self._i, self._tc, self._p,
                      self._coeffs, self._dcoeffs, self.te)
        fn = self._select(ev_signal, ev_signal_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, alpha, mass_ratio, self._i, self._tc, self._p, self._coeffs, self.te)

    def emission_phase_curve(self, k: float, fratio: float, offset: float):
        """Thermal-emission (cosine model) phase curve at the times bound via :meth:`set_data`.

        Parameters
        ----------
        k : float
            Planet-to-star radius ratio :math:`R_p / R_\\star`.
        fratio : float
            Dayside-to-nightside per-surface-element flux ratio, scaling the
            phase-curve amplitude.
        offset : float
            Hotspot offset [radians], shifting the peak away from secondary
            eclipse.

        Returns
        -------
        flux : ndarray, shape (N,)
            Emitted planet-to-star flux ratio per time. Returned alone if
            ``derivatives=False``.
        dflux : ndarray, shape (N, 10)
            Only returned if ``derivatives=True``: partial derivatives of
            ``flux`` with respect to
            ``(tc, p, a, i, e, w, lan, k, fratio, offset)``.
        """
        if self._derivatives:
            fn = self._select(emission_phase_curve_d, emission_phase_curve_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, k, fratio, offset, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(emission_phase_curve, emission_phase_curve_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, k, fratio, offset, self._tc, self._p, self._coeffs, self.te)

    # ------------------------------------------------------------------
    # Transit geometry (from the coefficient matrix, no time grid needed)
    # ------------------------------------------------------------------
    def duration(self, k: float, kind: int = 14) -> float:
        """Transit duration of the requested type [days].

        Parameters
        ----------
        k : float
            Planet-to-star radius ratio.
        kind : int, optional
            Which duration to return: 14 (total, first-to-fourth contact;
            the default), 23 (full, second-to-third contact), 12 (ingress,
            first-to-second contact), or 34 (egress, third-to-fourth
            contact).

        Returns
        -------
        float
            The requested transit duration [days].
        """
        durations = {14: t14, 23: t23, 12: t12, 34: t34}
        if kind not in durations:
            raise ValueError("kind must be one of 14, 23, 12, or 34.")
        return durations[kind](k, self._coeffs)

    def contact_point(self, k: float, point: int) -> float:
        """Absolute time of a transit contact point [days].

        Parameters
        ----------
        k : float
            Planet-to-star radius ratio.
        point : int
            Contact point, one of 1, 2, 3, or 4.

        Returns
        -------
        float
            Absolute time of the requested contact point.
        """
        return self._ep_time + find_contact_point(k, point, self._coeffs)

    def bounding_box(self, k: float):
        """Absolute first- and fourth-contact times bracketing the transit.

        Parameters
        ----------
        k : float
            Planet-to-star radius ratio.

        Returns
        -------
        tuple
            ``(T1, T4)`` absolute contact times [days].
        """
        bt1, bt4 = bounding_box(k, self._coeffs)
        return self._ep_time + bt1, self._ep_time + bt4

    def min_separation(self, guess: float = 0.0):
        """Locate the minimum projected separation near the expansion point.

        Parameters
        ----------
        guess : float, optional
            Initial guess for the time of minimum separation, as an offset
            in days from the expansion point. Defaults to 0.0 (the expansion
            point itself).

        Returns
        -------
        t_min : float
            Absolute time of minimum projected separation [days].
        z_min : float
            Projected separation at the minimum, in units of the stellar
            radius.
        """
        t_min, z_min = find_z_min(guess, self._coeffs)
        return self._ep_time + t_min, z_min
