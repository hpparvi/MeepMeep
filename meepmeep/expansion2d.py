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

from .backends.numba.point2d import (solve2d, pos, sep, pos_vp, sep_vp,
                                            t12, t14, t23, t34,
                                            bounding_box, find_contact_point, find_z_min)
from .backends.numba.point2dd import solve2d_d, pos_d, sep_d, pos_d_vp, sep_d_vp


class Expansion2D:

    # Minimum time-array sizes for which the prange kernel twins beat the
    # serial kernels (measured on a 16-core machine). The single-expansion-point value
    # kernels run at a few ns per sample, so their break-even is higher
    # than for the gradient kernels.
    _PARALLEL_NMIN_VALUE = 100_000
    _PARALLEL_NMIN_GRAD = 10_000

    def __init__(self, tc: float, p: float, a: float, i: float, e: float, w: float,
                 lan: float = 0.0, te: float = 0.0, derivatives: bool = False,
                 parallel: bool = False):
        """High-level wrapper over the single-expansion-point 2D Taylor evaluators.

        An *expansion point* is a point along the orbit that serves as the
        center of a local 5th-order Taylor expansion of the planet's
        trajectory in time. This class builds one such expansion at a chosen
        expansion-point time ``te`` and exposes the sky-plane (x, y)
        position, the sky-projected separation between the centers of the
        star and planet (in units of the stellar radius), and the transit
        contact-point / duration utilities, all sharing the single
        ``(2, 5)`` coefficient matrix solved by :meth:`set_pars`.

        Usage mirrors the high-level :class:`~meepmeep.orbit.Orbit` class:
        construct once, then rebind orbital elements with :meth:`set_pars` and
        the observation times with :meth:`set_data` as needed. The
        constructor itself simply forwards its orbital arguments to
        :meth:`set_pars`. The :attr:`position` and :attr:`projected_separation`
        **properties** evaluate the bound time grid directly: the underlying
        direct evaluators take the transit centre ``tc`` and the expansion-point offset
        ``te`` and epoch-fold around the expansion point.

        The ``derivatives`` flag is a once-per-instance switch: when ``True``,
        :attr:`position` and :attr:`projected_separation` return the value
        together with its orbital-parameter derivatives; when ``False`` they
        return the value only.

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
            Expansion-point time [days], measured relative to the transit centre (time of
            inferior conjunction). ``te = 0`` (the default) expands the series
            at the transit centre. The expansion-point time is fixed for the lifetime of
            the instance; rebinding via :meth:`set_pars` reuses it.
        derivatives : bool, optional
            If ``True``, :attr:`position` and :attr:`projected_separation`
            also return parameter derivatives. Defaults to ``False``.
        parallel : bool, optional
            If ``True``, :attr:`position` and :attr:`projected_separation`
            route time grids with at least ``_PARALLEL_NMIN_GRAD``
            (derivative mode) or ``_PARALLEL_NMIN_VALUE`` (value mode)
            samples to multi-threaded ``prange`` kernel twins; smaller
            grids always take the serial kernels, which are faster below
            those sizes. The results are identical either way. Defaults to
            ``False``: leave it off when the surrounding application
            already parallelises at process level (e.g. one process per
            MCMC chain), where nested thread pools oversubscribe the
            machine.
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
        elements explicitly. The expansion-point time ``te`` is a construction-time
        constant and is reused on every call.

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
            sky-plane (x, y) coordinates about the line of sight. Defaults to
            0.0. In derivative mode the gradient w.r.t. ``lan`` is the seventh
            orbital-parameter column.

        Notes
        -----
        After this call, ``self._coeffs`` holds the ``(2, 5)`` coefficient
        matrix (and ``self._dcoeffs`` the ``(7, 2, 5)`` derivative tensor when
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

        # Absolute time of the expansion point, used to convert centered contact-point
        # offsets back to absolute times.
        self._ep_time = tc + self.te

        if self._derivatives:
            self._coeffs, self._dcoeffs = solve2d_d(self.te, p, a, i, e, w, lan)
        else:
            self._coeffs = solve2d(self.te, p, a, i, e, w, lan)
            self._dcoeffs = None

    def _select(self, serial, par, nmin):
        """Pick the serial kernel or its prange twin for this evaluation.

        The twin is used only when the instance was constructed with
        ``parallel=True`` and the bound grid is an array with at least
        ``nmin`` samples; below that the serial kernel is faster.
        """
        if self._parallel and isinstance(self.times, ndarray) and self.times.size >= nmin:
            return par
        return serial

    def set_data(self, times):
        """Bind a time grid evaluated by the position / separation properties.

        Parameters
        ----------
        times : ndarray, shape (N,)
            Absolute observation times [days] at which :attr:`position` and
            :attr:`projected_separation` evaluate the orbit.
        """
        self.times = times

    @property
    def position(self):
        """Sky-plane (x, y) position at the times bound via :meth:`set_data`.

        Returns
        -------
        tuple
            ``(xs, ys)`` if the instance was created with
            ``derivatives=False``; ``(xs, ys, dxs, dys)`` otherwise, where
            ``dxs`` and ``dys`` are shape ``(N, 7)`` arrays of partial
            derivatives with respect to ``(tc, p, a, i, e, w, lan)``. All
            positions are in units of the stellar radius.
        """
        if self._derivatives:
            fn = self._select(pos_d, pos_d_vp, self._PARALLEL_NMIN_GRAD)
            return fn(self.times, self._tc, self._p, self._coeffs, self._dcoeffs, self.te)
        fn = self._select(pos, pos_vp, self._PARALLEL_NMIN_VALUE)
        return fn(self.times, self._tc, self._p, self._coeffs, self.te)

    @property
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
            in days from the expansion point. Defaults to 0.0 (the expansion point itself).

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
