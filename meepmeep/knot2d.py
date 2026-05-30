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


from .backends.numba.taylor.solve2d import solve2d
from .backends.numba.taylor.solve2dd import solve2d_d
from .backends.numba.taylor.position2d import pos, sep
from .backends.numba.taylor.position2dd import pos_d, sep_d
from .backends.numba.taylor.util2d import (t12, t14, t23, t34,
                                           bounding_box, find_contact_point, find_z_min)


class Knot2D:

    def __init__(self, tc: float, p: float, a: float, i: float, e: float, w: float,
                 lan: float = 0.0, tk: float = 0.0, derivatives: bool = False):
        """High-level wrapper over the single-knot 2D Taylor evaluators.

        A *knot* is a point along the orbit that serves as the center of a
        local 5th-order Taylor expansion of the planet's trajectory in time.
        This class builds one such expansion at a chosen knot time ``tk`` and
        exposes the sky-plane (x, y) position, the sky-projected separation
        between the centers of the star and planet (in units of the stellar
        radius), and the transit contact-point / duration utilities, all
        sharing the single ``(2, 5)`` coefficient matrix solved at
        construction. (The name follows spline terminology, but the knot here
        is the expansion *center*, not a segment boundary.)

        The ``derivatives`` flag is a once-per-instance switch: when ``True``,
        :meth:`position` and :meth:`projected_separation` return the value
        together with its orbital-parameter derivatives; when ``False`` they
        return the value only. This mirrors the dispatch pattern of the
        high-level :class:`~meepmeep.orbit.Orbit` class.

        Evaluation methods take **absolute** observation times. Internally the
        instance evaluates at ``t - tc`` so that the underlying direct
        evaluators epoch-fold around the knot.

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
        tk : float, optional
            Knot time [days], measured relative to the transit centre (time of
            inferior conjunction). ``tk = 0`` (the default) expands the series
            at the transit centre.
        derivatives : bool, optional
            If ``True``, :meth:`position` and :meth:`projected_separation`
            also return parameter derivatives. Defaults to ``False``.

        Notes
        -----
        A joint position-and-separation method is intentionally not provided:
        the low-level API has no derivative-returning ``pos_and_sep`` variant,
        so it could not honour the ``derivatives`` flag symmetrically.
        """
        self.derivatives = derivatives
        self.tk = tk
        self.tc = tc
        self.p = p
        self.a = a
        self.i = i
        self.e = e
        self.w = w
        self.lan = lan

        # Absolute time of the knot, used to convert centered contact-point
        # offsets back to absolute times.
        self._knot_time = tc + tk

        if derivatives:
            self._coeffs, self._dcoeffs = solve2d_d(tk, p, a, i, e, w, lan)
        else:
            self._coeffs = solve2d(tk, p, a, i, e, w, lan)
            self._dcoeffs = None

    def position(self, t):
        """Sky-plane (x, y) position at absolute time(s) ``t``.

        Parameters
        ----------
        t : float or ndarray
            Absolute observation time(s) [days].

        Returns
        -------
        tuple
            ``(x, y)`` if the instance was created with ``derivatives=False``;
            ``(x, y, dx, dy)`` otherwise, where ``dx`` and ``dy`` are shape
            ``(7,)`` arrays of partial derivatives with respect to
            ``(tc, p, a, i, e, w, lan)``. All positions are in units of the
            stellar radius.
        """
        if self.derivatives:
            return pos_d(t - self.tc, self.tk, self.p, self._coeffs, self._dcoeffs)
        return pos(t - self.tc, self.tk, self.p, self._coeffs)

    def projected_separation(self, t):
        """Sky-projected star-planet separation at absolute time(s) ``t``.

        The sky-projected separation between the centers of the star and
        planet, in units of the stellar radius.

        Parameters
        ----------
        t : float or ndarray
            Absolute observation time(s) [days].

        Returns
        -------
        d : float or ndarray
            Projected separation, returned alone if ``derivatives=False``.
        dd : ndarray
            Only returned if ``derivatives=True``: shape ``(7,)`` partial
            derivatives of ``d`` with respect to ``(tc, p, a, i, e, w, lan)``.
        """
        if self.derivatives:
            return sep_d(t - self.tc, self.tk, self.p, self._coeffs, self._dcoeffs)
        return sep(t - self.tc, self.tk, self.p, self._coeffs)

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
        return self._knot_time + find_contact_point(k, point, self._coeffs)

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
        return self._knot_time + bt1, self._knot_time + bt4

    def min_separation(self, guess: float = 0.0):
        """Locate the minimum projected separation near the knot.

        Parameters
        ----------
        guess : float, optional
            Initial guess for the time of minimum separation, as an offset
            in days from the knot. Defaults to 0.0 (the knot itself).

        Returns
        -------
        t_min : float
            Absolute time of minimum projected separation [days].
        z_min : float
            Projected separation at the minimum, in units of the stellar
            radius.
        """
        t_min, z_min = find_z_min(guess, self._coeffs)
        return self._knot_time + t_min, z_min
