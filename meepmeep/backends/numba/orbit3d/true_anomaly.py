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

"""Multi-knot true-anomaly evaluators."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, pi, floor, sqrt, arccos, ndarray

from ..point3d.position import pos_c
from ._common import _is_1d_array


@njit
def _true_anomaly_os(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """Scalar kernel for :func:`true_anomaly_o`. See that function for documentation."""
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        # Circular-orbit fast path: with the periastron anchor tpa, the true
        # anomaly equals the mean anomaly, f = 2*pi*(t - tpa)/p, folded into
        # [0, 2*pi). This is the e -> 0 limit of the geometric branch below
        # and matches the fast path in orbit3dd.true_anomaly exactly.
        tau = t - tpa
        epoch = floor(tau / p)
        return 2.0 * pi * (tau - epoch * p) / p

    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    tcc = tc - points[ix] * p
    c = coeffs[ix]
    x, y, z = pos_c(tcc, c)
    edp = (x * ex + y * ey + z * ez) / sqrt((x * x + y * y + z * z) * nes)

    if edp <= -1.0:
        return pi
    elif edp >= 1.0:
        return 0.0
    elif tc < 0.5 * p:
        # Branch selection from the mean anomaly: the folded time since
        # periastron gives M = 2*pi*tc/p exactly, and f and M always share
        # the half-plane (both run from 0 at periastron to pi at apoastron),
        # so M selects the arccos branch. The sign of r.v would do the same
        # in exact arithmetic, but it is O(e) and drowns in the Taylor
        # truncation noise for near-circular orbits.
        return arccos(edp)
    else:
        return 2.0 * pi - arccos(edp)


@njit
def _true_anomaly_ov(times, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """Vector kernel for :func:`true_anomaly_o`. See that function for documentation."""
    npt = times.size
    f = zeros(npt)
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        # Circular-orbit fast path; see _true_anomaly_os.
        twopi = 2.0 * pi
        for i in range(npt):
            tau = times[i] - tpa
            epoch = floor(tau / p)
            f[i] = twopi * (tau - epoch * p) / p
    else:
        for i in range(npt):
            t = times[i]
            epoch = floor((t - tpa) / p)
            tc = t - tpa - epoch * p
            ix = pktable[int(floor(tc / (dt * p)))]
            tcc = tc - points[ix] * p
            c = coeffs[ix]
            x, y, z = pos_c(tcc, c)
            edp = (x * ex + y * ey + z * ez) / sqrt((x * x + y * y + z * z) * nes)

            if edp <= -1.0:
                f[i] = pi
            elif edp >= 1.0:
                f[i] = 0.0
            elif tc < 0.5 * p:
                # Branch selection from the mean anomaly; see _true_anomaly_os.
                f[i] = arccos(edp)
            else:
                f[i] = 2.0 * pi - arccos(edp)
    return f


def true_anomaly_o(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """True anomaly at an array of times.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_true_anomaly_os`) or vector (:func:`_true_anomaly_ov`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Computed from the angle between the planet position vector and the
    eccentricity vector :math:`(e_x, e_y, e_z)`. The mean anomaly,
    computed exactly from the periastron anchor, disambiguates the two
    branches of :math:`\\arccos` (the sign of
    :math:`\\mathbf{r}\\cdot\\mathbf{v}` would do the same in exact
    arithmetic, but it is of order ``e`` and is unreliable against the
    Taylor truncation noise for near-circular orbits).

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the true anomaly.
    tpa : float
        Periastron time.
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector pointing from the focus
        toward periastron. ``(-1, 0, 0)`` is the sentinel that
        :func:`~meepmeep.backends.numba.utils.eccentricity_vector`
        returns for near-circular orbits and triggers the fast path.
    w : float
        Argument of periastron [radians]. Kept for signature parity with
        the gradient variant (``true_anomaly_od``); currently unused
        because the eccentricity vector is passed explicitly and the
        circular-orbit fast path needs only ``tpa`` and ``p``.
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    f : float or ndarray
        True anomaly at each input time [radians], in :math:`[0, 2\\pi)`. Arrays of shape (N,) for an array ``t``.

    Notes
    -----
    The circular-orbit fast path skips the geometric chain to avoid
    division by a near-zero :math:`|\\mathbf{e}|`. ``utils.eccentricity_vector``
    emits the ``(-1, 0, 0)`` sentinel when ``e < 1e-5``, and the test
    here matches that sentinel. On the fast path the true anomaly equals
    the mean anomaly, :math:`f = 2\\pi(t - t_\\mathrm{pa})/p` - the same
    closed form used by the gradient variant ``true_anomaly_od``, so the
    two stay in exact agreement for circular orbits.
    """
    if isinstance(t, ndarray):
        return _true_anomaly_ov(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs)
    return _true_anomaly_os(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs)


@overload(true_anomaly_o)
def _true_anomaly_o_overload(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
            return _true_anomaly_ov(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
            return _true_anomaly_os(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs)
        return impl
    return None
