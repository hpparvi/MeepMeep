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

from ..position3d import pos_c
from ..velocity3d import vel_c
from ...utils import mean_anomaly
from ._common import _is_1d_array


@njit
def _true_anomaly_os(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """True anomaly at scalar time.

    Scalar counterpart of :func:`_true_anomaly_ov`. See that function for
    the geometric definition, the sign convention from
    :math:`\\mathbf{r}\\cdot\\mathbf{v}`, and the near-circular fast-path
    sentinel.

    Parameters
    ----------
    t : float
        Time at which to evaluate the true anomaly.
    tpa : float
        Periastron time anchoring the knot grid.
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector. ``(-1, 0, 0)`` triggers the
        circular-orbit fast path (see :func:`_true_anomaly_ov`).
    w : float
        Argument of periastron [radians]. Kept for signature parity with
        :func:`_true_anomaly_ov`; unused (the circular fast path collapses
        to mean anomaly with ``w = 0``).
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    f : float
        True anomaly [radians], in :math:`[0, 2\\pi)`.
    """
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        # Circular-orbit fast path; match the array variant by going through
        # ``mean_anomaly`` (which folds in the transit-offset correction from
        # ``mean_anomaly_at_transit``). The simpler ``2π(t - tpa)/p`` used in
        # the gradient module's fast path is intentionally NOT used here.
        return mean_anomaly(t, tpa, p, 0.0, w)

    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    tcc = tc - points[ix] * p
    c = coeffs[ix]
    x, y, z = pos_c(tcc, c)
    vx, vy, vz = vel_c(tcc, c)
    edp = (x * ex + y * ey + z * ez) / sqrt((x * x + y * y + z * z) * nes)

    if edp <= -1.0:
        return pi
    elif edp >= 1.0:
        return 0.0
    elif (x * vx + y * vy + z * vz) > 0.0:
        return arccos(edp)
    else:
        return 2.0 * pi - arccos(edp)


@njit
def _true_anomaly_ov(times, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """True anomaly at an array of times.

    Computed from the angle between the planet position vector and the
    eccentricity vector :math:`(e_x, e_y, e_z)`. The sign of
    :math:`\\mathbf{r}\\cdot\\mathbf{v}` disambiguates the two branches of
    :math:`\\arccos`.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the true anomaly.
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
        Argument of periastron [radians]. Used only on the
        circular-orbit fast path, where the true anomaly is
        approximated by the mean anomaly.
    dt, pktable, points, coeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    f : ndarray, shape (N,)
        True anomaly at each input time [radians], in :math:`[0, 2\\pi)`.

    Notes
    -----
    The circular-orbit fast path skips the geometric chain to avoid
    division by a near-zero :math:`|\\mathbf{e}|`. ``utils.eccentricity_vector``
    emits the ``(-1, 0, 0)`` sentinel when ``e < 1e-5``, and the test
    here matches that sentinel.
    """
    npt = times.size
    f = zeros(npt)
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        f[:] = mean_anomaly(times, tpa, p, 0.0, w)
    else:
        for i in range(npt):
            t = times[i]
            epoch = floor((t - tpa) / p)
            tc = t - tpa - epoch * p
            ix = pktable[int(floor(tc / (dt * p)))]
            tcc = tc - points[ix] * p
            c = coeffs[ix]
            x, y, z = pos_c(tcc, c)
            vx, vy, vz = vel_c(tcc, c)
            edp = (x * ex + y * ey + z * ez) / sqrt((x * x + y * y + z * z) * nes)

            if edp <= -1.0:
                f[i] = pi
            elif edp >= 1.0:
                f[i] = 0.0
            elif (x * vx + y * vy + z * vz) > 0.0:
                f[i] = arccos(edp)
            else:
                f[i] = 2.0 * pi - arccos(edp)
    return f


def true_anomaly_o(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs):
    """True anomaly. See :func:`_true_anomaly_os` / :func:`_true_anomaly_ov`."""
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
