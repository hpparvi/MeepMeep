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

"""Multi-knot planet (x, y, z) position evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..position3dd import pos_cd
from ._common import _is_1d_array


@njit(fastmath=True)
def _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (x, y, z) position and orbital-parameter derivatives at scalar time.

    Parameters
    ----------
    t : float
        Time at which to evaluate the position and gradient.
    tpa : float
        Periastron time anchoring the knot grid. Note the convention
        difference: the high-level :class:`~meepmeep.orbit.Orbit` API
        takes the transit-center time as ``t0`` and converts it to
        periastron time before calling functions in this module (see
        ``Orbit.__init__``).
    p : float
        Orbital period [days].
    dt : float
        ``pktable`` bucket width in fraction of the period.
    pktable : ndarray of int
        Time-to-knot lookup table.
    points : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]``.
    coeffs : ndarray, shape (npt, 3, 5)
        Per-knot Taylor coefficient matrices from :func:`solve3d_orbit_d`.
    dcoeffs : ndarray, shape (npt, 7, 3, 5)
        Per-knot derivative-coefficient tensors from
        :func:`solve3d_orbit_d`.

    Returns
    -------
    px, py, pz : float
        Sky-frame position components in units of the stellar radius.
    dpx, dpy, dpz : ndarray, shape (7,)
        Gradients w.r.t. ``(phase, p, a, i, e, w, lan)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pos_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _pos_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (x, y, z) position and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the position and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    xs, ys, zs : ndarray, shape (N,)
        Position components per time.
    dxs, dys, dzs : ndarray, shape (N, 7)
        Gradients w.r.t. ``(phase, p, a, i, e, w, lan)`` per time.
    """
    n = times.size
    xs = zeros(n)
    ys = zeros(n)
    zs = zeros(n)
    dxs = zeros((n, 7))
    dys = zeros((n, 7))
    dzs = zeros((n, 7))
    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        xs[j] = x
        ys[j] = y
        zs[j] = z
        for k in range(7):
            dxs[j, k] = dx[k]
            dys[j, k] = dy[k]
            dzs[j, k] = dz[k]
    return xs, ys, zs, dxs, dys, dzs


def pos_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (x, y, z) position with gradients. See :func:`_pos_osd` / :func:`_pos_ovd`."""
    if isinstance(t, ndarray):
        return _pos_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(pos_od, jit_options={'fastmath': True})
def _pos_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _pos_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
