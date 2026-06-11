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

from ..point3dd.position import _pos_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _pos_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz):
    """Write-into orbit kernel: epoch fold, knot lookup, and evaluation.

    Writes the seven-parameter gradients into the caller-provided ``(7,)``
    buffers ``dx``, ``dy``, and ``dz`` and returns the position values.
    The vector kernels here and in the derived-quantity modules pass
    preallocated output rows or reusable scratch buffers instead of
    allocating per sample.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return _pos_cd_w(tc - points[ix] * p, coeffs[ix], dcoeffs[ix], dx, dy, dz)


@njit(fastmath=True)
def _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`pos_od`. See that function for documentation."""
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    x, y, z = _pos_ow(t, tpa, p, dt, pktable, points, coeffs, dcoeffs, dx, dy, dz)
    return x, y, z, dx, dy, dz


@njit(fastmath=True)
def _pos_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`pos_od`. See that function for documentation."""
    n = times.size
    xs = zeros(n)
    ys = zeros(n)
    zs = zeros(n)
    dxs = zeros((n, 7))
    dys = zeros((n, 7))
    dzs = zeros((n, 7))
    for j in range(n):
        xs[j], ys[j], zs[j] = _pos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs,
                                      dxs[j], dys[j], dzs[j])
    return xs, ys, zs, dxs, dys, dzs


def pos_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (x, y, z) position and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_pos_osd`) or vector (:func:`_pos_ovd`) kernel at compile time
    (inside ``@njit``) or at call time (pure Python).

    Parameters
    ----------
    t : float or ndarray
        Time at which to evaluate the position and gradient.
    tpa : float
        Periastron time anchoring the knot grid. Note the convention
        difference: the high-level :class:`~meepmeep.orbit.Orbit` API
        takes the transit-center time as ``tc`` and converts it to
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
    px, py, pz : float or ndarray
        Sky-frame position components in units of the stellar radius.
        Arrays of shape (N,) for an array ``t``.
    dpx, dpy, dpz : ndarray
        Gradients w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a
        scalar ``t``, (N, 7) for an array ``t``.
    """
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
