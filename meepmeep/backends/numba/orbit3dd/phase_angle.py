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

"""Multi-knot phase-angle cosine evaluators with parameter derivatives."""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, sqrt, ndarray

from .position import _pos_osd
from ._common import _is_1d_array


@njit(fastmath=True)
def _cos_alpha_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`cos_alpha_od`. See that function for documentation."""
    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    ca = -z / r
    dca = zeros(7)
    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    for k in range(7):
        # d(-z/r)/dθ = -dz/r + z·(x·dx + y·dy + z·dz)/r^3
        dca[k] = -dz[k] * inv_r + z * (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r3
    return ca, dca


@njit(fastmath=True)
def _cos_alpha_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`cos_alpha_od`. See that function for documentation."""
    n = times.size
    cas = zeros(n)
    dcas = zeros((n, 7))
    for j in range(n):
        ca, dca = _cos_alpha_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        cas[j] = ca
        for k in range(7):
            dcas[j, k] = dca[k]
    return cas, dcas


def cos_alpha_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the phase angle and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_cos_alpha_osd`) or vector (:func:`_cos_alpha_ovd`) kernel at
    compile time (inside ``@njit``) or at call time (pure Python).

    With :math:`\\cos\\alpha = -z/r` and :math:`r = \\sqrt{x^2+y^2+z^2}`, the
    gradient is

    .. math::

        \\frac{\\partial(-z/r)}{\\partial \\theta_k}
            = -\\frac{1}{r}\\frac{\\partial z}{\\partial \\theta_k}
              + \\frac{z}{r^3}\\,
                \\Bigl(x\\,\\tfrac{\\partial x}{\\partial \\theta_k}
                       + y\\,\\tfrac{\\partial y}{\\partial \\theta_k}
                       + z\\,\\tfrac{\\partial z}{\\partial \\theta_k}\\Bigr).

    Parameters
    ----------
    t : float or ndarray
        Time at which to evaluate the phase-angle cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`~meepmeep.backends.numba.orbit3dd.position.pos_od`.

    Returns
    -------
    ca : float or ndarray
        Cosine of the phase angle. Arrays of shape (N,) for an array ``t``.
    dca : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a
        scalar ``t``, (N, 7) for an array ``t``.
    """
    if isinstance(t, ndarray):
        return _cos_alpha_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _cos_alpha_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(cos_alpha_od, jit_options={'fastmath': True})
def _cos_alpha_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _cos_alpha_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _cos_alpha_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
