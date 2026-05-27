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
    """Cosine of the phase angle and orbital-parameter derivatives at scalar time.

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
    t : float
        Time at which to evaluate the phase-angle cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    ca : float
        Cosine of the phase angle.
    dca : ndarray, shape (7,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
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
    """Cosine of the phase angle and orbital-parameter derivatives at array of times.

    See :func:`_cos_alpha_osd` for the gradient formula.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the phase-angle cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    cas : ndarray, shape (N,)
        Cosine of the phase angle per time.
    dcas : ndarray, shape (N, 7)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
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
    """Cosine of phase angle with gradients. See :func:`_cos_alpha_osd` / :func:`_cos_alpha_ovd`."""
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
