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

"""Multi-expansion-point phase-angle cosine evaluators with parameter derivatives."""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, floor, ndarray

from ..point3dd.cos_phase_angle import _cos_alpha_cd_w
from ._common import _is_1d_array


@njit(fastmath=True, inline='always')
def _cos_alpha_ow(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dca, dx, dy, dz):
    """Write-into orbit kernel for the phase-angle cosine and its gradient.

    Epoch-folds, looks up the expansion point, and delegates the cosine and
    gradient evaluation to the single-expansion-point
    :func:`~meepmeep.backends.numba.point3dd.cos_phase_angle._cos_alpha_cd_w`.
    Writes the seven-parameter gradient into the caller-provided ``(7,)``
    buffer ``dca`` and returns the cosine. ``dx``, ``dy``, and ``dz`` are
    ``(7,)`` scratch buffers for the position gradients; vector loops
    (here and in ``lambert``) allocate them once and reuse them.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = ep_table[int(floor(tc / (dt * p)))]
    return _cos_alpha_cd_w(tc - ep_times[ix] * p, coeffs[ix], dcoeffs[ix], dca, dx, dy, dz)


@njit(fastmath=True)
def _cos_alpha_osd(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Scalar kernel for :func:`cos_alpha_od`. See that function for documentation."""
    dca = zeros(7)
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    ca = _cos_alpha_ow(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dca, dx, dy, dz)
    return ca, dca


@njit(fastmath=True)
def cos_alpha_ovd(times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Vector kernel for :func:`cos_alpha_od`. See that function for documentation."""
    n = times.size
    cas = zeros(n)
    dcas = zeros((n, 7))
    dx = zeros(7)
    dy = zeros(7)
    dz = zeros(7)
    for j in range(n):
        cas[j] = _cos_alpha_ow(times[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                               dcas[j], dx, dy, dz)
    return cas, dcas


@njit(fastmath=True, parallel=True)
def cos_alpha_ovdp(times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`cos_alpha_ovd`.

    The position-gradient scratch is hoisted per thread; a single shared
    buffer would be a data race under ``prange``.
    """
    n = times.size
    cas = zeros(n)
    dcas = zeros((n, 7))
    nt = get_num_threads()
    dx, dy, dz = zeros((nt, 7)), zeros((nt, 7)), zeros((nt, 7))
    for j in prange(n):
        tid = get_thread_id()
        cas[j] = _cos_alpha_ow(times[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
                               dcas[j], dx[tid], dy[tid], dz[tid])
    return cas, dcas


def cos_alpha_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    """Cosine of the phase angle and orbital-parameter derivatives for any orbital phase.

    Accepts a scalar time ``t`` or a 1-D array of times and dispatches to the
    scalar (:func:`_cos_alpha_osd`) or vector (:func:`cos_alpha_ovd`) kernel at
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
    tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs :
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
        return cos_alpha_ovd(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    return _cos_alpha_osd(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)


@overload(cos_alpha_od, jit_options={'fastmath': True})
def _cos_alpha_od_overload(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return cos_alpha_ovd(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs):
            return _cos_alpha_osd(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        return impl
    return None
