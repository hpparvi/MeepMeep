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

"""Multi-knot light-travel-time correction evaluators with parameter derivatives."""

from numba import njit, prange, types, get_num_threads, get_thread_id
from numba.extending import overload
from numpy import zeros, pi, floor, ndarray

from ..point3d.zvelocity import zvel_c
from .zposition import _zpos_osd, _zpos_ow
from ..utils import mean_anomaly_at_transit_with_derivatives
from ._common import _is_1d_array


# Time taken by light to traverse one solar radius, in days. Kept in sync
# with ``orbit3d.LTT_DAYS_PER_RSUN``.
LTT_DAYS_PER_RSUN = 2.685885891543453e-05


@njit(fastmath=True)
def _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs):
    """Compute ``z(t_transit)`` and its full chain-rule derivative.

    Helper for the light-travel-time derivatives. The transit time depends
    on the orbital parameters via
    :math:`t_o = M_\\mathrm{tr}(e, w) \\cdot p / (2\\pi)`, so the total
    derivative is

    .. math::

        \\frac{\\mathrm d}{\\mathrm d\\theta_k}\\bigl[z(t_\\mathrm{transit}(\\theta);\\theta)\\bigr]
            = v_z(t_\\mathrm{transit})\\,\\frac{\\mathrm d t_o}{\\mathrm d \\theta_k}
              + \\left.\\frac{\\partial z}{\\partial \\theta_k}\\right|_{t=t_\\mathrm{transit}}

    with the non-zero entries of ``dto`` being ``dto/dp = M_tr / (2π)``,
    ``dto/de = (dM_tr/de) · p / (2π)``, ``dto/dw = (dM_tr/dw) · p / (2π)``,
    and zero for ``phase, a, i``.

    Parameters
    ----------
    tpa : float
        Periastron time anchoring the knot grid (see :func:`_pos_osd`).
    p, e, w : float
        Orbital period [days], eccentricity, argument of periastron [radians].
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    z_tr : float
        Line-of-sight planet coordinate at transit time.
    dz_tr_total : ndarray, shape (7,)
        Full total derivative of ``z(t_transit)`` w.r.t.
        ``(tc, p, a, i, e, w, lan)``.

    Notes
    -----
    The ``phase`` slot inherits the multi-knot caveat documented at module
    level: it reflects a per-knot phase shift at the knot containing
    ``t_transit``, not a global user-facing T0 shift.
    """
    m_tr, dm_tr_de, dm_tr_dw = mean_anomaly_at_transit_with_derivatives(e, w)
    two_pi = 2.0 * pi
    to = m_tr / two_pi * p
    t_transit = tpa + to

    # Evaluate z and its (∂z/∂θ)|_{t=t_transit}.
    z_tr, dz_tr_partial = _zpos_osd(t_transit, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    # Velocity at transit (for the dt_transit/dθ chain term).
    vz_tr = _zvel_os(t_transit, tpa, p, dt, pktable, points, coeffs)

    # dto/dθ: only slots 1 (p), 4 (e), 5 (w) are non-zero.
    dto = zeros(7)
    dto[1] = m_tr / two_pi
    dto[4] = dm_tr_de * p / two_pi
    dto[5] = dm_tr_dw * p / two_pi

    dz_tr_total = zeros(7)
    for k in range(7):
        dz_tr_total[k] = vz_tr * dto[k] + dz_tr_partial[k]
    return z_tr, dz_tr_total


@njit(fastmath=True)
def _zvel_os(t, tpa, p, dt, pktable, points, coeffs):
    """Local z-velocity helper used by ``_ltt_transit_z_and_d``.

    Mirrors ``orbit3d.zvel_os`` but kept private here to avoid a
    cross-module import cycle.

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-velocity.
    tpa, p, dt, pktable, points, coeffs :
        See :func:`_pos_osd` (no ``dcoeffs`` — this is a value-only helper).

    Returns
    -------
    vz : float
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`].
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_c(tc - points[ix] * p, coeffs[ix])


@njit(fastmath=True)
def _light_travel_time_osd(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Scalar kernel for :func:`light_travel_time_od`. See that function for documentation."""
    z_t, dz_t = _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    z_tr, dz_tr = _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    ltt = factor * (z_t - z_tr)
    dltt = zeros(7)
    for k in range(7):
        dltt[k] = factor * (dz_t[k] - dz_tr[k])
    return ltt, dltt


@njit(fastmath=True)
def _light_travel_time_ovd(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Vector kernel for :func:`light_travel_time_od`. See that function for documentation."""
    n = times.size
    ltt = zeros(n)
    dltt = zeros((n, 7))
    factor = -rstar * LTT_DAYS_PER_RSUN
    # Reference (z and its full derivative chain) computed once.
    z_tr, dz_tr = _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    dz = zeros(7)
    for j in range(n):
        z = _zpos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dz)
        ltt[j] = factor * (z - z_tr)
        for k in range(7):
            dltt[j, k] = factor * (dz[k] - dz_tr[k])
    return ltt, dltt


@njit(fastmath=True, parallel=True)
def _light_travel_time_ovdp(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Parallel (prange) twin of :func:`_light_travel_time_ovd`.

    The z-gradient scratch is hoisted per thread; a single shared buffer
    would be a data race under ``prange``. The transit reference and its
    derivative chain are computed once, as in the serial kernel.
    """
    n = times.size
    ltt = zeros(n)
    dltt = zeros((n, 7))
    factor = -rstar * LTT_DAYS_PER_RSUN
    z_tr, dz_tr = _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    dz = zeros((get_num_threads(), 7))
    for j in prange(n):
        dzj = dz[get_thread_id()]
        z = _zpos_ow(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs, dzj)
        ltt[j] = factor * (z - z_tr)
        for kk in range(7):
            dltt[j, kk] = factor * (dzj[kk] - dz_tr[kk])
    return ltt, dltt


def light_travel_time_od(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Light travel time correction with gradients.

    Accepts a scalar time or a 1-D array of times and dispatches to the
    scalar (:func:`_light_travel_time_osd`) or vector
    (:func:`_light_travel_time_ovd`) kernel at compile time (inside ``@njit``)
    or at call time (pure Python).

    The correction is referenced to primary transit:

    .. math::

        \\mathrm{ltt}(t) = -(z(t) - z(t_\\mathrm{transit}))\\,r_\\star\\,(R_\\odot / c)

    where :math:`t_\\mathrm{transit} = t_\\mathrm{pa} + M_\\mathrm{tr}(e, w)\\,p/(2\\pi)`.

    Per spec, the partial derivative w.r.t. ``rstar`` is *not* returned -
    only the seven orbital derivatives in the canonical
    ``(tc, p, a, i, e, w, lan)`` order. The reference ``z(t_transit)`` and
    its parameter derivatives are computed by :func:`_ltt_transit_z_and_d`,
    which includes the chain rule through ``t_transit(p, e, w)`` using
    ``vz(t_transit)``.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the correction and gradient.
    tpa : float
        Periastron time anchoring the knot grid (see :func:`_pos_osd`).
    p : float
        Orbital period [days].
    e : float
        Eccentricity.
    w : float
        Argument of periastron [radians].
    rstar : float
        Stellar radius [R_sun].
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    ltt : float or ndarray
        Light travel time correction [days]. Arrays of shape (N,) for an array
        time argument.
    dltt : ndarray
        Gradient w.r.t. ``(tc, p, a, i, e, w, lan)``. Shape (7,) for a scalar
        time, (N, 7) for an array time.
    """
    if isinstance(t, ndarray):
        return _light_travel_time_ovd(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs)
    return _light_travel_time_osd(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs)


@overload(light_travel_time_od, jit_options={'fastmath': True})
def _light_travel_time_od_overload(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
            return _light_travel_time_ovd(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
            return _light_travel_time_osd(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None
