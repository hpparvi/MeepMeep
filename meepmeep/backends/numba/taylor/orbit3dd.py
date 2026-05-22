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

"""Multi-knot Taylor-series evaluators with parameter derivatives.

Derivative-returning counterparts of the routines in ``orbit3d.py``. Every
function returns both the value and its partial derivatives with respect to
the orbital parameters ``(phase, p, a, i, e, w)`` and any extra physical
inputs the routine takes (appended to the orbital block in argument order).

Coefficient layout:
- ``coeffs`` : ``(npt, 3, 5)`` — Taylor coefficients, as in ``orbit3d``.
- ``dcoeffs`` : ``(npt, 6, 3, 5)`` — derivatives of the Taylor coefficients
  w.r.t. the 6 orbital parameters, produced by ``solve3d_orbit_d``.

Vector evaluators (``*_ovd``) return per-coordinate derivative arrays of
shape ``(N, ndp)`` where ``ndp`` is ``6`` for orbital-only routines and
``6 + n_extra`` for routines with extra physical inputs.
"""

from numba import njit, types
from numba.extending import overload
from numpy import zeros, pi, floor, sqrt, sin, cos, arccos, ndarray

from .velocity3d import zvel_c
from .position3dd import pos_cd, sep_cd, pz_cd
from .velocity3dd import vel_cd, zvel_cd, rv_cd
from .solve3dd import solve3d_d
from ..utils import mean_anomaly_at_transit, mean_anomaly_at_transit_with_derivatives


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

@njit
def solve3d_orbit_d(knot_times, p, a, i, e, w, npt):
    """Pre-compute Taylor and derivative coefficients at every knot of one orbit.

    Derivative-returning counterpart of
    :func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit`. Calls
    :func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d` once per
    interior knot and stacks the resulting ``(3, 5)`` and ``(6, 3, 5)``
    matrices into per-orbit arrays. The last slot is the periodic image of
    the first and is copied rather than recomputed.

    Parameters
    ----------
    knot_times : ndarray, shape (npt,)
        Normalised knot phases in ``[0, 1]``, with ``knot_times[-1]``
        equal to ``knot_times[0] + 1``. Built by
        :func:`~meepmeep.backends.numba.knots.create_knots`.
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity, :math:`0 \\le e < 1`.
    w : float
        Argument of periastron [radians].
    npt : int
        Number of knots, including the periodic-image slot.

    Returns
    -------
    coeffs : ndarray, shape (npt, 3, 5)
        Taylor coefficient matrices at every knot (same layout as in
        :func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit`).
    dcoeffs : ndarray, shape (npt, 6, 3, 5)
        Parameter-derivative tensors at every knot. The second axis is
        ordered ``(phase, p, a, i, e, w)``.

    Notes
    -----
    If you hand-roll ``knot_times`` you must enforce the periodic-image
    contract yourself; ``knots.create_knots`` does this automatically.
    """
    coeffs = zeros((npt, 3, 5))
    dcoeffs = zeros((npt, 6, 3, 5))
    to = mean_anomaly_at_transit(e, w) / (2 * pi) * p
    for ix in range(npt - 1):
        cf, dcf = solve3d_d(p * knot_times[ix] - to, p, a, i, e, w)
        coeffs[ix, :, :] = cf
        dcoeffs[ix, :, :, :] = dcf
    coeffs[-1, :, :] = coeffs[0]
    dcoeffs[-1, :, :, :] = dcoeffs[0]
    return coeffs, dcoeffs


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

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
    dcoeffs : ndarray, shape (npt, 6, 3, 5)
        Per-knot derivative-coefficient tensors from
        :func:`solve3d_orbit_d`.

    Returns
    -------
    px, py, pz : float
        Sky-frame position components in units of the stellar radius.
    dpx, dpy, dpz : ndarray, shape (6,)
        Gradients w.r.t. ``(phase, p, a, i, e, w)``.
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
    dxs, dys, dzs : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    xs = zeros(n)
    ys = zeros(n)
    zs = zeros(n)
    dxs = zeros((n, 6))
    dys = zeros((n, 6))
    dzs = zeros((n, 6))
    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        xs[j] = x
        ys[j] = y
        zs[j] = z
        for k in range(6):
            dxs[j, k] = dx[k]
            dys[j, k] = dy[k]
            dzs[j, k] = dz[k]
    return xs, ys, zs, dxs, dys, dzs


@njit(fastmath=True)
def _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives at scalar time.

    Cheaper than :func:`_pos_osd` when only the line-of-sight coordinate
    and its gradient are needed.

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-coordinate and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    pz : float
        Line-of-sight planet coordinate [stellar radii].
    dpz : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return pz_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _zpos_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-coordinate and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    zs : ndarray, shape (N,)
        Line-of-sight coordinates per time.
    dzs : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    zs = zeros(n)
    dzs = zeros((n, 6))
    for j in range(n):
        z, dz = _zpos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        zs[j] = z
        for k in range(6):
            dzs[j, k] = dz[k]
    return zs, dzs


@njit(fastmath=True)
def _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected planet-star separation and orbital-parameter derivatives at scalar time.

    Returns :math:`\\sqrt{x^2 + y^2}` together with its gradient w.r.t.
    the six orbital parameters. The chain rule
    :math:`\\partial d/\\partial \\theta = (p_x \\partial p_x / \\partial \\theta + p_y \\partial p_y / \\partial \\theta)/d`
    is applied inside :func:`~meepmeep.backends.numba.taylor.position3dd.sep_cd`;
    this dispatcher just locates the knot.

    Parameters
    ----------
    t : float
        Time at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    d : float
        Sky-projected separation [stellar radii].
    dd : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return sep_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _sep_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected planet-star separation and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    ds : ndarray, shape (N,)
        Sky-projected separations per time.
    dds : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    ds = zeros(n)
    dds = zeros((n, 6))
    for j in range(n):
        d, dd = _sep_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        ds[j] = d
        for k in range(6):
            dds[j, k] = dd[k]
    return ds, dds


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives at scalar time.

    Parameters
    ----------
    t : float
        Time at which to evaluate the velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vx, vy, vz : float
        Velocity components in :math:`R_\\star/\\mathrm{day}`.
    dvx, dvy, dvz : ndarray, shape (6,)
        Gradients w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return vel_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _vel_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet (vx, vy, vz) velocity and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vxs, vys, vzs : ndarray, shape (N,)
        Velocity components per time.
    dvxs, dvys, dvzs : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    vxs = zeros(n)
    vys = zeros(n)
    vzs = zeros(n)
    dvxs = zeros((n, 6))
    dvys = zeros((n, 6))
    dvzs = zeros((n, 6))
    for j in range(n):
        vx, vy, vz, dvx, dvy, dvz = _vel_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        vxs[j] = vx
        vys[j] = vy
        vzs[j] = vz
        for k in range(6):
            dvxs[j, k] = dvx[k]
            dvys[j, k] = dvy[k]
            dvzs[j, k] = dvz[k]
    return vxs, vys, vzs, dvxs, dvys, dvzs


@njit(fastmath=True)
def _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives at scalar time.

    Cheaper than :func:`_vel_osd` when only the line-of-sight component is
    needed (e.g. for radial-velocity gradients).

    Parameters
    ----------
    t : float
        Time at which to evaluate the z-velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vz : float
        Line-of-sight velocity [:math:`R_\\star/\\mathrm{day}`].
    dvz : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    return zvel_cd(tc - points[ix] * p, coeffs[ix], dcoeffs[ix])


@njit(fastmath=True)
def _zvel_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity and orbital-parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the z-velocity and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    vzs : ndarray, shape (N,)
        Line-of-sight velocities per time.
    dvzs : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    vzs = zeros(n)
    dvzs = zeros((n, 6))
    for j in range(n):
        vz, dvz = _zvel_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        vzs[j] = vz
        for k in range(6):
            dvzs[j, k] = dvz[k]
    return vzs, dvzs


# ---------------------------------------------------------------------------
# Anomalies and angles
# ---------------------------------------------------------------------------

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
    dca : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    ca = -z / r
    dca = zeros(6)
    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    for k in range(6):
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
    dcas : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    cas = zeros(n)
    dcas = zeros((n, 6))
    for j in range(n):
        ca, dca = _cos_alpha_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        cas[j] = ca
        for k in range(6):
            dcas[j, k] = dca[k]
    return cas, dcas


@njit(fastmath=True)
def _star_planet_distance_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """3D star-planet distance and orbital-parameter derivatives at scalar time.

    Scalar counterpart of :func:`_star_planet_distance_ovd`. Returns
    :math:`r = \\sqrt{x^2 + y^2 + z^2}` and
    :math:`\\partial r/\\partial \\theta_k = (x\\,\\partial x/\\partial \\theta_k
    + y\\,\\partial y/\\partial \\theta_k + z\\,\\partial z/\\partial \\theta_k)/r`.

    Parameters
    ----------
    t : float
        Time at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    r : float
        3D star-planet distance [stellar radii].
    dr : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    r = sqrt(x * x + y * y + z * z)
    inv_r = 1.0 / r
    dr = zeros(6)
    for k in range(6):
        dr[k] = (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r
    return r, dr


@njit(fastmath=True)
def _star_planet_distance_ovd(times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """3D star-planet distance and orbital-parameter derivatives at array of times.

    Returns :math:`r = \\sqrt{x^2 + y^2 + z^2}` and
    :math:`\\partial r/\\partial \\theta_k = (x\\,\\partial x/\\partial \\theta_k
    + y\\,\\partial y/\\partial \\theta_k + z\\,\\partial z/\\partial \\theta_k)/r`.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the separation and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    rs : ndarray, shape (N,)
        3D star-planet separations per time.
    drs : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    rs = zeros(n)
    drs = zeros((n, 6))
    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        r = sqrt(x * x + y * y + z * z)
        rs[j] = r
        inv_r = 1.0 / r
        for k in range(6):
            drs[j, k] = (x * dx[k] + y * dy[k] + z * dz[k]) * inv_r
    return rs, drs


@njit(fastmath=True)
def _cos_v_p_angle_osd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the angle between planet position and a fixed reference vector at scalar time.

    Scalar counterpart of :func:`_cos_v_p_angle_ovd`. The reference vector
    ``v`` is treated as a constant; gradients are w.r.t. the 6 orbital
    parameters only.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Fixed reference vector.
    t : float
        Time at which to evaluate the cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    cs : float
        Cosine of the angle.
    dcs : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    r2 = x * x + y * y + z * z
    r = sqrt(r2)
    inv_r = 1.0 / r
    inv_r3 = inv_r / r2
    dot = x * v[0] + y * v[1] + z * v[2]
    cs = dot * inv_nv * inv_r
    dcs = zeros(6)
    for k in range(6):
        ddot = dx[k] * v[0] + dy[k] * v[1] + dz[k] * v[2]
        xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
        dcs[k] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3)
    return cs, dcs


@njit(fastmath=True)
def _cos_v_p_angle_ovd(v, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of the angle between planet position and a fixed reference vector ``v``.

    The reference vector ``v`` is treated as a constant; gradients are
    computed w.r.t. the 6 orbital parameters only.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Fixed reference vector. Need not be unit-norm; the cosine is
        normalised internally.
    times : ndarray, shape (N,)
        Times at which to evaluate the cosine and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    cs : ndarray, shape (N,)
        Cosine values per time.
    dcs : ndarray, shape (N, 6)
        Gradients w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    cs = zeros(n)
    dcs = zeros((n, 6))
    inv_nv = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        inv_r = 1.0 / r
        inv_r3 = inv_r / r2
        dot = x * v[0] + y * v[1] + z * v[2]
        cs[j] = dot * inv_nv * inv_r
        # d/dθ[(x·v)/(|x|·|v|)] = ((dx·v)/|x| - (x·v)·(x·dx)/|x|^3) / |v|
        for k in range(6):
            ddot = dx[k] * v[0] + dy[k] * v[1] + dz[k] * v[2]
            xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
            dcs[j, k] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3)
    return cs, dcs


# ---------------------------------------------------------------------------
# True anomaly
# ---------------------------------------------------------------------------
#
# The geometric definition uses the angle between the planet position vector
# and the eccentricity vector. Differentiating that (with the prograde sign
# correction from r·v) gives a well-defined gradient everywhere except at
# the two singular configurations ``edp = ±1`` (planet on the apsidal line).
# At those points the analytic derivative diverges; we set it to zero so
# downstream gradient-based fits don't get a NaN. The circular fast path
# (``ex ≤ -0.9999`` sentinel from ``eccentricity_vector``) collapses true
# anomaly to mean anomaly: ``f = 2π(t - tpa)/p`` ⇒ analytic derivatives are
# trivial in (phase, p) and zero in the rest.

@njit
def _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """True anomaly and orbital-parameter derivatives at scalar time.

    Scalar counterpart of :func:`_true_anomaly_ovd`. See that function for
    the geometric derivation, the singular-configuration policy
    (``df = 0`` at ``edp = +/-1``), and the circular-orbit fast path.

    Parameters
    ----------
    t : float
        Time at which to evaluate the true anomaly and gradient.
    tpa : float
        Periastron time.
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector. ``(-1, 0, 0)`` triggers the
        circular-orbit fast path.
    w : float
        Argument of periastron [radians]. Kept for signature parity.
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    f : float
        True anomaly [radians], in :math:`[0, 2\\pi)`.
    df : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    df = zeros(6)
    nes = ex * ex + ey * ey + ez * ez

    if ex <= -0.9999 and nes > 0.99:
        twopi = 2.0 * pi
        tau = t - tpa
        epoch = floor(tau / p)
        tau_red = tau - epoch * p
        f = twopi * tau_red / p
        df[0] = -twopi / p
        df[1] = -twopi * tau_red / (p * p)
        return f, df

    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    tcc = tc - points[ix] * p
    c = coeffs[ix]
    dc = dcoeffs[ix]

    x, y, z, dx, dy, dz = pos_cd(tcc, c, dc)
    vx, vy, vz, dvx, dvy, dvz = vel_cd(tcc, c, dc)

    r2 = x * x + y * y + z * z
    sqrt_r2_nes = sqrt(r2 * nes)
    edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes
    rdotv = x * vx + y * vy + z * vz

    if edp <= -1.0:
        return pi, df
    if edp >= 1.0:
        return 0.0, df

    sign = 1.0 if rdotv > 0.0 else -1.0
    base = arccos(edp)
    f = base if sign > 0.0 else 2.0 * pi - base
    denom = sqrt(1.0 - edp * edp)
    xdote = x * ex + y * ey + z * ez
    for k in range(6):
        dxdote = dx[k] * ex + dy[k] * ey + dz[k] * ez
        xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
        dedp = dxdote / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
        df_k = -dedp / denom
        df[k] = df_k if sign > 0.0 else -df_k
    return f, df


@njit
def _true_anomaly_ovd(times, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """True anomaly and its orbital-parameter derivatives at array of times.

    Computed from the geometric angle between the planet position vector
    and the eccentricity vector :math:`(e_x, e_y, e_z)`, with
    :math:`r \\cdot v` resolving the two branches of :math:`\\arccos`.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the true anomaly and gradient.
    tpa : float
        Periastron time anchoring the knot grid (see :func:`_pos_osd`).
    p : float
        Orbital period [days].
    ex, ey, ez : float
        Components of the eccentricity vector. ``(-1, 0, 0)`` is the
        sentinel produced by
        :func:`~meepmeep.backends.numba.utils.eccentricity_vector` for
        near-circular orbits and triggers the closed-form fast path.
    w : float
        Argument of periastron [radians]. Kept for signature parity with
        the base function; currently unused inside this routine because
        the eccentricity vector is passed explicitly.
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays from :func:`solve3d_orbit_d` /
        :func:`~meepmeep.backends.numba.knots.create_knots`.

    Returns
    -------
    f : ndarray, shape (N,)
        True anomaly per time [radians], in :math:`[0, 2\\pi)`.
    df : ndarray, shape (N, 6)
        Gradient w.r.t. ``(phase, p, a, i, e, w)`` per time. The
        ``ex, ey, ez, w`` inputs are treated as known constants — they
        are functions of the orbital parameters but the dependency is
        captured implicitly through the geometric chain rule on the
        position vector.

    Notes
    -----
    At the singular configurations ``edp = ±1`` (``edp`` = cosine of the
    angle between position and eccentricity vector) the analytic gradient
    diverges and is replaced by zero. The circular-orbit fast path uses
    the mean-anomaly identity :math:`f = 2\\pi(t - t_\\mathrm{pa}) / p`.
    """
    n = times.size
    f = zeros(n)
    df = zeros((n, 6))
    nes = ex * ex + ey * ey + ez * ez

    # Circular-orbit fast path: f = 2π·(t - tpa) / p.
    # df/d(phase) = -2π/p (since phase parameter shifts tpa by 1 unit of phase
    # which is equivalent to a +1-day shift here — solve3d_d's "phase" is in
    # days, so dphase = +1 ⇒ dtpa = +1 ⇒ df = -2π/p).
    # df/dp = -2π·(t - tpa) / p^2.
    if ex <= -0.9999 and nes > 0.99:
        twopi = 2.0 * pi
        for j in range(n):
            tau = times[j] - tpa
            # Reduce to one period for the value (mean_anomaly does this in base).
            epoch = floor(tau / p)
            tau_red = tau - epoch * p
            f[j] = twopi * tau_red / p
            df[j, 0] = -twopi / p
            df[j, 1] = -twopi * tau_red / (p * p)
        return f, df

    for j in range(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p
        c = coeffs[ix]
        dc = dcoeffs[ix]

        x, y, z, dx, dy, dz = pos_cd(tcc, c, dc)
        vx, vy, vz, dvx, dvy, dvz = vel_cd(tcc, c, dc)

        r2 = x * x + y * y + z * z
        r = sqrt(r2)
        sqrt_r2_nes = sqrt(r2 * nes)
        edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes
        rdotv = x * vx + y * vy + z * vz

        if edp <= -1.0:
            f[j] = pi
            # Singular: leave df[j] = 0.
        elif edp >= 1.0:
            f[j] = 0.0
            # Singular: leave df[j] = 0.
        else:
            sign = 1.0 if rdotv > 0.0 else -1.0
            base = arccos(edp)
            f[j] = base if sign > 0.0 else 2.0 * pi - base
            # d(arccos(edp))/dθ = -dedp/sqrt(1 - edp^2)
            denom = sqrt(1.0 - edp * edp)
            inv_r2 = 1.0 / r2
            for k in range(6):
                # edp = (x·e)/(r·|e|). Treat |e| (and ex,ey,ez) as constants
                # for this routine — they're inputs. d(edp)/dθ_k
                # = (dx·e)/(r·|e|) - (x·e)·(x·dx)/(r^3·|e|)
                xdote = x * ex + y * ey + z * ez
                dxdote = dx[k] * ex + dy[k] * ey + dz[k] * ez
                xdotdx = x * dx[k] + y * dy[k] + z * dz[k]
                dedp = dxdote / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes)
                # Equivalent: dedp = (dxdote * r2 - xdote * xdotdx * inv_r2 * r2) ... keep clarity.
                df_k = -dedp / denom
                df[j, k] = df_k if sign > 0.0 else -df_k
    return f, df


# ---------------------------------------------------------------------------
# Photometric / RV signals
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _lambert_kernel_d(cos_alpha):
    """Lambertian phase function, alpha, and ``dphase/dcos_alpha``.

    The analytic derivative of
    :math:`\\mathrm{phase}(c) = (\\sqrt{1-c^2} + (\\pi - \\arccos c)\\,c)/\\pi`
    simplifies to :math:`(\\pi - \\arccos c)/\\pi` because the contributions
    from :math:`d/dc \\sqrt{1-c^2}` and :math:`c \\cdot d/dc \\arccos c`
    cancel exactly.

    Parameters
    ----------
    cos_alpha : float
        Cosine of the phase angle. Clamped internally to ``[-1, 1]``.

    Returns
    -------
    phase : float
        Value of the Lambert kernel, in :math:`[0, 1]`.
    alpha : float
        Phase angle :math:`\\arccos(\\mathrm{cos\\_alpha})` [radians].
    dphase_dc : float
        Derivative :math:`d\\,\\mathrm{phase}/d\\,\\mathrm{cos\\_alpha}`.
    """
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
    alpha = arccos(cos_alpha)
    phase = (sin_alpha + (pi - alpha) * cos_alpha) / pi
    dphase_dc = (pi - alpha) / pi
    return phase, alpha, dphase_dc


@njit(fastmath=True)
def _lambert_phase_curve_osd(time, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux and parameter derivatives at scalar time.

    Derivative ordering: ``(phase, p, a, i, e, w, ag, k)`` — length 8.

    Parameters
    ----------
    time : float
        Time at which to evaluate the flux contribution and gradient.
    ag : float
        Geometric albedo.
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    flux : float
        Reflected planet-to-star flux ratio.
    dflux : ndarray, shape (8,)
        Gradient w.r.t. ``(phase, p, a, i, e, w, ag, k)``.
    """
    amplitude = k * k * ag / (a * a)
    ca, dca = _cos_alpha_osd(time, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    phase, _, dphase_dc = _lambert_kernel_d(ca)
    flux = amplitude * phase

    dflux = zeros(8)
    # Orbital block — chain through cos_alpha and through amplitude (only `a` matters).
    for kk in range(6):
        dflux[kk] = amplitude * dphase_dc * dca[kk]
    # Add d(amplitude)/da contribution to the `a` slot (index 2):
    # damplitude/da = -2 k^2 ag / a^3.
    dflux[2] += -2.0 * k * k * ag / (a * a * a) * phase
    # Extras: ag (index 6), k (index 7).
    dflux[6] = (k * k / (a * a)) * phase
    dflux[7] = (2.0 * k * ag / (a * a)) * phase
    return flux, dflux


@njit(fastmath=True)
def _lambert_phase_curve_ovd(times, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux and parameter derivatives at array of times.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the flux contribution and gradient.
    ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_lambert_phase_curve_osd`.

    Returns
    -------
    flux : ndarray, shape (N,)
        Reflected planet-to-star flux ratio per time.
    dflux : ndarray, shape (N, 8)
        Gradient w.r.t. ``(phase, p, a, i, e, w, ag, k)`` per time.
    """
    n = times.size
    flux = zeros(n)
    dflux = zeros((n, 8))
    inv_a2 = 1.0 / (a * a)
    amplitude = k * k * ag * inv_a2
    da_amp = -2.0 * k * k * ag / (a * a * a)
    dag_amp = k * k * inv_a2
    dk_amp = 2.0 * k * ag * inv_a2
    for j in range(n):
        ca, dca = _cos_alpha_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        phase, _, dphase_dc = _lambert_kernel_d(ca)
        flux[j] = amplitude * phase
        for kk in range(6):
            dflux[j, kk] = amplitude * dphase_dc * dca[kk]
        dflux[j, 2] += da_amp * phase
        dflux[j, 6] = dag_amp * phase
        dflux[j, 7] = dk_amp * phase
    return flux, dflux


@njit(fastmath=True)
def _lambert_and_emission_osd(t, ag, fr_night, fr_day, emi_offset, a, k,
                              tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus cosine-emission day/night model with derivatives at scalar time.

    Scalar counterpart of :func:`_lambert_and_emission_ovd`. Derivative
    ordering: ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``
    — length 11.

    Parameters
    ----------
    t : float
        Time at which to evaluate the flux contributions and gradients.
    ag, fr_night, fr_day, emi_offset, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_lambert_and_emission_ovd`.

    Returns
    -------
    ref : float
        Reflected (Lambertian) flux contribution.
    emi : float
        Thermal emission contribution.
    dref : ndarray, shape (11,)
        Gradient of ``ref`` w.r.t.
        ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``.
    demi : ndarray, shape (11,)
        Gradient of ``emi`` w.r.t. the same parameter block.
    """
    k2 = k * k
    inv_a2 = 1.0 / (a * a)
    aref = k2 * ag * inv_a2
    daref_da = -2.0 * k2 * ag / (a * a * a)
    daref_dag = k2 * inv_a2
    daref_dk = 2.0 * k * ag * inv_a2

    ca, dca = _cos_alpha_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    phase, alpha, dphase_dc = _lambert_kernel_d(ca)

    dref = zeros(11)
    demi = zeros(11)

    ref = aref * phase
    for kk in range(6):
        dref[kk] = aref * dphase_dc * dca[kk]
    dref[2] += daref_da * phase
    dref[6] = daref_dag * phase
    dref[10] = daref_dk * phase

    cs = cos(alpha + emi_offset)
    sn = sin(alpha + emi_offset)
    bracket = fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cs)
    emi = k2 * bracket

    ca_clamped = ca
    if ca_clamped > 1.0:
        ca_clamped = 1.0
    elif ca_clamped < -1.0:
        ca_clamped = -1.0
    s = sqrt(1.0 - ca_clamped * ca_clamped)
    if s < 1e-12:
        dalpha_dc = 0.0
    else:
        dalpha_dc = -1.0 / s
    demi_dalpha = k2 * (fr_day - fr_night) * 0.5 * sn
    for kk in range(6):
        demi[kk] = demi_dalpha * dalpha_dc * dca[kk]
    demi[7] = k2 * (1.0 - 0.5 * (1.0 - cs))
    demi[8] = k2 * 0.5 * (1.0 - cs)
    demi[9] = k2 * (fr_day - fr_night) * 0.5 * sn
    demi[10] = 2.0 * k * bracket

    return ref, emi, dref, demi


@njit(fastmath=True)
def _lambert_and_emission_ovd(times, ag, fr_night, fr_day, emi_offset, a, k,
                             tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus cosine-emission day/night model with parameter derivatives.

    Derivative ordering: ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``
    — length 11.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the flux contributions and gradients.
    ag : float
        Geometric albedo (reflected component).
    fr_night : float
        Night-side flux ratio (planet/star).
    fr_day : float
        Day-side flux ratio (planet/star).
    emi_offset : float
        Phase-angle offset of the emission peak [radians].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    k : float
        Planet-to-star radius ratio :math:`R_p/R_\\star`.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    ref : ndarray, shape (N,)
        Reflected (Lambertian) flux contribution per time.
    emi : ndarray, shape (N,)
        Thermal emission contribution per time.
    dref : ndarray, shape (N, 11)
        Gradient of ``ref`` w.r.t.
        ``(phase, p, a, i, e, w, ag, fr_night, fr_day, emi_offset, k)``.
    demi : ndarray, shape (N, 11)
        Gradient of ``emi`` w.r.t. the same parameter block.
    """
    n = times.size
    ref = zeros(n)
    emi = zeros(n)
    dref = zeros((n, 11))
    demi = zeros((n, 11))
    k2 = k * k
    inv_a2 = 1.0 / (a * a)
    aref = k2 * ag * inv_a2
    daref_da = -2.0 * k2 * ag / (a * a * a)
    daref_dag = k2 * inv_a2
    daref_dk = 2.0 * k * ag * inv_a2

    for j in range(n):
        ca, dca = _cos_alpha_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        phase, alpha, dphase_dc = _lambert_kernel_d(ca)

        # --- reflected component ---
        ref[j] = aref * phase
        for kk in range(6):
            dref[j, kk] = aref * dphase_dc * dca[kk]
        dref[j, 2] += daref_da * phase
        dref[j, 6] = daref_dag * phase
        # fr_night, fr_day, emi_offset (indices 7..9) are zero for ref.
        dref[j, 10] = daref_dk * phase

        # --- emission component ---
        # emi = k^2 · (fr_night + (fr_day - fr_night) · 0.5 · (1 - cos(alpha + emi_offset)))
        cs = cos(alpha + emi_offset)
        sn = sin(alpha + emi_offset)
        bracket = fr_night + (fr_day - fr_night) * 0.5 * (1.0 - cs)
        emi[j] = k2 * bracket

        # d(alpha)/d(cos_alpha) = -1/sqrt(1 - ca^2). Avoid blow-up at |ca|=1
        # by clamping like _lambert_kernel_d does (interior tests safe).
        ca_clamped = ca
        if ca_clamped > 1.0:
            ca_clamped = 1.0
        elif ca_clamped < -1.0:
            ca_clamped = -1.0
        s = sqrt(1.0 - ca_clamped * ca_clamped)
        if s < 1e-12:
            dalpha_dc = 0.0
        else:
            dalpha_dc = -1.0 / s
        # demi/dorbital via cos_alpha → alpha → bracket
        # demi/dα = k^2 · (fr_day - fr_night) · 0.5 · sin(alpha + emi_offset)
        demi_dalpha = k2 * (fr_day - fr_night) * 0.5 * sn
        for kk in range(6):
            demi[j, kk] = demi_dalpha * dalpha_dc * dca[kk]
        # ag (6) does not enter emi; leave 0.
        # fr_night (7): k^2 · (1 - 0.5·(1-cs)) = k^2 · (0.5 + 0.5·cs)
        demi[j, 7] = k2 * (1.0 - 0.5 * (1.0 - cs))
        # fr_day (8):   k^2 · 0.5 · (1 - cs)
        demi[j, 8] = k2 * 0.5 * (1.0 - cs)
        # emi_offset (9): k^2 · (fr_day - fr_night) · 0.5 · sin(alpha + emi_offset)
        demi[j, 9] = k2 * (fr_day - fr_night) * 0.5 * sn
        # k (10): 2k · bracket
        demi[j, 10] = 2.0 * k * bracket

    return ref, emi, dref, demi


@njit(fastmath=True)
def _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal and derivatives at scalar time.

    Scalar counterpart of :func:`_ev_signal_ovd`. Derivative ordering:
    ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` — length 9.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient.
    mass_ratio : float
        Planet-to-star mass ratio.
    inc : float
        Orbital inclination [radians], independent of the orbital ``i`` axis.
    t : float
        Time at which to evaluate the signal and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    out : float
        Ellipsoidal variation signal.
    dout : ndarray, shape (9,)
        Gradient w.r.t. ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)``.
    """
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    x, y, z, dx, dy, dz = _pos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    d2 = x * x + y * y + z * z
    d = sqrt(d2)
    cz = z / d
    g = (2.0 * cz * cz - 1.0) / (d2 * d)
    out = pre * g

    d5 = d2 * d2 * d
    A = 2.0 * z * z - d2
    dout = zeros(9)
    for kk in range(6):
        xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
        dd = xdotdx / d
        dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
        dg = (dA - 5.0 * A * dd / d2) / d5
        dout[kk] = pre * dg
    dout[6] = -mass_ratio * sin2_inc * g
    dout[7] = -alpha * sin2_inc * g
    dout[8] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g
    return out, dout


@njit(fastmath=True)
def _ev_signal_ovd(alpha, mass_ratio, inc, times, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal and parameter derivatives.

    Implements
    :math:`S = -\\alpha\\,q\\,\\sin^2 i\\,(2 c_z^2 - 1) / d^3`
    where :math:`c_z = z/d` and :math:`d = \\sqrt{x^2 + y^2 + z^2}`. The
    function-local ``inc`` parameter is independent of the orbital
    inclination ``i`` — callers that share them should sum the two
    derivative slots.

    Derivative ordering: ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` —
    length 9.

    Parameters
    ----------
    alpha : float
        Gravity-darkening coefficient (Lillo-Box et al. 2014, Eq. 7).
    mass_ratio : float
        Planet-to-star mass ratio :math:`M_p / M_\\star`.
    inc : float
        Orbital inclination [radians]. Treated as a function-local input
        independent of the orbital ``i`` axis of the gradient.
    times : ndarray, shape (N,)
        Times at which to evaluate the signal and gradient.
    tpa, p, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_pos_osd`.

    Returns
    -------
    out : ndarray, shape (N,)
        Ellipsoidal variation signal per time.
    dout : ndarray, shape (N, 9)
        Gradient w.r.t.
        ``(phase, p, a, i, e, w, alpha, mass_ratio, inc)`` per time.
    """
    n = times.size
    out = zeros(n)
    dout = zeros((n, 9))
    sin_inc = sin(inc)
    cos_inc = cos(inc)
    sin2_inc = sin_inc * sin_inc
    pre = -alpha * mass_ratio * sin2_inc

    for j in range(n):
        x, y, z, dx, dy, dz = _pos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        d2 = x * x + y * y + z * z
        d = sqrt(d2)
        cz = z / d
        # S = pre · g, where g = (2 cz^2 - 1) / d^3.
        # Rewrite g = (2 z^2 - d^2) / d^5 = (2 z^2 / d^5) - 1/d^3.
        g = (2.0 * cz * cz - 1.0) / (d2 * d)
        out[j] = pre * g

        # dg/dθ via dx, dy, dz. Use g = (2 z^2 - d^2) / d^5.
        # Let A = 2 z^2 - d^2,  d^5 = d2^2 · d.
        # dA = 4 z·dz - 2(x·dx + y·dy + z·dz)
        #    = -2(x·dx + y·dy) + 2 z·dz
        # d(d^5)/dθ = 5 d^3 · dd, with dd = (x·dx + y·dy + z·dz)/d.
        # dg = (dA · d^5 - A · 5 d^3 · dd) / d^10
        #    = (dA - 5 A · dd / d^2) / d^5.
        d5 = d2 * d2 * d
        A = 2.0 * z * z - d2
        for kk in range(6):
            xdotdx = x * dx[kk] + y * dy[kk] + z * dz[kk]
            dd = xdotdx / d
            dA = -2.0 * (x * dx[kk] + y * dy[kk]) + 2.0 * z * dz[kk]
            dg = (dA - 5.0 * A * dd / d2) / d5
            dout[j, kk] = pre * dg
        # Extras (no orbital chain).
        # alpha (6): dS/dalpha = -mass_ratio · sin2_inc · g
        dout[j, 6] = -mass_ratio * sin2_inc * g
        # mass_ratio (7): dS/dmr = -alpha · sin2_inc · g
        dout[j, 7] = -alpha * sin2_inc * g
        # inc (8): d(sin^2 inc)/dinc = 2 sin_inc · cos_inc
        dout[j, 8] = -alpha * mass_ratio * 2.0 * sin_inc * cos_inc * g

    return out, dout


# ---------------------------------------------------------------------------
# Light travel time
# ---------------------------------------------------------------------------

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
    dz_tr_total : ndarray, shape (6,)
        Full total derivative of ``z(t_transit)`` w.r.t.
        ``(phase, p, a, i, e, w)``.

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
    dto = zeros(6)
    dto[1] = m_tr / two_pi
    dto[4] = dm_tr_de * p / two_pi
    dto[5] = dm_tr_dw * p / two_pi

    dz_tr_total = zeros(6)
    for k in range(6):
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
    """Light travel time correction and orbital-parameter derivatives at scalar time.

    The correction is referenced to primary transit:

    .. math::

        \\mathrm{ltt}(t) = -(z(t) - z(t_\\mathrm{transit}))\\,r_\\star\\,(R_\\odot / c)

    where :math:`t_\\mathrm{transit} = t_\\mathrm{pa} + M_\\mathrm{tr}(e, w)\\,p/(2\\pi)`.

    Per spec, the partial derivative w.r.t. ``rstar`` is *not* returned —
    only the 6 orbital derivatives in the canonical
    ``(phase, p, a, i, e, w)`` order. The reference ``z(t_transit)`` and
    its parameter derivatives are computed by :func:`_ltt_transit_z_and_d`,
    which includes the chain rule through ``t_transit(p, e, w)`` using
    ``vz(t_transit)``.

    Parameters
    ----------
    t : float
        Time at which to evaluate the correction and gradient.
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
    ltt : float
        Light travel time correction [days].
    dltt : ndarray, shape (6,)
        Gradient w.r.t. ``(phase, p, a, i, e, w)``.
    """
    z_t, dz_t = _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    z_tr, dz_tr = _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    factor = -rstar * LTT_DAYS_PER_RSUN
    ltt = factor * (z_t - z_tr)
    dltt = zeros(6)
    for k in range(6):
        dltt[k] = factor * (dz_t[k] - dz_tr[k])
    return ltt, dltt


@njit(fastmath=True)
def _light_travel_time_ovd(times, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Light travel time correction and orbital-parameter derivatives at array of times.

    Vectorised version of :func:`_light_travel_time_osd`. Caches the
    transit-time reference (``z_tr`` and its gradient) once outside the
    loop and reuses it for every input time.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the correction and gradient.
    tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_light_travel_time_osd`.

    Returns
    -------
    ltt : ndarray, shape (N,)
        Light travel time corrections [days].
    dltt : ndarray, shape (N, 6)
        Gradient w.r.t. ``(phase, p, a, i, e, w)`` per time.
    """
    n = times.size
    ltt = zeros(n)
    dltt = zeros((n, 6))
    factor = -rstar * LTT_DAYS_PER_RSUN
    # Reference (z and its full derivative chain) computed once.
    z_tr, dz_tr = _ltt_transit_z_and_d(tpa, p, e, w, dt, pktable, points, coeffs, dcoeffs)
    for j in range(n):
        z, dz = _zpos_osd(times[j], tpa, p, dt, pktable, points, coeffs, dcoeffs)
        ltt[j] = factor * (z - z_tr)
        for k in range(6):
            dltt[j, k] = factor * (dz[k] - dz_tr[k])
    return ltt, dltt


@njit(fastmath=True)
def _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Radial velocity and parameter derivatives at scalar time.

    Scalar counterpart of :func:`_rv_ovd`. Derivative ordering:
    ``(phase, p, a, i, e, w, k)`` — length 7.

    Parameters
    ----------
    t : float
        Time at which to evaluate the radial velocity and gradient.
    k : float
        Radial-velocity semi-amplitude [m s\\ :sup:`-1`].
    tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs :
        See :func:`_rv_ovd`.

    Returns
    -------
    rv : float
        Radial velocity [m s\\ :sup:`-1`].
    drv : ndarray, shape (7,)
        Gradient w.r.t. ``(phase, p, a, i, e, w, k)``.
    """
    epoch = floor((t - tpa) / p)
    tc = t - tpa - epoch * p
    ix = pktable[int(floor(tc / (dt * p)))]
    tcc = tc - points[ix] * p
    rv_val, drv_orb = rv_cd(tcc, k, p, a, i, e, coeffs[ix], dcoeffs[ix])
    drv = zeros(7)
    for kk in range(6):
        drv[kk] = drv_orb[kk]
    drv[6] = rv_val / k if k != 0.0 else 0.0
    return rv_val, drv


@njit(fastmath=True)
def _rv_ovd(times, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Radial velocity and parameter derivatives at array of times.

    Derivative ordering: ``(phase, p, a, i, e, w, k)`` — length 7.

    Parameters
    ----------
    times : ndarray, shape (N,)
        Times at which to evaluate the radial velocity and gradient.
    k : float
        Radial-velocity semi-amplitude [m s\\ :sup:`-1`].
    tpa : float
        Periastron time anchoring the knot grid (see :func:`_pos_osd`).
    p : float
        Orbital period [days].
    a : float
        Scaled semi-major axis :math:`a/R_\\star`.
    i : float
        Inclination [radians].
    e : float
        Eccentricity.
    dt, pktable, points, coeffs, dcoeffs :
        Multi-knot dispatch arrays.

    Returns
    -------
    rvs : ndarray, shape (N,)
        Radial velocities per time [m s\\ :sup:`-1`].
    drvs : ndarray, shape (N, 7)
        Gradients w.r.t. ``(phase, p, a, i, e, w, k)`` per time.
    """
    n = times.size
    rvs = zeros(n)
    drvs = zeros((n, 7))
    for j in range(n):
        t = times[j]
        epoch = floor((t - tpa) / p)
        tc = t - tpa - epoch * p
        ix = pktable[int(floor(tc / (dt * p)))]
        tcc = tc - points[ix] * p
        rv_val, drv_orb = rv_cd(tcc, k, p, a, i, e, coeffs[ix], dcoeffs[ix])
        rvs[j] = rv_val
        for kk in range(6):
            drvs[j, kk] = drv_orb[kk]
        # drv/dk = rv / k  (rv is linear in k via the scale factor s = k/n).
        drvs[j, 6] = rv_val / k if k != 0.0 else 0.0
    return rvs, drvs


# ---------------------------------------------------------------------------
# Unified scalar/array dispatchers (gradient-returning)
# ---------------------------------------------------------------------------
# Each ``<name>_od`` mirrors the orbit3d.py ``<name>_o`` pattern for the
# gradient-returning families. Scalar inputs route to ``_<name>_osd``;
# 1-D float64 array inputs route to ``_<name>_ovd``. Same time-argument
# conventions as orbit3d (most use arg 0; ``cos_v_p_angle_od`` uses arg 1;
# ``ev_signal_od`` uses arg 3).


def _is_1d_array(typ):
    """True for a 1-D Numba array type (any layout)."""
    return isinstance(typ, types.Array) and typ.ndim == 1


# --- pos -------------------------------------------------------------------

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


# --- zpos ------------------------------------------------------------------

def zpos_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-position with gradients. See :func:`_zpos_osd` / :func:`_zpos_ovd`."""
    if isinstance(t, ndarray):
        return _zpos_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(zpos_od, jit_options={'fastmath': True})
def _zpos_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zpos_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zpos_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- sep -------------------------------------------------------------------

def sep_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Sky-projected separation with gradients. See :func:`_sep_osd` / :func:`_sep_ovd`."""
    if isinstance(t, ndarray):
        return _sep_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(sep_od, jit_options={'fastmath': True})
def _sep_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _sep_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _sep_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- vel -------------------------------------------------------------------

def vel_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet velocity with gradients. See :func:`_vel_osd` / :func:`_vel_ovd`."""
    if isinstance(t, ndarray):
        return _vel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(vel_od, jit_options={'fastmath': True})
def _vel_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _vel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _vel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- zvel ------------------------------------------------------------------

def zvel_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Planet z-velocity with gradients. See :func:`_zvel_osd` / :func:`_zvel_ovd`."""
    if isinstance(t, ndarray):
        return _zvel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(zvel_od, jit_options={'fastmath': True})
def _zvel_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zvel_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _zvel_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- cos_alpha -------------------------------------------------------------

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


# --- star_planet_distance --------------------------------------------------

def star_planet_distance_od(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """3D star-planet distance with gradients.

    See :func:`_star_planet_distance_osd` / :func:`_star_planet_distance_ovd`.
    """
    if isinstance(t, ndarray):
        return _star_planet_distance_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _star_planet_distance_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(star_planet_distance_od, jit_options={'fastmath': True})
def _star_planet_distance_od_overload(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _star_planet_distance_ovd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _star_planet_distance_osd(t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- cos_v_p_angle ---------------------------------------------------------

def cos_v_p_angle_od(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Cosine of angle between planet position and fixed vector, with gradients.

    See :func:`_cos_v_p_angle_osd` / :func:`_cos_v_p_angle_ovd`.
    """
    if isinstance(t, ndarray):
        return _cos_v_p_angle_ovd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _cos_v_p_angle_osd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(cos_v_p_angle_od, jit_options={'fastmath': True})
def _cos_v_p_angle_od_overload(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _cos_v_p_angle_ovd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _cos_v_p_angle_osd(v, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- true_anomaly ----------------------------------------------------------

def true_anomaly_od(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    """True anomaly with gradients. See :func:`_true_anomaly_osd` / :func:`_true_anomaly_ovd`."""
    if isinstance(t, ndarray):
        return _true_anomaly_ovd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)
    return _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)


@overload(true_anomaly_od)
def _true_anomaly_od_overload(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
            return _true_anomaly_ovd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs):
            return _true_anomaly_osd(t, tpa, p, ex, ey, ez, w, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- lambert_phase_curve ---------------------------------------------------

def lambert_phase_curve_od(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian phase-curve flux with gradients.

    See :func:`_lambert_phase_curve_osd` / :func:`_lambert_phase_curve_ovd`.
    """
    if isinstance(t, ndarray):
        return _lambert_phase_curve_ovd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _lambert_phase_curve_osd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(lambert_phase_curve_od, jit_options={'fastmath': True})
def _lambert_phase_curve_od_overload(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_phase_curve_ovd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_phase_curve_osd(t, ag, a, k, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- lambert_and_emission --------------------------------------------------

def lambert_and_emission_od(t, ag, fr_night, fr_day, emi_offset, a, k,
                            tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Lambertian reflection plus emission with gradients.

    See :func:`_lambert_and_emission_osd` / :func:`_lambert_and_emission_ovd`.
    """
    if isinstance(t, ndarray):
        return _lambert_and_emission_ovd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                         tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _lambert_and_emission_osd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                     tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(lambert_and_emission_od, jit_options={'fastmath': True})
def _lambert_and_emission_od_overload(t, ag, fr_night, fr_day, emi_offset, a, k,
                                      tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, ag, fr_night, fr_day, emi_offset, a, k,
                 tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_and_emission_ovd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                             tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, ag, fr_night, fr_day, emi_offset, a, k,
                 tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _lambert_and_emission_osd(t, ag, fr_night, fr_day, emi_offset, a, k,
                                             tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- ev_signal -------------------------------------------------------------

def ev_signal_od(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    """Ellipsoidal variation signal with gradients.

    Time argument is the 4th positional. See :func:`_ev_signal_osd` /
    :func:`_ev_signal_ovd`.
    """
    if isinstance(t, ndarray):
        return _ev_signal_ovd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
    return _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)


@overload(ev_signal_od, jit_options={'fastmath': True})
def _ev_signal_od_overload(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _ev_signal_ovd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs):
            return _ev_signal_osd(alpha, mass_ratio, inc, t, tpa, p, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- rv --------------------------------------------------------------------

def rv_od(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    """Radial velocity with gradients. See :func:`_rv_osd` / :func:`_rv_ovd`."""
    if isinstance(t, ndarray):
        return _rv_ovd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)
    return _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)


@overload(rv_od, jit_options={'fastmath': True})
def _rv_od_overload(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
    if _is_1d_array(t):
        def impl(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
            return _rv_ovd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)
        return impl
    if isinstance(t, types.Float):
        def impl(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs):
            return _rv_osd(t, k, tpa, p, a, i, e, dt, pktable, points, coeffs, dcoeffs)
        return impl
    return None


# --- light_travel_time -----------------------------------------------------

def light_travel_time_od(t, tpa, p, e, w, rstar, dt, pktable, points, coeffs, dcoeffs):
    """Light travel time correction with gradients.

    See :func:`_light_travel_time_osd` / :func:`_light_travel_time_ovd`.
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
