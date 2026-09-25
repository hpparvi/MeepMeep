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

"""Multi-expansion-point, orbit-spanning evaluators.

JAX ports of the ``meepmeep.backends.numba.orbit3d`` package. Each
``X_o`` function takes the same arguments in the same order as its numba
twin: the time(s), any physical inputs, the periastron anchor ``tpa``, the
period, and the grid from
:func:`~meepmeep.jax3d.create_expansion_points`
plus the coefficients from
:func:`~meepmeep.jax3d.solve3d_orbit`. Each time is folded into
one period, dispatched to its expansion point through ``ep_table``, and
evaluated with the single-expansion-point kernels of
``meepmeep.backends.jax.point3d``.

There are no ``_od`` gradient variants. The numba ``_od`` kernels return
gradients in whatever basis ``dcoeffs`` was built in; here the basis is set
by the function being differentiated. Build ``tpa`` from ``tc`` inside it
for the transit-centre basis, or take ``tpa`` directly for the periastron
basis::

    def model(tc, p, a, i, e, w):
        tpa = tc - mean_anomaly_at_transit(e, w) / (2 * pi) * p
        coeffs = solve3d_orbit(ep_times, p, a, i, e, w)
        return sep_o(times, tpa, p, dt, ep_table, ep_times, coeffs)

    d, dd = model(*pars), jnp.stack(jax.jacfwd(model, argnums=range(6))(*pars), -1)

Keep the grid (``ep_times``, ``ep_table``) outside the differentiated
arguments, as numba does, so the expansion points stay at fixed phases.
"""

import jax.numpy as jnp

from ._common import TWO_PI, ep_lookup
from .point3d import (pos_c, zpos_c, sep_c, vel_c, zvel_c, cos_alpha_c, lambert_phase_curve_c,
                      ev_signal_c, emission_phase_curve_c, _rv_scale)
from .newton import LTT_DAYS_PER_RSUN
from .utils import mean_anomaly_at_transit


def ep_ix(t, tpa, p, dt, ep_table):
    """Expansion-point index for time(s) ``t``.

    Parameters
    ----------
    t : float or NDArray
        Time(s) [days].
    tpa : float
        Periastron time anchoring the grid [days].
    p : float
        Orbital period [days].
    dt : float
        Table bin width in fraction of the period.
    ep_table : NDArray of int
        Time-to-expansion-point table.

    Returns
    -------
    ix : int or NDArray of int
        Index into ``coeffs`` / ``ep_times``.
    """
    t = jnp.asarray(t, dtype=float)
    epoch = jnp.floor((t - tpa) / p)
    tf = t - tpa - epoch * p
    return jnp.take(ep_table, jnp.floor(tf / (dt * p)).astype(jnp.int32), mode='clip')


def _local(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Centered time and the coefficient matrix of its expansion point, per time."""
    _, tcc, ix = ep_lookup(t, tpa, p, dt, ep_table, ep_times)
    return tcc, jnp.take(coeffs, ix, axis=0)


def pos_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Planet position ``(x, y, z)`` [R_star] at time(s) ``t``.

    Parameters
    ----------
    t : float or NDArray
        Time(s) [days].
    tpa : float
        Periastron time anchoring the expansion-point grid [days]. Related to
        the transit centre by ``tpa = tc - M_tr(e, w) p / (2 pi)``.
    p : float
        Orbital period [days].
    dt : float
        Table bin width in fraction of the period.
    ep_table : NDArray of int
        Time-to-expansion-point table.
    ep_times : NDArray, shape (npt,)
        Expansion-point phases from periastron.
    coeffs : NDArray, shape (npt, 3, 5)
        Coefficients from :func:`~meepmeep.jax3d.solve3d_orbit`.

    Returns
    -------
    x, y, z : float or NDArray
        Position components, shaped like ``t``.
    """
    return pos_c(*_local(t, tpa, p, dt, ep_table, ep_times, coeffs))


def zpos_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Line-of-sight coordinate z [R_star]. See :func:`pos_o` for the arguments."""
    return zpos_c(*_local(t, tpa, p, dt, ep_table, ep_times, coeffs))


def sep_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Sky-projected separation between the star and planet centers [R_star]. See :func:`pos_o`."""
    return sep_c(*_local(t, tpa, p, dt, ep_table, ep_times, coeffs))


def vel_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Velocity ``(vx, vy, vz)`` [R_star / day]. See :func:`pos_o` for the arguments."""
    return vel_c(*_local(t, tpa, p, dt, ep_table, ep_times, coeffs))


def zvel_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Line-of-sight velocity [R_star / day]. See :func:`pos_o` for the arguments."""
    return zvel_c(*_local(t, tpa, p, dt, ep_table, ep_times, coeffs))


def rv_o(t, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs):
    """Radial velocity with semi-amplitude ``k``. See :func:`pos_o` for the shared arguments."""
    return zvel_o(t, tpa, p, dt, ep_table, ep_times, coeffs) * _rv_scale(k, p, a, i, e)


def cos_alpha_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Cosine of the star-planet-observer phase angle. See :func:`pos_o` for the arguments."""
    return cos_alpha_c(*_local(t, tpa, p, dt, ep_table, ep_times, coeffs))


def star_planet_distance_o(t, tpa, p, dt, ep_table, ep_times, coeffs):
    """3D star-planet distance [R_star]. See :func:`pos_o` for the arguments."""
    x, y, z = pos_o(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return jnp.sqrt(x * x + y * y + z * z)


def cos_v_p_angle_o(v, t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Cosine of the angle between the planet position and a fixed vector ``v`` (shape (3,)).

    See :func:`pos_o` for the shared arguments.
    """
    v = jnp.asarray(v, dtype=float)
    inv_nv = 1.0 / jnp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    x, y, z = pos_o(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return (x * v[0] + y * v[1] + z * v[2]) * inv_nv / jnp.sqrt(x * x + y * y + z * z)


def lambert_phase_curve_o(t, ag, k, tpa, p, dt, ep_table, ep_times, coeffs):
    """Lambert-sphere phase curve; see :func:`~meepmeep.jax3d.lambert_phase_curve_c`."""
    tcc, c = _local(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return lambert_phase_curve_c(tcc, ag, k, c)


def ev_signal_o(alpha, mass_ratio, inc, t, tpa, p, dt, ep_table, ep_times, coeffs):
    """Ellipsoidal-variation signal; see :func:`~meepmeep.jax3d.ev_signal_c`.

    The argument order (physical inputs before the time) follows the numba ``ev_signal_o``.
    """
    tcc, c = _local(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return ev_signal_c(tcc, alpha, mass_ratio, inc, c)


def emission_phase_curve_o(t, k, fratio, offset, tpa, p, dt, ep_table, ep_times, coeffs):
    """Thermal-emission phase curve; see :func:`~meepmeep.jax3d.emission_phase_curve_c`."""
    tcc, c = _local(t, tpa, p, dt, ep_table, ep_times, coeffs)
    return emission_phase_curve_c(tcc, k, fratio, offset, c)


def light_travel_time_o(t, tpa, p, e, w, rstar, dt, ep_table, ep_times, coeffs):
    """Light travel time relative to the transit centre [days].

    Positive when the planet is farther from the observer than at transit,
    i.e. its light arrives later. ``rstar`` is the stellar radius [R_sun];
    the other arguments are as in :func:`pos_o`.
    """
    to = mean_anomaly_at_transit(e, w) / TWO_PI * p
    z_tr = zpos_o(tpa + to, tpa, p, dt, ep_table, ep_times, coeffs)
    return -(zpos_o(t, tpa, p, dt, ep_table, ep_times, coeffs) - z_tr) * rstar * LTT_DAYS_PER_RSUN


def true_anomaly_o(t, tpa, p, ex, ey, ez, w, dt, ep_table, ep_times, coeffs):
    """True anomaly [radians] in ``[0, 2 pi)`` from the Taylor positions.

    Parameters
    ----------
    t : float or NDArray
        Time(s) [days].
    tpa, p : float
        Periastron time and period [days].
    ex, ey, ez : float
        Eccentricity vector, from
        :func:`~meepmeep.jax3d.eccentricity_vector` (pass it the
        same ``lan`` as the solver). The sentinel ``(-1, 0, 0)`` of a circular
        orbit switches to the mean anomaly.
    w : float
        Argument of periastron [radians]. Unused, kept for signature parity
        with numba.
    dt, ep_table, ep_times, coeffs
        As in :func:`pos_o`.

    Notes
    -----
    The true anomaly is the angle between the position and the eccentricity
    vector. The gradient follows wherever the eccentricity vector came from:
    compute it from traced ``(i, e, w, lan)`` for the full derivative (what
    numba's ``true_anomaly_od`` gives with ``dev`` from
    ``eccentricity_vector_d``), or pass it through ``jax.lax.stop_gradient``
    to hold it constant (numba with ``dev = 0``). The
    ``arccos`` is guarded, so exactly aligned positions (``f = 0`` or
    ``pi``) give a zero gradient instead of NaN.
    """
    t = jnp.asarray(t, dtype=float)
    nes = ex * ex + ey * ey + ez * ez
    circular = (ex <= -0.9999) & (nes > 0.99)

    tau = t - tpa
    f_circ = TWO_PI * (tau - jnp.floor(tau / p) * p) / p

    tf, tcc, ix = ep_lookup(t, tpa, p, dt, ep_table, ep_times)
    x, y, z = pos_c(tcc, jnp.take(coeffs, ix, axis=0))
    edp = (x * ex + y * ey + z * ez) / jnp.sqrt((x * x + y * y + z * z) * nes)
    inside = (edp > -1.0) & (edp < 1.0)
    base = jnp.arccos(jnp.where(inside, edp, 0.0))
    f_ecc = jnp.where(tf < 0.5 * p, base, TWO_PI - base)
    f_ecc = jnp.where(edp <= -1.0, jnp.pi, jnp.where(edp >= 1.0, 0.0, f_ecc))
    return jnp.where(circular, f_circ, f_ecc)
