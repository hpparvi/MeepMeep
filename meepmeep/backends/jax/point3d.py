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

"""Single-expansion-point 3D Taylor evaluators.

JAX ports of the ``meepmeep.backends.numba.point3d`` package. Each
quantity has a centered evaluator ``X_c`` (time already relative to the
expansion point) and a direct evaluator ``X`` (absolute time, epoch-folded
around the expansion point at ``tc + te``). The functions are element-wise:
a scalar time gives a scalar, an array of times an array of the same shape.
They trace, ``jit``, ``vmap`` and differentiate like any other JAX code.

There are no ``_d``/``_cd`` gradient variants and no ``_v``/``_vp`` vector
kernels. For parameter gradients, differentiate a function that builds the
coefficients with :func:`~meepmeep.jax3d.solve3d` and
evaluates them here; the epoch-folding chain term (``d/dp += epoch *
d/dtc`` in the numba kernels) comes out of autodiff by itself.

The coefficient argument ``c`` is a ``(3, 5)`` matrix, or a stack
``(..., 3, 5)`` broadcastable against the times.
"""

import jax
import jax.numpy as jnp

from ._common import TWO_PI, horner, horner_d, fold


# Position
# --------

def pos_c(time, c):
    """Position ``(x, y, z)`` [R_star] at time ``time`` relative to the expansion point."""
    return horner(time, c, 0), horner(time, c, 1), horner(time, c, 2)


def pos(time, tc, p, c, te=0.0):
    """Position ``(x, y, z)`` [R_star] at absolute time ``time``.

    Parameters
    ----------
    time : float or NDArray
        Time [days].
    tc : float
        Transit-centre time [days].
    p : float
        Orbital period [days].
    c : NDArray, shape (3, 5)
        Coefficients from :func:`~meepmeep.jax3d.solve3d`.
    te : float, optional
        Expansion-point offset from the transit centre, the same value given
        to the solver. Defaults to 0.0.
    """
    return pos_c(fold(time, tc, p, te), c)


def zpos_c(time, c):
    """Line-of-sight coordinate z [R_star] at centered time."""
    return horner(time, c, 2)


def zpos(time, tc, p, c, te=0.0):
    """Line-of-sight coordinate z [R_star] at absolute time. See :func:`pos` for the arguments."""
    return zpos_c(fold(time, tc, p, te), c)


def sep_c(time, c):
    """Sky-projected separation [R_star] at centered time.

    The sky-projected separation between the centers of the star and the
    planet, in units of the stellar radius.
    """
    px = horner(time, c, 0)
    py = horner(time, c, 1)
    return jnp.sqrt(px ** 2 + py ** 2)


def sep(time, tc, p, c, te=0.0):
    """Projected separation [R_star] at absolute time. See :func:`pos` for the arguments."""
    return sep_c(fold(time, tc, p, te), c)


# Velocity
# --------

def vel_c(time, c):
    """Velocity ``(vx, vy, vz)`` [R_star / day] at centered time."""
    return horner_d(time, c, 0), horner_d(time, c, 1), horner_d(time, c, 2)


def vel(time, tc, p, c, te=0.0):
    """Velocity ``(vx, vy, vz)`` [R_star / day] at absolute time. See :func:`pos` for the arguments."""
    return vel_c(fold(time, tc, p, te), c)


def zvel_c(time, c):
    """Line-of-sight velocity [R_star / day] at centered time."""
    return horner_d(time, c, 2)


def zvel(time, tc, p, c, te=0.0):
    """Line-of-sight velocity [R_star / day] at absolute time. See :func:`pos` for the arguments."""
    return zvel_c(fold(time, tc, p, te), c)


def _rv_scale(k, p, a, i, e):
    """Factor converting the line-of-sight velocity [R_star / day] to a radial velocity of semi-amplitude k."""
    return k / (TWO_PI / p * (a * jnp.sin(i)) / jnp.sqrt(1.0 - e ** 2))


def rv_c(time, k, p, a, i, e, c):
    """Radial velocity with semi-amplitude ``k`` at centered time."""
    return zvel_c(time, c) * _rv_scale(k, p, a, i, e)


def rv(time, k, tc, p, a, i, e, c, te=0.0):
    """Radial velocity with semi-amplitude ``k`` at absolute time."""
    return zvel(time, tc, p, c, te) * _rv_scale(k, p, a, i, e)


# Phase angle and phase curves
# ----------------------------

def cos_alpha_c(time, c):
    """Cosine of the star-planet-observer phase angle at centered time, ``-z / r``."""
    px, py, pz = pos_c(time, c)
    return -pz / jnp.sqrt(px ** 2 + py ** 2 + pz ** 2)


def cos_alpha(time, tc, p, c, te=0.0):
    """Cosine of the phase angle at absolute time. See :func:`pos` for the arguments."""
    return cos_alpha_c(fold(time, tc, p, te), c)


@jax.custom_jvp
def lambert_kernel(cos_alpha):
    """Lambert-sphere phase function of the phase-angle cosine.

    ``cos_alpha`` is clipped into ``[-1, 1]`` first, so Taylor rounding
    overshoot cannot produce a NaN. The derivative, ``(pi - alpha) / pi``,
    is finite everywhere; it comes from a ``custom_jvp`` rule because
    differentiating through ``arccos`` would give ``0 * inf = NaN`` at full
    phase (``alpha = 0``).
    """
    ca = jnp.clip(cos_alpha, -1.0, 1.0)
    sin_alpha = jnp.sqrt(1.0 - ca * ca)
    alpha = jnp.arccos(ca)
    return (sin_alpha + (jnp.pi - alpha) * ca) / jnp.pi


@lambert_kernel.defjvp
def _lambert_kernel_jvp(primals, tangents):
    ca, = primals
    dca, = tangents
    alpha = jnp.arccos(jnp.clip(ca, -1.0, 1.0))
    return lambert_kernel(ca), (jnp.pi - alpha) / jnp.pi * dca


def lambert_phase_curve_c(time, ag, k, c):
    """Lambert-sphere reflected-light phase curve at centered time.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the expansion point [days].
    ag : float
        Geometric albedo.
    k : float
        Planet-star radius ratio.
    c : NDArray, shape (3, 5)
        Taylor coefficients.

    Returns
    -------
    flux : float or NDArray
        Planet-star flux ratio ``k^2 ag / r^2 * Phi(alpha)``, with ``r`` the
        star-planet distance in stellar radii.
    """
    px, py, pz = pos_c(time, c)
    r2 = px * px + py * py + pz * pz
    return k * k * ag / r2 * lambert_kernel(-pz / jnp.sqrt(r2))


def lambert_phase_curve(time, ag, k, tc, p, c, te=0.0):
    """Lambert-sphere phase curve at absolute time. See :func:`lambert_phase_curve_c` and :func:`pos`."""
    return lambert_phase_curve_c(fold(time, tc, p, te), ag, k, c)


def ev_signal_c(time, alpha, mass_ratio, inc, c):
    """Ellipsoidal-variation signal at centered time.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the expansion point [days].
    alpha : float
        Ellipsoidal-variation amplitude coefficient.
    mass_ratio : float
        Planet-star mass ratio.
    inc : float
        Inclination [radians].
    c : NDArray, shape (3, 5)
        Taylor coefficients.

    Returns
    -------
    ev : float or NDArray
        ``-alpha q sin^2(inc) (2 (z/d)^2 - 1) / d^3``, with ``d`` the 3D
        star-planet distance in stellar radii.
    """
    sin_inc = jnp.sin(inc)
    pre = -alpha * mass_ratio * sin_inc * sin_inc
    px, py, pz = pos_c(time, c)
    d2 = px * px + py * py + pz * pz
    d = jnp.sqrt(d2)
    cz = pz / d
    return pre * (2.0 * cz * cz - 1.0) / (d2 * d)


def ev_signal(time, alpha, mass_ratio, inc, tc, p, c, te=0.0):
    """Ellipsoidal-variation signal at absolute time. See :func:`ev_signal_c` and :func:`pos`."""
    return ev_signal_c(fold(time, tc, p, te), alpha, mass_ratio, inc, c)


def emission_phase_curve_c(time, k, fratio, offset, c):
    """Thermal-emission phase curve with a hotspot offset at centered time.

    Parameters
    ----------
    time : float or NDArray
        Time relative to the expansion point [days].
    k : float
        Planet-star radius ratio.
    fratio : float
        Day-side surface brightness ratio.
    offset : float
        Hotspot offset [radians], positive eastward.
    c : NDArray, shape (3, 5)
        Taylor coefficients.

    Returns
    -------
    flux : float or NDArray
        ``k^2 fratio / 2 * (1 + cos(offset) cz + sin(offset) s)``, where
        ``cz = -z / d`` is the cosine of the phase angle and
        ``s = -(n_x y - n_y x) / d`` the signed in-plane component, with
        ``n = (r x v) / |r x v|`` the orbital normal.
    """
    x, y, z = pos_c(time, c)
    vx, vy, vz = vel_c(time, c)
    d = jnp.sqrt(x * x + y * y + z * z)
    wx = y * vz - z * vy
    wy = z * vx - x * vz
    wz = x * vy - y * vx
    el = jnp.sqrt(wx * wx + wy * wy + wz * wz)
    cz = -z / d
    m = wx * y - wy * x
    s = -m / (el * d)
    g = 0.5 * (1.0 + jnp.cos(offset) * cz + jnp.sin(offset) * s)
    return k * k * fratio * g


def emission_phase_curve(time, k, fratio, offset, tc, p, c, te=0.0):
    """Thermal-emission phase curve at absolute time. See :func:`emission_phase_curve_c` and :func:`pos`."""
    return emission_phase_curve_c(fold(time, tc, p, te), k, fratio, offset, c)
