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

"""Shared helpers for the JAX backend.

Holds the constants, the working-dtype rule and the per-dtype Kepler
tolerance, the Horner evaluators of the Taylor polynomial rows, and the two
epoch-folding schemes (single expansion point and multi-expansion-point
table lookup) that every evaluator module builds on.

The coefficient helpers index ``c[..., row, col]``, so they accept a single
``(D, 5)`` matrix (broadcast against an array of times) as well as a stack
``(N, D, 5)`` gathered per time by the multi-expansion-point lookup.
"""

import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi
HALF_PI = 0.5 * jnp.pi


SUPPORTED_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))

# Kepler-solver convergence threshold per working dtype. float64 keeps numba's
# literal 1e-13. float32 cannot reach it (eps ~ 1.2e-7) and would run every
# element to the 50-step cap, so it uses the OpenCL backend's fp32 MM_EA_TOL.
EA_TOLERANCE = {jnp.dtype(jnp.float64): 1e-13, jnp.dtype(jnp.float32): 1e-6}


def _leaves(args):
    for a in args:
        if isinstance(a, (list, tuple)):
            yield from _leaves(a)
        else:
            yield a


def working_dtype(*args):
    """Floating dtype a computation over ``args`` runs in.

    Promotes the arguments with JAX's rules (``jnp.result_type``). Python
    scalars are weakly typed and adopt the dtype of the array arguments, so
    float32 arrays mixed with Python floats stay float32, while an explicit
    float64 array (a NumPy array, say) promotes the computation to float64.
    Lists and tuples count element by element. Integer-only arguments fall
    back to the default float dtype: float64 when ``jax_enable_x64`` is on,
    float32 otherwise. Runs at trace time, so it costs nothing inside a
    jitted function.

    Parameters
    ----------
    args : float, NDArray, list or tuple
        The inputs whose dtypes decide the working precision.

    Returns
    -------
    dtype : numpy.dtype
        ``float32`` or ``float64``.

    Raises
    ------
    TypeError
        If the promoted dtype is neither float32 nor float64. Half precision
        cannot carry the Kepler solve or the fourth-order Taylor coefficients.
    """
    dtype = jnp.result_type(*_leaves(args))
    if jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.bool_):
        dtype = jnp.result_type(float)
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"The MeepMeep JAX backend computes in float32 or float64, not {dtype}.")
    return dtype


def horner(t, c, row):
    """Evaluate one row of the Taylor polynomial (position) at centered time ``t``."""
    return c[..., row, 0] + t * (c[..., row, 1] + t * (c[..., row, 2] + t * (c[..., row, 3] + t * c[..., row, 4])))


def horner_d(t, c, row):
    """Evaluate the time derivative of one row of the Taylor polynomial (velocity)."""
    return c[..., row, 1] + t * (2.0 * c[..., row, 2] + t * (3.0 * c[..., row, 3] + t * 4.0 * c[..., row, 4]))


def fold(time, tc, p, te):
    """Map absolute time to time relative to the nearest image of a single expansion point.

    The expansion point sits at ``tc + te`` on the observation time axis; the
    epoch is chosen so that the returned time lies in ``[-p/2, p/2)``.
    """
    time = jnp.asarray(time, dtype=working_dtype(time, tc, p, te))
    epoch = jnp.floor((time - tc - te + 0.5 * p) / p)
    return time - (tc + te + epoch * p)


def ep_lookup(t, tpa, p, dt, ep_table, ep_times):
    """Fold a time into one period and dispatch it to its expansion point.

    Parameters
    ----------
    t : float or NDArray
        Time(s) [days].
    tpa : float
        Periastron time anchoring the expansion-point grid [days].
    p : float
        Orbital period [days].
    dt : float
        Width of one ``ep_table`` bucket in fraction of the period.
    ep_table : NDArray of int
        Time-to-expansion-point lookup table.
    ep_times : NDArray
        Normalised expansion-point phases.

    Returns
    -------
    tf : float or NDArray
        Time since the most recent periastron passage, in ``[0, p)``.
    tcc : float or NDArray
        Time relative to the selected expansion point.
    ix : int or NDArray of int
        Expansion-point index per time.

    Notes
    -----
    The table index is clipped into range, so a NaN time (whose integer
    cast is implementation-defined) cannot read out of bounds; its value
    comes out NaN through ``tcc`` regardless of the bucket.
    """
    t = jnp.asarray(t, dtype=working_dtype(t, tpa, p, dt, ep_times))
    epoch = jnp.floor((t - tpa) / p)
    tf = t - tpa - epoch * p
    bucket = jnp.floor(tf / (dt * p)).astype(jnp.int32)
    ix = jnp.take(ep_table, bucket, mode='clip')
    return tf, tf - jnp.take(ep_times, ix) * p, ix
