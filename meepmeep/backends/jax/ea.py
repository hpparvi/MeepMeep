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

import jax
import jax.numpy as jnp
from jax import lax

def _wrap_to_pi(x):
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def _eccentric_anomaly_forward(ma, ecc, n_iter=6):
    ma = _wrap_to_pi(ma)
    ea0 = ma + ecc * jnp.sin(ma) + 0.5 * ecc ** 2 * jnp.sin(2.0 * ma)

    def step(_, ea):
        f  = ea - ecc * jnp.sin(ea) - ma
        fp = 1.0 - ecc * jnp.cos(ea)
        return ea - f / fp

    return lax.fori_loop(0, n_iter, step, ea0)

@jax.custom_jvp
def eccentric_anomaly(ma, ecc):
    return _eccentric_anomaly_forward(ma, ecc)

@eccentric_anomaly.defjvp
def eccentric_anomaly_jvp(primals, tangents):
    ma, ecc = primals
    dma, decc = tangents

    ea = _eccentric_anomaly_forward(ma, ecc)

    denom = 1.0 - ecc * jnp.cos(ea)
    dea_dma = 1.0 / denom
    dea_decc = jnp.sin(ea) / denom

    dea = dea_dma * dma + dea_decc * decc
    return ea, dea