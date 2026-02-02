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