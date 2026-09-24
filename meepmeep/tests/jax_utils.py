"""Shared helpers for the JAX backend tests. Not a test module.

Importing this module requires jax, so every test module must guard with
``pytest.importorskip("jax")`` *before* importing from here. Importing it
switches JAX to double precision, which the backend requires.

The JAX backend has no hand-written gradient kernels; its gradients are
autodiff through the value code. The suites therefore compare JAX values
against the numba value kernels and ``jax.jacfwd`` Jacobians against the
numba ``_d``/``_od`` kernels. Both are exact derivatives of the same
evaluated polynomial, so they should agree to round-off.
"""

import jax
import numpy as np
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

# (p, a, i, e, w, lan): circular, mildly and strongly eccentric, edge-on,
# retrograde-ish node, and a near-parabolic stress case.
ORBITS = {
    "circular": (3.0, 10.0, 1.50, 0.0, 0.0, 0.0),
    "eccentric": (5.0, 15.0, 1.55, 0.3, 0.5, 0.2),
    "high_e": (7.0, 20.0, 1.40, 0.7, 1.2, -0.4),
    "edge_on": (2.5, 8.0, 0.5 * np.pi, 0.1, 0.3, 0.0),
    "very_high_e": (11.0, 30.0, 1.52, 0.9, 4.0, 1.1),
}


def jacobian(f, n, *args):
    """Stack ``jax.jacfwd`` over the first ``n`` arguments into a trailing axis.

    ``f`` returns an array (or a tuple of arrays); the result has the same
    structure with a trailing axis of length ``n``.
    """
    jac = jax.jacfwd(f, argnums=tuple(range(n)))(*args)
    if isinstance(jac[0], tuple):
        return tuple(jnp.stack(jm, axis=-1) for jm in jac)
    return jnp.stack(jac, axis=-1)


def assert_grad_close(actual, desired, rtol=1e-9, atol_scale=1e-11):
    """Compare gradients with an absolute tolerance scaled by each slot's magnitude.

    Analytically-zero slots come out as round-off of either sign, so a pure
    relative tolerance cannot be used; ``atol`` is ``atol_scale`` times the
    largest magnitude of the desired gradient in the same slot, floored by
    the largest magnitude overall.
    """
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    assert actual.shape == desired.shape, (actual.shape, desired.shape)
    d2 = desired.reshape(-1, desired.shape[-1])
    scale = np.maximum(np.abs(d2).max(axis=0), 1e-3 * np.abs(d2).max())
    for k in range(desired.shape[-1]):
        assert_allclose(actual[..., k], desired[..., k], rtol=rtol, atol=atol_scale * scale[k] + 1e-300,
                        err_msg=f"gradient slot {k}")
