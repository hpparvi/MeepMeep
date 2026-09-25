"""Surface-integrity tests for the public jax2d / jax3d aggregators.

Mirrors ``test_numba_aggregators.py``: every ``__all__`` entry resolves to a
callable, ``__all__`` agrees with the public attributes, and every JAX name
has a numba namesake (``JaxOrbit`` and the model-building helpers aside).
"""

import pytest

jax = pytest.importorskip("jax")

from meepmeep.tests import jax_utils  # noqa: E402, F401  (enables x64)

import jax.numpy as jnp  # noqa: E402

from meepmeep import jax2d, jax3d, numba2d, numba3d  # noqa: E402

JAX_ONLY = {"JaxOrbit", "ea_from_ma"}


@pytest.fixture(params=[(jax2d, numba2d), (jax3d, numba3d)], ids=["jax2d", "jax3d"])
def pair(request):
    return request.param


def test_every_all_entry_resolves_to_callable(pair):
    agg, _ = pair
    bad = [n for n in agg.__all__ if not callable(getattr(agg, n, None))]
    assert not bad


def test_all_matches_public_attributes(pair):
    agg, _ = pair
    public = {n for n in dir(agg) if not n.startswith("_") and callable(getattr(agg, n))}
    assert public == set(agg.__all__)


def test_names_mirror_numba(pair):
    agg, nb = pair
    unmatched = set(agg.__all__) - set(nb.__all__) - JAX_ONLY
    assert not unmatched, f"JAX names without a numba namesake: {sorted(unmatched)}"


def test_value_surface_is_complete(pair):
    """Every numba value evaluator has a JAX port; only the gradient, vector and
    basis-transform variants are intentionally absent."""
    agg, nb = pair
    skip_suffixes = ("_d", "_cd", "_od", "_v", "_vp", "_ov", "_ovp", "_ovd", "_ovdp", "_d_v", "_d_vp")
    skip = {"tc_to_tp_gradient", "tp_to_tc_gradient", "tp_to_tc_gradient_orbit"}
    missing = [n for n in nb.__all__
               if not n.endswith(skip_suffixes) and n not in skip and n not in agg.__all__]
    assert not missing, f"numba value functions without a JAX port: {missing}"


def test_jax_smoke_jit():
    @jax.jit
    def run(t):
        c = jax2d.solve2d(0.0, 3.0, 10.0, 1.5, 0.0, 0.0)
        return jax2d.sep(t, 0.0, 3.0, c), jax2d.t14(0.1, c)

    s, d = run(jnp.linspace(-0.1, 0.1, 5))
    assert jnp.all(jnp.isfinite(s)) and jnp.isfinite(d)
