"""JAX backend: expansion-point placement against the numba (scipy brentq) version."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

jax = pytest.importorskip("jax")

from meepmeep.tests import jax_utils  # noqa: E402, F401  (enables x64)

import jax.numpy as jnp  # noqa: E402

from meepmeep.backends.jax.expansion_points import create_expansion_points  # noqa: E402
from meepmeep.backends.numba.expansion_points import create_expansion_points as nb_create  # noqa: E402


@pytest.mark.parametrize("quantity", ["mm", "ea", "ta"])
@pytest.mark.parametrize("n_ep", [7, 15, 25])
@pytest.mark.parametrize("e", [0.0, 0.2, 0.5, 0.8, 0.95])
def test_matches_numba(quantity, n_ep, e):
    expected = nb_create(n_ep, e, quantity)
    ep_times, change_times, dt, ep_table = create_expansion_points(n_ep, e, quantity)
    # brentq's default xtol is 2e-12
    assert_allclose(np.asarray(ep_times), expected[0], rtol=0, atol=1e-11)
    assert_allclose(np.asarray(change_times), expected[1], rtol=0, atol=1e-11)
    assert dt == expected[2]
    # A bin edge that coincides with a change time (e = 0 makes the anomaly grids
    # uniform in time) is decided by round-off: brentq's root and the closed form
    # can fall on opposite sides of the strict comparison. Only such ties may differ.
    table = np.asarray(ep_table)
    edges = np.arange(len(table)) * dt
    ties = np.min(np.abs(edges[:, None] - expected[1][None, :]), axis=1) < 1e-10
    assert_array_equal(table[~ties], expected[3][~ties])
    assert np.all(np.abs(table - expected[3]) <= 1)
    assert table.dtype == np.int32


def test_traceable_in_eccentricity():
    @jax.jit
    def grid(e):
        return create_expansion_points(15, e, 'ea')

    for e in (0.1, 0.6):
        for act, exp in zip(grid(e), nb_create(15, e, 'ea')):
            assert_allclose(np.asarray(act), exp, atol=1e-11)


def test_placement_is_differentiable():
    g = jax.jacfwd(lambda e: create_expansion_points(15, e, 'ta')[0])(0.4)
    assert np.all(np.isfinite(np.asarray(g)))
    # The midpoint and the periodic endpoints do not move with e.
    assert_allclose(np.asarray(g)[[0, 7, 14]], 0.0, atol=0)


def test_periodic_image_contract():
    ep_times = np.asarray(create_expansion_points(15, 0.3, 'ea')[0])
    assert ep_times[0] == 0.0 and ep_times[-1] == 1.0 and ep_times[7] == 0.5


@pytest.mark.parametrize("kwargs, match", [(dict(n_ep=14, e=0.1), "odd"),
                                           (dict(n_ep=15, e=0.1, quantity="xx"), "Quantity")])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_expansion_points(**kwargs)


def test_int32_table_indexes_like_numba_table():
    _, _, dt, table = create_expansion_points(15, 0.3, 'ea')
    assert jnp.issubdtype(table.dtype, jnp.integer)
    assert int(table[0]) == 0 and int(table[-1]) == 14
