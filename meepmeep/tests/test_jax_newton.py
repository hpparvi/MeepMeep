"""JAX backend: Newton-Raphson references and orbital-mechanics utilities against numba."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS  # noqa: E402

from meepmeep.backends.jax import newton as jn, utils as ju  # noqa: E402
from meepmeep.backends.numba.newton import newton as nn  # noqa: E402
from meepmeep.backends.numba import utils as nu  # noqa: E402

ORBIT_IDS = list(ORBITS)
TC = 0.8


@pytest.mark.parametrize("name", ORBIT_IDS)
def test_newton_references(name):
    p, a, i, e, w, _ = ORBITS[name]
    t = TC + np.linspace(-2.2, 3.1, 257) * p
    assert_allclose(np.asarray(jn.ea_newton(t, TC, p, e, w)), nn.ea_newton_v(t, TC, p, e, w), atol=1e-12)
    assert_allclose(np.asarray(jn.ta_newton(t, TC, p, e, w)), nn.ta_newton_v(t, TC, p, e, w), atol=1e-11)
    for act, exp in zip(jn.xy_newton(t, TC, p, a, i, e, w), nn.xy_newton_v(t, TC, p, a, i, e, w)):
        assert_allclose(np.asarray(act), exp, atol=1e-10)
    for act, exp in zip(jn.xyz_newton(t, TC, p, a, i, e, w), nn.xyz_newton_v(t, TC, p, a, i, e, w)):
        assert_allclose(np.asarray(act), exp, atol=1e-10)
    assert_allclose(np.asarray(jn.z_newton(t, TC, p, a, i, e, w)), nn.z_newton_v(t, TC, p, a, i, e, w), atol=1e-10)
    assert_allclose(np.asarray(jn.rv_newton(t, 10.0, TC, p, e, w)), nn.rv_newton_v(t, 10.0, TC, p, e, w),
                    atol=1e-10)
    assert_allclose(float(jn.eclipse_light_travel_time(p, a, i, e, w, 1.1)),
                    nn.eclipse_light_travel_time(p, a, i, e, w, 1.1), rtol=1e-9)


@pytest.mark.parametrize("name", ORBIT_IDS)
def test_utils(name):
    p, a, i, e, w, _ = ORBITS[name]
    assert_allclose(np.asarray(ju.eccentricity_vector(i, e, w)), nu.eccentricity_vector(i, e, w), atol=1e-15)
    assert_allclose(float(ju.eclipse_time_offset(p, i, e, w)), nu.eclipse_time_offset(p, i, e, w), rtol=1e-13)
    assert_allclose(float(ju.transit_distance_factor(e, w)), nu.transit_distance_factor(e, w), rtol=1e-14)
    assert_allclose(float(ju.mean_anomaly_at_transit(e, w)), nu.mean_anomaly_at_transit(e, w), rtol=1e-14)
    assert_allclose(float(ju.ta_from_ea(1.1, e)), nu.ta_from_ea(1.1, e), rtol=1e-14)
    assert_allclose(float(ju.mean_anomaly(3.3, TC, p, e, w)), nu.mean_anomaly(3.3, TC, p, e, w), rtol=1e-13)
    assert_allclose(float(ju.z_from_ta(0.4, a, i, e, w)), nu.z_from_ta(0.4, a, i, e, w), rtol=1e-13)
    assert_allclose(float(ju.impact_parameter(a, i)), nu.impact_parameter(a, i), rtol=1e-14)
    assert_allclose(float(ju.impact_parameter_ec(a, i, e, w, 1.0)), nu.impact_parameter_ec(a, i, e, w, 1.0),
                    rtol=1e-14)


def test_transit_helpers():
    assert_allclose(float(ju.i_from_baew(0.3, 12.0, 0.2, 0.7)), nu.i_from_baew(0.3, 12.0, 0.2, 0.7), rtol=1e-14)
    assert_allclose(float(ju.as_from_rhop(1.4, 3.2)), nu.as_from_rhop(1.4, 3.2), rtol=1e-13)
    for kind in (14, 23):
        assert_allclose(float(ju.d_from_pkaiews(3.0, 0.1, 10.0, 1.5, 0.1, 0.3, 1.0, kind)),
                        nu.d_from_pkaiews(3.0, 0.1, 10.0, 1.5, 0.1, 0.3, 1.0, kind), rtol=1e-13)


def test_eccentricity_vector_with_lan():
    i, e, w, lan = 1.4, 0.3, 0.7, 0.9
    assert_allclose(np.asarray(ju.eccentricity_vector(i, e, w, lan)), nu.eccentricity_vector(i, e, w, lan),
                    atol=1e-15)
    assert_allclose(np.asarray(ju.eccentricity_vector(i, 0.0, w, lan)), [-1.0, 0.0, 0.0], atol=0)
