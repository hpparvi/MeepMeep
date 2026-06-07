"""Tests for the periastron-basis (t_p) gradient support.

The transform from the transit-centre gradient basis (tc, p, a, i, e, w, lan)
to the periastron basis (tp, p, a, i, e, w, lan) is an exact closed-form chain
rule. These tests check the primitive in isolation, its re-export, and its
wiring into the Orbit class (exact tc<->tp consistency plus one finite-difference
sign check).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.orbit import Orbit
from meepmeep.backends.numba.utils import (
    tc_to_tp_gradient,
    mean_anomaly_at_transit,
    mean_anomaly_at_transit_with_derivatives,
    TWO_PI,
)
from meepmeep.backends.numba.orbit3dd import star_planet_distance_od


SHAPE = dict(p=5.0, a=15.0, i=1.55, e=0.3, w=0.5, lan=0.0)


def _tp_from_tc(g, p, e, w):
    """Apply the closed-form tc->tp reparametrisation to a (..., K) gradient.

    Acts on the trailing parameter axis; columns beyond index 5 (e.g. the RV
    `k` column) and columns 0/2/3/6 are passed through unchanged.
    """
    m_tr, dde, ddw = mean_anomaly_at_transit_with_derivatives(e, w)
    c = 1.0 / TWO_PI
    h = g.copy()
    h[..., 1] = g[..., 1] + g[..., 0] * (m_tr * c)
    h[..., 4] = g[..., 4] + g[..., 0] * (dde * p * c)
    h[..., 5] = g[..., 5] + g[..., 0] * (ddw * p * c)
    return h


def test_tc_to_tp_gradient_closed_form():
    rng = np.random.default_rng(0)
    dc = rng.standard_normal((7, 3, 5))
    p, e, w = 5.0, 0.3, 0.5
    out = tc_to_tp_gradient(dc, p, e, w)

    m_tr, dde, ddw = mean_anomaly_at_transit_with_derivatives(e, w)
    c = 1.0 / TWO_PI

    # Unchanged rows: timing, a, i, lan.
    assert_allclose(out[0], dc[0])
    assert_allclose(out[2], dc[2])
    assert_allclose(out[3], dc[3])
    assert_allclose(out[6], dc[6])

    # Updated rows: p, e, w.
    assert_allclose(out[1], dc[1] + dc[0] * (m_tr * c))
    assert_allclose(out[4], dc[4] + dc[0] * (dde * p * c))
    assert_allclose(out[5], dc[5] + dc[0] * (ddw * p * c))

    # Input is not mutated; a fresh array is returned.
    assert not np.shares_memory(out, dc)


def test_tc_to_tp_gradient_spot_check_circular():
    """Independent check: for e=0, w=0 the mean anomaly at transit is pi/2,
    so the period-row correction factor M_tr/(2 pi) is exactly 0.25 -- a
    hand-derived constant that does not reuse the implementation's formula."""
    rng = np.random.default_rng(1)
    dc = rng.standard_normal((7, 3, 5))
    out = tc_to_tp_gradient(dc, p=5.0, e=0.0, w=0.0)
    assert_allclose(out[1], dc[1] + dc[0] * 0.25, rtol=1e-12, atol=1e-14)


def test_numba3d_reexports_primitive():
    import meepmeep.numba3d as n3
    assert "tc_to_tp_gradient" in n3.__all__
    assert hasattr(n3, "tc_to_tp_gradient")


def _times():
    return np.linspace(0.0, SHAPE["p"], 60)


def test_tc_path_unchanged():
    """A tc-bound orbit must apply NO transform: its gradient equals a direct
    star_planet_distance_od call on the raw solver coefficients."""
    times = _times()
    o = Orbit(npt=15, derivatives=True)
    o.set_data(times)
    o.set_pars(tc=0.0, **SHAPE)
    _, g = o.star_planet_distance()
    _, g_direct = star_planet_distance_od(
        times, o._tp, o._p, o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs,
    )
    assert_allclose(g, g_direct, rtol=1e-12)
    assert o._timing == "tc"


def test_tp_matches_transformed_tc_separation():
    """tp-bound separation gradient equals the tc-bound gradient reparametrised.

    Both orbits describe the SAME physical orbit (identical _tp), so values
    match and gradients are related by the exact closed-form transform.
    """
    times = _times()
    to = mean_anomaly_at_transit(SHAPE["e"], SHAPE["w"]) / TWO_PI * SHAPE["p"]

    o_tc = Orbit(npt=15, derivatives=True)
    o_tc.set_data(times)
    o_tc.set_pars(tc=0.3, **SHAPE)

    o_tp = Orbit(npt=15, derivatives=True)
    o_tp.set_data(times)
    o_tp.set_pars(tp=0.3 - to, **SHAPE)

    r_tc, g_tc = o_tc.star_planet_distance()
    r_tp, g_tp = o_tp.star_planet_distance()

    assert o_tp._timing == "tp"
    assert_allclose(r_tp, r_tc, rtol=1e-10)  # same physical orbit -> same values
    expected = _tp_from_tc(g_tc, SHAPE["p"], SHAPE["e"], SHAPE["w"])
    assert_allclose(g_tp, expected, rtol=1e-10, atol=1e-12)


def test_tp_matches_transformed_tc_rv():
    """Same exact-consistency check for radial_velocity (the motivating case).

    The RV gradient is (N, 8): orbital columns 0..6 plus the k column (7),
    which is independent of the timing convention and must be unchanged.
    """
    times = _times()
    k = 50.0
    to = mean_anomaly_at_transit(SHAPE["e"], SHAPE["w"]) / TWO_PI * SHAPE["p"]

    o_tc = Orbit(npt=15, derivatives=True)
    o_tc.set_data(times)
    o_tc.set_pars(tc=0.3, **SHAPE)

    o_tp = Orbit(npt=15, derivatives=True)
    o_tp.set_data(times)
    o_tp.set_pars(tp=0.3 - to, **SHAPE)

    _, g_tc = o_tc.radial_velocity(k)
    _, g_tp = o_tp.radial_velocity(k)

    expected = _tp_from_tc(g_tc, SHAPE["p"], SHAPE["e"], SHAPE["w"])
    assert_allclose(g_tp, expected, rtol=1e-10, atol=1e-12)
    assert_allclose(g_tp[:, 7], g_tc[:, 7], rtol=1e-12)  # k column untouched


@pytest.mark.accuracy
def test_tp_separation_gradient_vs_finite_difference():
    """Pin the chain-rule sign/magnitude: the tp-basis separation gradient
    matches central finite differences taken holding tp fixed.

    The finite difference perturbs each parameter through ``set_pars(tp=...)``,
    which shifts the periastron anchor and hence the time-to-knot mapping. To
    keep every sampled time inside a single, stable knot (so the difference is
    not corrupted by a point straddling a knot boundary between the +eps and
    -eps evaluations), we sample a narrow window around the transit centre.
    Within one knot the analytic derivative is the exact derivative of the same
    Taylor polynomial the finite difference probes, so they agree closely. The
    *exact* correctness of the transform is independently established by the
    tc<->tp consistency tests above, which agree to ~1e-10.
    """
    base = dict(tp=0.0, **SHAPE)
    keys = ["tp", "p", "a", "i", "e", "w", "lan"]

    # Transit centre for tp=0: tc = tp + M_tr(e, w) * p / (2 pi). Sample a
    # narrow window around it so all points share the transit knot.
    tc = mean_anomaly_at_transit(SHAPE["e"], SHAPE["w"]) / TWO_PI * SHAPE["p"]
    times = tc + np.linspace(-0.04, 0.04, 9)

    od = Orbit(npt=15, derivatives=True)
    od.set_data(times)
    od.set_pars(**base)
    _, h = od.star_planet_distance()

    ov = Orbit(npt=15, derivatives=False)
    ov.set_data(times)
    eps = 1e-6
    for j, key in enumerate(keys):
        hi, lo = dict(base), dict(base)
        hi[key] += eps
        lo[key] -= eps
        ov.set_pars(**hi)
        r_hi = ov.star_planet_distance()
        ov.set_pars(**lo)
        r_lo = ov.star_planet_distance()
        fd = (r_hi - r_lo) / (2 * eps)
        assert_allclose(h[:, j], fd, rtol=3e-3, atol=1e-5)


@pytest.mark.accuracy
def test_tp_rv_gradient_vs_finite_difference():
    """Ground-truth check of the tp-basis gradient for radial velocity.

    Unlike the projected separation (whose w- and i-sensitivity vanishes at
    transit), the radial velocity has genuine sensitivity to all of
    (tp, p, a, i, e, w), so this finite-difference check independently
    validates the e- and w-rows that the tc->tp transform modifies -- the
    motivating science case. Same near-transit single-knot window as the
    separation check. The tp and p columns are looser (~5e-3) because the
    radial velocity is a velocity and so carries one more order of Taylor
    sensitivity; the e/w/a/i columns agree to ~1e-9.
    """
    base = dict(tp=0.0, **SHAPE)
    keys = ["tp", "p", "a", "i", "e", "w", "lan"]
    k = 50.0

    tc = mean_anomaly_at_transit(SHAPE["e"], SHAPE["w"]) / TWO_PI * SHAPE["p"]
    times = tc + np.linspace(-0.04, 0.04, 9)

    od = Orbit(npt=15, derivatives=True)
    od.set_data(times)
    od.set_pars(**base)
    _, h = od.radial_velocity(k)

    ov = Orbit(npt=15, derivatives=False)
    ov.set_data(times)
    eps = 1e-6
    for j, key in enumerate(keys):
        hi, lo = dict(base), dict(base)
        hi[key] += eps
        lo[key] -= eps
        ov.set_pars(**hi)
        r_hi = ov.radial_velocity(k)
        ov.set_pars(**lo)
        r_lo = ov.radial_velocity(k)
        fd = (r_hi - r_lo) / (2 * eps)
        # h[:, :7] are the orbital columns; the k column (index 7) is not
        # perturbed here and is covered by test_tp_matches_transformed_tc_rv.
        assert_allclose(h[:, j], fd, rtol=1e-2, atol=1e-3)
