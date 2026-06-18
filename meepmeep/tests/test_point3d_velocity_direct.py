"""Parity tests for the direct single-expansion-point 3D velocity evaluators.

The direct ``vel`` / ``vel_d`` evaluators are "epoch-fold, then call the
centered ``vel_c`` / ``vel_cd``" — the absolute observation time is folded
into a single orbital epoch around the expansion point at ``tc + te`` before
the 4th-order velocity polynomial is evaluated. These tests pin that
contract three ways:

* within the home epoch the direct evaluator equals the centered one fed the
  hand-shifted time (``vel(t) == vel_c(t - tc - te)``);
* the direct evaluator is periodic across epochs (``vel(t) == vel(t + p)``),
  which actually exercises the fold rather than re-deriving its formula;
* the gradient-returning ``vel_d`` returns values identical to ``vel``.

Parity against the centered routines is preferred over finite differences
here: a finite-difference probe in ``tc`` or ``p`` would shift the fold and
can remap a sample across an expansion-point boundary, giving O(1) error at
isolated points (see CLAUDE.md).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.point3d import vel_c, vel
from meepmeep.backends.numba.point3dd import solve3d_d, vel_cd, vel_d

PARS = {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5, "lan": 0.7}
TC = 1.234            # transit centre on the absolute time axis
# Bit-identical fold arithmetic, so exact in principle; a tiny atol absorbs
# ulp-level fastmath contraction differences between inlining contexts.
TOL = {"rtol": 1e-12, "atol": 1e-14}
# Folding across several epochs is NOT bit-exact: the absolute times carry
# less precision far from tc, so the floor()-based fold subtraction leaves a
# ~1e-15 slack that the 4th-order velocity polynomial amplifies to ~1e-13.
# A relative tolerance still catches a genuinely wrong epoch (off by O(1)+).
PERIODIC_TOL = {"rtol": 1e-9, "atol": 1e-11}


@pytest.fixture(scope="module")
def coeffs():
    """(c, dc) solved at the transit centre (te = 0)."""
    return solve3d_d(0.0, **PARS)


@pytest.fixture
def in_epoch_times():
    """Times strictly inside the home epoch |t - TC| < p/2 (so epoch == 0)."""
    return TC + np.linspace(-2.0, 2.0, 41)


class TestVelDirectVsCentered:
    """Direct evaluators equal the centered ones fed the hand-shifted time."""

    def test_vel_matches_centered(self, coeffs, in_epoch_times):
        c, _ = coeffs
        vx, vy, vz = vel(in_epoch_times, TC, PARS["p"], c)
        cx, cy, cz = vel_c(in_epoch_times - TC, c)
        assert_allclose(vx, cx, **TOL)
        assert_allclose(vy, cy, **TOL)
        assert_allclose(vz, cz, **TOL)

    def test_vel_d_matches_centered(self, coeffs, in_epoch_times):
        c, dc = coeffs
        vx, vy, vz, dvx, dvy, dvz = vel_d(in_epoch_times, TC, PARS["p"], c, dc)
        cx, cy, cz, dcx, dcy, dcz = vel_cd(in_epoch_times - TC, c, dc)
        assert_allclose(vx, cx, **TOL)
        assert_allclose(vy, cy, **TOL)
        assert_allclose(vz, cz, **TOL)
        assert_allclose(dvx, dcx, **TOL)
        assert_allclose(dvy, dcy, **TOL)
        assert_allclose(dvz, dcz, **TOL)


class TestVelEpochFolding:
    """The fold makes the direct evaluators periodic across epochs."""

    @pytest.mark.parametrize("nepoch", [-3, -1, 1, 4])
    def test_vel_periodic(self, coeffs, in_epoch_times, nepoch):
        c, _ = coeffs
        p = PARS["p"]
        base = vel(in_epoch_times, TC, p, c)
        shifted = vel(in_epoch_times + nepoch * p, TC, p, c)
        for b, s in zip(base, shifted):
            assert_allclose(s, b, **PERIODIC_TOL)

    @pytest.mark.parametrize("nepoch", [-3, -1, 1, 4])
    def test_vel_d_periodic(self, coeffs, in_epoch_times, nepoch):
        c, dc = coeffs
        p = PARS["p"]
        base = vel_d(in_epoch_times, TC, p, c, dc)
        shifted = vel_d(in_epoch_times + nepoch * p, TC, p, c, dc)
        for b, s in zip(base, shifted):
            assert_allclose(s, b, **PERIODIC_TOL)

    def test_vel_nonzero_te_matches_centered(self, in_epoch_times):
        """With a non-zero expansion point te, the fold centres on tc + te."""
        te = 0.6
        c, dc = solve3d_d(te, **PARS)
        # Keep times inside the home epoch around tc + te.
        t = TC + te + np.linspace(-2.0, 2.0, 41)
        vx, vy, vz = vel(t, TC, PARS["p"], c, te)
        cx, cy, cz = vel_c(t - TC - te, c)
        assert_allclose([vx, vy, vz], [cx, cy, cz], **TOL)
        # And periodicity still holds with te set.
        sx, sy, sz = vel(t + 2 * PARS["p"], TC, PARS["p"], c, te)
        assert_allclose([sx, sy, sz], [vx, vy, vz], **PERIODIC_TOL)


class TestVelValueGradientConsistency:
    """vel_d must return the same velocity values as the value-only vel."""

    def test_vel_d_values_match_vel(self, coeffs, in_epoch_times):
        c, dc = coeffs
        vx, vy, vz = vel(in_epoch_times, TC, PARS["p"], c)
        gx, gy, gz, _, _, _ = vel_d(in_epoch_times, TC, PARS["p"], c, dc)
        assert_allclose([gx, gy, gz], [vx, vy, vz], **TOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
