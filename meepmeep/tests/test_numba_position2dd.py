"""Test suite for meepmeep.backends.numba.taylor.position2dd module.

Validates the 2D parameter-derivative evaluators (pos_cd, pos_d, sep_cd,
sep_d) against finite differences of the corresponding plain evaluators.
The derivative tensor produced by solve2d_d has shape (7, 2, 5) with the
seventh row holding the longitude-of-ascending-node (lan) derivative, so
the evaluators must return length-7 gradients that include lan.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.taylor.solve2d import solve2d
from meepmeep.backends.numba.taylor.solve2dd import solve2d_d
from meepmeep.backends.numba.taylor.position2d import pos_c, sep_c
from meepmeep.backends.numba.taylor.position2dd import pos_cd, pos_d, sep_cd, sep_d


@pytest.fixture
def eccentric_orbit():
    return {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}


# Parameter axis of the derivative tensor: (t0, p, a, i, e, w, lan).
PARAM_NAMES = ("t0", "p", "a", "i", "e", "w", "lan")


def _solve2d_perturbed(t_expand, orbit, lan, kidx, delta):
    """Return solve2d coefficients with parameter ``kidx`` shifted by ``delta``."""
    args = [t_expand, orbit["p"], orbit["a"], orbit["i"], orbit["e"], orbit["w"]]
    lan_val = lan
    if kidx < 6:
        args[kidx] += delta
    else:
        lan_val = lan + delta
    return solve2d(*args, lan=lan_val)


class TestPos2dDLan:
    """pos_cd / pos_d must return length-7 gradients including lan."""

    def test_pos_cd_gradient_length(self, eccentric_orbit):
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=0.7)
        _, _, dpx, dpy = pos_cd(0.003, cf, dcf)
        assert dpx.shape == (7,)
        assert dpy.shape == (7,)

    def test_pos_d_gradient_length(self, eccentric_orbit):
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=0.7)
        _, _, dpx, dpy = pos_d(0.003, 0.0, eccentric_orbit["p"], cf, dcf)
        assert dpx.shape == (7,)
        assert dpy.shape == (7,)

    def test_pos_cd_lan_derivative_is_nonzero(self, eccentric_orbit):
        """The position components rotate with lan, so their lan derivative is
        not identically zero (a regression guard against dropping the lan row)."""
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=0.7)
        _, _, dpx, dpy = pos_cd(0.05, cf, dcf)
        assert abs(dpx[6]) > 1e-6 or abs(dpy[6]) > 1e-6

    @pytest.mark.parametrize("kidx", range(7))
    def test_pos_cd_derivative_vs_finite_diff(self, eccentric_orbit, kidx):
        """Each of the 7 position partials matches a central finite difference."""
        lan = 0.7
        time = 0.003
        h = 1e-6
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=lan)
        _, _, dpx, dpy = pos_cd(time, cf, dcf)

        cf_p = _solve2d_perturbed(0.0, eccentric_orbit, lan, kidx, +h)
        cf_m = _solve2d_perturbed(0.0, eccentric_orbit, lan, kidx, -h)
        xp, yp = pos_c(time, cf_p)
        xm, ym = pos_c(time, cf_m)
        dpx_fd = (xp - xm) / (2 * h)
        dpy_fd = (yp - ym) / (2 * h)
        # Slot 0 is d/dt0; the perturbation above is in the expansion-time
        # argument t (d/dt), and d/dt0 = -d/dt, so the slot-0 reference is negated.
        if kidx == 0:
            dpx_fd, dpy_fd = -dpx_fd, -dpy_fd

        assert_allclose(dpx[kidx], dpx_fd, rtol=1e-5, atol=1e-7,
                        err_msg=f"dpx/d{PARAM_NAMES[kidx]} mismatch")
        assert_allclose(dpy[kidx], dpy_fd, rtol=1e-5, atol=1e-7,
                        err_msg=f"dpy/d{PARAM_NAMES[kidx]} mismatch")


class TestSep2dDLan:
    """sep_cd / sep_d must return length-7 gradients including lan."""

    def test_sep_cd_gradient_length(self, eccentric_orbit):
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=0.7)
        _, dd = sep_cd(0.003, cf, dcf)
        assert dd.shape == (7,)

    def test_sep_d_gradient_length(self, eccentric_orbit):
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=0.7)
        _, dd = sep_d(0.003, 0.0, eccentric_orbit["p"], cf, dcf)
        assert dd.shape == (7,)

    @pytest.mark.parametrize("kidx", range(7))
    def test_sep_cd_derivative_vs_finite_diff(self, eccentric_orbit, kidx):
        """Each of the 7 separation partials matches a central finite difference."""
        lan = 0.7
        time = 0.003
        h = 1e-6
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=lan)
        _, dd = sep_cd(time, cf, dcf)

        cf_p = _solve2d_perturbed(0.0, eccentric_orbit, lan, kidx, +h)
        cf_m = _solve2d_perturbed(0.0, eccentric_orbit, lan, kidx, -h)
        d_p = sep_c(time, cf_p)
        d_m = sep_c(time, cf_m)
        dd_fd = (d_p - d_m) / (2 * h)
        # Slot 0 is d/dt0; the perturbation above is in the expansion-time
        # argument t (d/dt), and d/dt0 = -d/dt, so the slot-0 reference is negated.
        if kidx == 0:
            dd_fd = -dd_fd

        assert_allclose(dd[kidx], dd_fd, rtol=1e-5, atol=1e-7,
                        err_msg=f"dsep/d{PARAM_NAMES[kidx]} mismatch")

    def test_sep_cd_lan_derivative_is_zero(self, eccentric_orbit):
        """The projected separation is rotation-invariant, so its lan derivative
        is identically zero (the norm is preserved under the R(lan) rotation)."""
        cf, dcf = solve2d_d(0.0, **eccentric_orbit, lan=0.7)
        _, dd = sep_cd(0.05, cf, dcf)
        assert_allclose(dd[6], 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
