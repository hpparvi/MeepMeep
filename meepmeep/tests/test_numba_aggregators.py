"""Surface-integrity tests for the public numba2d / numba3d aggregators.

These tests assert the contract that ``meepmeep.numba2d`` and
``meepmeep.numba3d`` are the canonical public low-level API: every name
in ``__all__`` is importable and points at a callable (or, for
``TWO_PI``, a real number), and ``__all__`` agrees with the set of
non-underscore module attributes.
"""
from numbers import Real

import pytest

from meepmeep import numba2d, numba3d


@pytest.fixture(params=[numba2d, numba3d], ids=["numba2d", "numba3d"])
def aggregator(request):
    return request.param


def test_all_is_defined(aggregator):
    assert hasattr(aggregator, "__all__"), "aggregator must define __all__"
    assert isinstance(aggregator.__all__, list)
    assert len(aggregator.__all__) > 0


def test_every_all_entry_resolves(aggregator):
    missing = [n for n in aggregator.__all__ if not hasattr(aggregator, n)]
    assert not missing, f"__all__ lists names not present on module: {missing}"


def test_every_all_entry_is_callable_or_real(aggregator):
    bad = []
    for name in aggregator.__all__:
        obj = getattr(aggregator, name)
        if isinstance(obj, Real) or callable(obj):
            continue
        bad.append((name, type(obj).__name__))
    assert not bad, f"non-callable, non-numeric entries in __all__: {bad}"


def test_all_matches_public_attributes(aggregator):
    """Every public (non-underscore) attribute that is a function or
    numeric constant defined for re-export must appear in __all__."""
    declared = set(aggregator.__all__)
    public = set()
    for name in dir(aggregator):
        if name.startswith("_"):
            continue
        obj = getattr(aggregator, name)
        if callable(obj) or isinstance(obj, Real):
            public.add(name)
    extra = public - declared
    missing = declared - public
    assert not extra, f"public attributes not declared in __all__: {sorted(extra)}"
    assert not missing, f"__all__ entries not present as public attributes: {sorted(missing)}"


# ----------------------------------------------------------------------
# Numba-callability smoke tests
# ----------------------------------------------------------------------
# These compile a tiny @njit function that calls a handful of
# representative re-exports from each aggregator. If the re-export
# layer ever breaks Numba's dispatch (for example, by wrapping a jitted
# function in a pure-Python decorator), these tests catch it.

import numpy as np
from numba import njit


def test_numba_smoke_numba2d():
    from meepmeep.numba2d import solve2d, sep, t14

    @njit(cache=False)
    def run():
        c = solve2d(0.0, 3.0, 10.0, 1.5, 0.0, 0.0)
        s = sep(0.0, 0.0, 3.0, c)
        # k = 0.1 -> total transit duration t14 should be finite for this geometry
        d = t14(0.1, c)
        return s, d

    s, d = run()
    assert np.isfinite(s)
    assert np.isfinite(d)


def test_numba_smoke_numba3d():
    from meepmeep.numba3d import (
        TWO_PI,
        create_knots,
        solve3d_orbit,
        pos_o,
        mean_anomaly_at_transit,
        ta_from_ea,
    )

    NPT = 15
    p, a, i, e, w = 3.0, 10.0, 1.5, 0.0, 0.0

    knot_times, _, dt, pktable = create_knots(NPT, max(e, 0.2), "ea")
    coeffs = solve3d_orbit(knot_times, p, a, i, e, w, npt=NPT)
    t0_periastron = -mean_anomaly_at_transit(e, w) / TWO_PI * p

    @njit(cache=False)
    def run(times, tpa, p_, dt_, pkt, pts, c):
        x, y, z = pos_o(times, tpa, p_, dt_, pkt, pts, c)
        v = ta_from_ea(0.3, 0.0)
        return x[0], y[0], z[0], v

    times = np.linspace(0.0, p, 11)
    x0, y0, z0, v = run(times, t0_periastron, p, dt, pktable, knot_times, coeffs)
    assert np.isfinite(x0) and np.isfinite(y0) and np.isfinite(z0)
    assert np.isfinite(v)
