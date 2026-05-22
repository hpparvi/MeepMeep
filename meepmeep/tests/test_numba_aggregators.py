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
