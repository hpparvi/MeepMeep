"""Shared pytest fixtures and configuration for MeepMeep tests."""
import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def test_orbital_params():
    """Collection of test orbital parameter sets."""
    return {
        "circular": {
            "p": 3.0,
            "a": 10.0,
            "i": 1.5,
            "e": 0.0,
            "w": 0.0,
        },
        "eccentric": {
            "p": 5.0,
            "a": 15.0,
            "i": 1.55,
            "e": 0.3,
            "w": 0.5,
        },
        "high_e": {
            "p": 7.0,
            "a": 20.0,
            "i": 1.4,
            "e": 0.7,
            "w": 1.2,
        },
        "edge_on": {
            "p": 2.5,
            "a": 8.0,
            "i": np.pi / 2,
            "e": 0.1,
            "w": 0.3,
        },
    }


@pytest.fixture
def tolerance_params():
    """Tolerance parameters for numerical comparisons."""
    return {
        "near": {"rtol": 1e-6, "atol": 1e-8},    # Near expansion point
        "medium": {"rtol": 1e-4, "atol": 1e-6},  # Medium distance
        "far": {"rtol": 1e-2, "atol": 1e-4},     # Far from expansion
    }
