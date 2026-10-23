
import numpy as np
import pytest

from src.concrete_model import evaluate_model


def test_perfect_predictions():
    """Perfect predictions should have zero error and R² = 1."""
    actual = [10, 20, 30]
    predicted = [10, 20, 30]

    results = evaluate_model(actual, predicted)

    assert np.isclose(results["MAE"], 0)
    assert np.isclose(results["RMSE"], 0)
    assert np.isclose(results["R2"], 1)


def test_known_errors():
    """Check the metrics against manually calculated values."""
    actual = [10, 20, 30]
    predicted = [12, 18, 30]

    results = evaluate_model(actual, predicted)

    assert np.isclose(results["MAE"], 4 / 3)
    assert np.isclose(results["RMSE"], np.sqrt(8 / 3))
    assert np.isclose(results["R2"], 0.96)


def test_mismatched_lengths():
    """Inputs with different lengths must be rejected."""
    with pytest.raises(ValueError):
        evaluate_model([10, 20, 30], [10, 20])


def test_missing_values():
    """Inputs containing NaN must be rejected."""
    with pytest.raises(ValueError):
        evaluate_model([10, np.nan, 30], [10, 20, 30])


def test_insufficient_observations():
    """At least two observations are required to calculate R²."""
    with pytest.raises(ValueError):
        evaluate_model([10], [10])
