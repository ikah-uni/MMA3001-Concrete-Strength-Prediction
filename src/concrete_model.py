
"""
Utility functions for concrete compressive strength prediction.

This module provides reusable functions for evaluating
machine learning regression models.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_true, y_pred):
    """
    Evaluate regression predictions against measured strengths.

    Parameters
    ----------
    y_true : array-like
        Measured concrete compressive strengths in MPa.
    y_pred : array-like
        Predicted concrete compressive strengths in MPa.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE and R2.

    Raises
    ------
    ValueError
        If inputs are not one-dimensional, have different
        lengths, contain fewer than two observations,
        or contain non-finite values.

    Notes
    -----
    Lower MAE and RMSE indicate better prediction accuracy.
    R2 measures performance relative to predicting the mean.
    """

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(actual) < 2 or len(actual) != len(predicted):
        raise ValueError(
            "Inputs must have equal lengths and at least two observations."
        )

    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Inputs must contain only finite values.")

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    }
