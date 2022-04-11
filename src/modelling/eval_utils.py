import numpy as np
from scipy import stats
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def pearson_r2(y_test, y_pred):
    """Return the pearson's r-squared value.
    Args:
        y_test (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.
    Returns:
        float: The pearson's r-squared value.
    """

    return stats.pearsonr(y_test, y_pred)[0] ** 2


def rmse(y_test, y_pred):
    """Return the root mean squared error (RMSE).
    Args:
        y_test (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.
    Returns:
        float: The RMSE value.
    """

    return np.sqrt(mean_squared_error(y_test, y_pred))


def evaluate(y_test, y_pred):
    """Returns a dictionary of performance metrics.
    Args:
        y_test (list or numpy array): A list of ground truth values.
        y_pred (list of numpy array): A list of prediction values.
    Returns:
        dict: A dictionary of performance metrics.
    """

    return {
        "sklearn_r2": r2_score(y_test, y_pred),
        "pearson_r2": pearson_r2(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
    }


def get_scoring():
    """Returns the dictionary of scorer objects."""

    return {
        "sklearn_r2": make_scorer(r2_score),
        "pearson_r2": make_scorer(pearson_r2),
        "rmse": make_scorer(rmse),
        "mae": make_scorer(mean_absolute_error),
        "mape": make_scorer(mean_absolute_percentage_error),
    }
