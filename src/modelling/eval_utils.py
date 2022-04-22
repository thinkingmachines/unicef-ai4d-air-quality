import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def plot_actual_vs_predicted(y_true, y_pred):
    plt.clf()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    ax_max = max(y_true + y_pred) + 50
    ax = plt.gca()
    ax.set_xlim([0, ax_max])
    ax.set_ylim([0, ax_max])
    plt.axline((0, 0), slope=1, color="black")
    return plt.gcf()


def generate_simplified_shap(
    df_shap,
    df,
    dir,
    top_n=20,
    figsize=(15, 13),
    color=["#4682B4", "#CD5C5C"],
    barwidth=0.8,
):
    try:
        shap_v = pd.DataFrame(df_shap)
    except ValueError:
        shap_v = pd.DataFrame(df_shap[1])

    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop("index", axis=1)

    # Determine the correlation in order to plot with different colors (pearson)
    corr_list = []
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)

    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(
        0
    )

    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ["feature", "corr"]
    corr_df["sign"] = np.where(corr_df["corr"] > 0, color[0], color[1])

    # Plot
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ["feature", "SHAP_abs"]

    k2 = k.merge(corr_df, on="feature", how="inner")
    k2 = k2.nlargest(top_n, "SHAP_abs")
    k2 = k2.sort_values(by="SHAP_abs", ascending=True)
    colorlist = k2["sign"]

    ax = k2.plot.barh(
        x="feature",
        y="SHAP_abs",
        width=barwidth,
        color=colorlist,
        figsize=figsize,
        legend=False,
    )
    ax.set_xlabel(
        "Average Impact on Model Output (Blue = Positive Feature Correlation)"
    )
    plt.rcParams.update({"font.size": 20})

    plt.savefig(dir, bbox_inches="tight")
    plt.clf()
