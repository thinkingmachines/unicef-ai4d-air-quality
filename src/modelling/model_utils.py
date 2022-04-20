import collections
import os

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from tune_sklearn import TuneGridSearchCV, TuneSearchCV

from src.config import settings
from src.config.models import ExperimentConfig
from src.modelling import clf_utils, eval_utils, reg_utils


def _get_scalers(scalers):
    """Returns a list of scalers for hyperparameter optimization.
    Args:
        scalers (list): A list of strings indicating the scalers
            to include in the hyperparameter search space.
    Returns:
        list: A list of scaler instances.
    """

    scalers_list = []

    if "MinMaxScaler" in scalers:
        scalers_list.append(MinMaxScaler())
    if "StandardScaler" in scalers:
        scalers_list.append(StandardScaler())
    if "RobustScaler" in scalers:
        scalers_list.append(RobustScaler())
    if "MaxAbsScaler" in scalers:
        scalers_list.append(MaxAbsScaler())
    return scalers_list


def _get_pipeline(model, selector):
    """Instantiates and returns a pipeline based on
    the input configuration.
    Args:
        model (object): The model instance to include in the pipeline.
        selector (object): The selector instance to include in the pipeline.
    Returns:
        sklearn pipeline instance.
    """

    if model in reg_utils.MODELS:
        model = reg_utils.get_model(model)
    elif model in clf_utils.MODELS:
        model = clf_utils.get_model(model)

    if selector in reg_utils.SELECTORS:
        selector = reg_utils.get_selector(selector)
    elif selector in clf_utils.SELECTORS:
        selector = clf_utils.get_selector(selector)

    return Pipeline(
        [
            ("scaler", "passthrough"),
            ("selector", selector),
            ("model", model),
        ]
    )


def _get_params(scalers, model_params, selector_params):
    """Instantiates the parameter grid for hyperparameter optimization.
    Args:
        scalers (dict): A dictionary indicating the the list of scalers.
        model_params (dict): A dictionary containing the model parameters.
        selector_params (dict): A dictionary containing the feature
            selector parameters.
    Returns
        dict: Contains the parameter grid, combined into a single dictionary.
    """

    def _get_range(param):
        if param[0] == "np.linspace":
            return list(np.linspace(*param[1:]).astype(int))
        elif param[0] == "range":
            return list(range(*param[1:]))
        elif param[0] == "list":
            return param[1:]
        return param

    scalers = {"scaler": _get_scalers(scalers)}

    if model_params:
        model_params = {
            "model__" + name: _get_range(param) for name, param in model_params.items()
        }
    else:
        model_params = {}

    if selector_params:
        selector_params = {
            "selector__" + name: _get_range(param)
            for name, param in selector_params.items()
        }
    else:
        selector_params = {}

    params = [model_params, selector_params, scalers]

    return dict(collections.ChainMap(*params))


def get_cv(c, seed=settings.SEED):
    """Returns a model selection instance.
    Args:
        c (dict): The config dictionary indicating the model,
            selector, scalers, parameters, and model selection
            instance.
    Returns:
        object: The model selector instance.
    """

    pipe = _get_pipeline(c["model"], c["selector"])
    params = _get_params(c["scalers"], c["model_params"], c["selector_params"])
    cv, cv_params = c["cv"], c["cv_params"]

    assert cv in [
        "TuneGridSearchCV",
        "TuneSearchCV",
        "RandomizedSearchCV",
        "GridSearchCV",
    ]

    scoring = eval_utils.get_scoring()
    if cv == "TuneGridSearchCV":
        return TuneGridSearchCV(pipe, params, scoring=scoring, **cv_params)
    elif cv == "TuneSearchCV":
        return TuneSearchCV(
            pipe,
            params,
            scoring=scoring,
            random_state=seed,
            **cv_params,
        )
    elif cv == "RandomizedSearchCV":
        return RandomizedSearchCV(
            pipe, params, scoring=scoring, random_state=seed, **cv_params
        )
    elif cv == "GridSearchCV":
        return GridSearchCV(pipe, params, scoring=scoring, **cv_params)


def nested_cv(c: ExperimentConfig, X, y, seed=settings.SEED, k=5, out_dir=None):

    outer_cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    outer_cv_result = {metric: [] for metric in eval_utils.get_scoring()}

    all_y_test = []
    all_y_pred = []

    for index, (train_index, test_index) in enumerate(outer_cv.split(X)):

        logger.info(f"Running Outer Fold: {index}")
        logger.info(f"Train length: {len(train_index)}")

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = get_cv(c)
        inner_cv.fit(X_train, y_train)

        y_pred = inner_cv.best_estimator_.predict(X_test)
        result = eval_utils.evaluate(y_test, y_pred)

        # Accumulate all the y_test and y_pred pairs for producing combined scatterplot.
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        for metric, value in result.items():
            outer_cv_result[metric].append(value)

        logger.info(f"Outer Fold {index} Results: {result}")

    mean_results = {}
    for metric, values in outer_cv_result.items():
        mean_results["mean_" + metric] = np.mean(values)

    if out_dir:
        fig = eval_utils.plot_actual_vs_predicted(all_y_test, all_y_pred)
        fig.savefig(os.path.join(out_dir, "scatterplot_nestedcv_combined.png"))
        plt.clf()

    return dict(collections.ChainMap(*[mean_results, outer_cv_result]))


def spatial_cv(c, df, X, y, seed=settings.SEED, k=5):
    cv_result = {metric: [] for metric in eval_utils.get_scoring()}

    grp = c["spatial_cv_params"]["groups"]
    grp_vals = df[grp].astype(str).values

    grp_kfold = GroupKFold(n_splits=k)
    spatial_fold = grp_kfold.split(df, y, grp_vals)
    train_indices, test_indices = [list(traintest) for traintest in zip(*spatial_fold)]
    cv = [*zip(train_indices, test_indices)]

    spatial_c = c.copy()
    spatial_c["cv_params"]["cv"] = cv

    cv = get_cv(spatial_c)
    cv.fit(X, y)

    y_pred = cv.best_estimator_.predict(X)
    result = eval_utils.evaluate(y, y_pred)

    for metric, value in result.items():
        cv_result[metric].append(value)

    logger.info(f"Results: {result}")

    return cv_result


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
    ax.set_xlabel("SHAP Value (Red = Negative Impact)")

    plt.savefig(dir, bbox_inches="tight")
    plt.clf()
