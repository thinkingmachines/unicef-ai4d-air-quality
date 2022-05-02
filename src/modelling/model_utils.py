import collections
import os

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
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


def spatial_cv(c, df, X, y, k=5, out_dir=None):
    outer_cv_result = {metric: [] for metric in eval_utils.get_scoring()}

    grp = c["spatial_cv_params"]["groups"]
    grp_vals = df[grp].astype(str).values

    grp_kfold = GroupKFold(n_splits=k)
    spatial_fold = grp_kfold.split(df, y, grp_vals)
    train_indices, test_indices = [list(traintest) for traintest in zip(*spatial_fold)]
    index = list(range(0, k))

    spatial_c = c.copy()

    all_y_test = []
    all_y_pred = []

    for index, train_index, test_index in zip(index, train_indices, test_indices):

        logger.info(f"Running Spatial Outer Fold: {index}")
        logger.info(f"Train length: {len(train_index)}")

        X_train_cv = df.loc[train_index]
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Getting spatial group kfold per outer fold
        inner_grp_vals = X_train_cv[grp].astype(str).values
        inner_grp_kfold = GroupKFold(n_splits=k)
        inner_spatial_fold = inner_grp_kfold.split(X_train_cv, y_train, inner_grp_vals)
        inner_train_indices, inner_test_indices = [
            list(traintest) for traintest in zip(*inner_spatial_fold)
        ]
        cv = [*zip(inner_train_indices, inner_test_indices)]

        spatial_c["cv_params"]["cv"] = cv

        inner_cv = get_cv(spatial_c)
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
        fig.savefig(os.path.join(out_dir, "scatterplot_nestedspatialcv_combined.png"))
        plt.clf()

    return dict(collections.ChainMap(*[mean_results, outer_cv_result]))


def simple_impute(df, cols, strategy, missing=np.NaN):
    orig_count = len(df)
    logger.info(f"Initial data count: {orig_count:,}")

    if strategy == "None":
        logger.info(f"Removing all null rows (strategy = {strategy})")
        orig_count = len(df)
        df.dropna(how="any", inplace=True)
        df.reset_index(drop=True, inplace=True)
        null_rows = orig_count - len(df)
        if null_rows > 0:
            logger.warning(f"Dropped {null_rows:,} rows with nulls.")
    else:
        logger.info(f"Imputing values (strategy = {strategy})")
        imputer = SimpleImputer(missing_values=missing, strategy=strategy)

        for col in cols:
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))

    logger.info(f"Final data count: {len(df):,}")

    return df
