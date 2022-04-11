from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from xgboost import XGBRegressor

from src.config import settings

SEED = 42
SELECTORS = [
    "SelectKBest_f_regression",
    "SelectKBest_mutual_info_regression",
    "VarianceThreshold",
    "RFE",
]
MODELS = [
    "LinearRegression",
    "Lasso",
    "Ridge,",
    "ElasticNet",
    "SGDRegressor",
    "LinearSVR",
    "SVR",
    "NuSVR",
    "GaussianProcessRegressor",
    "RandomForestRegressor",
    "AdaBoostRegressor",
    "GradientBoostingRegressor",
    "MLPRegressor",
    "LGBMRegressor",
    "XGBRegressor",
]


def get_selector(selector):
    """Instantiates and returns a selector instance.
    Args:
        selector (str): Indicates the selector to instantiate.
    Returns:
        object: The selector instance for feature selection.
    """

    assert selector in SELECTORS

    if selector == "SelectKBest_f_regression":
        return SelectKBest(f_regression)
    elif selector == "SelectKBest_mutual_info_regression":
        return SelectKBest(mutual_info_regression)
    elif selector == "VarianceThreshold":
        return VarianceThreshold()
    elif selector == "RFE":
        return RFE()


def get_model(model, seed=settings.SEED):
    """Instantiates and returns a model instance.
    Args:
        model (str): Indicates the model to instantiate.
    Returns:
        object: The model instance for development.
    """

    assert model in MODELS

    if model == "LinearRegression":
        return LinearRegression()
    elif model == "Lasso":
        return Lasso(random_state=seed)
    elif model == "Ridge":
        return Ridge(random_state=seed)
    elif model == "ElasticNet":
        return ElasticNet(random_state=seed)
    elif model == "SGDRegressor":
        return SGDRegressor(random_state=seed)
    elif model == "LinearSVR":
        return LinearSVR(random_state=seed)
    elif model == "SVR":
        return SVR()
    elif model == "NuSVR":
        return NuSVR()
    elif model == "GaussianProcessRegressor":
        return GaussianProcessRegressor(random_state=seed)
    elif model == "RandomForestRegressor":
        return RandomForestRegressor(random_state=seed)
    elif model == "AdaBoostRegressor":
        return AdaBoostRegressor(random_state=seed)
    elif model == "GradientBoostingRegressor":
        return GradientBoostingRegressor(random_state=seed)
    elif model == "MLPRegressor":
        return MLPRegressor(random_state=seed)
    elif model == "LGBMRegressor":
        return LGBMRegressor(random_state=seed)
    elif model == "XGBRegressor":
        return XGBRegressor(random_state=seed)
