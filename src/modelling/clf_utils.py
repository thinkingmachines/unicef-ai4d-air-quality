from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from xgboost import XGBClassifier

from src import settings

SELECTORS = [
    "SelectKBest",
    "SelectKBest_chi2",
    "SelectKBest_f_classif",
    "SelectKBest_mutual_info_classif",
    "RFE",
    "VarianceThreshold",
]
MODELS = [
    "LogisticRegression",
    "SGDClassifier",
    "RidgeClassifier",
    "LinearSVC",
    "SVC",
    "NuSVC",
    "MLPClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "MultinomialNB",
    "GaussianProcessClassifier",
    "LGBMClassifier",
    "XGBClassifier",
]


def get_selector(selector):
    """Instantiates and returns a selector instance.

    Args:
        selector (str): Indicates the selector to instantiate.

    Returns:
        object: The selector for feature selection.
    """

    assert selector in SELECTORS

    if selector == "SelectKBest":
        return SelectKBest()
    elif selector == "SelectKBest_chi2":
        return SelectKBest(chi2)
    elif selector == "SelectKBest_f_classif":
        return SelectKBest(f_classif)
    elif selector == "SelectKBest_mutual_info_classif":
        return SelectKBest(mutual_info_classif)
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

    if model == "LogisticRegression":
        return LogisticRegression(max_iter=1e8, random_state=seed)
    elif model == "SGDClassifier":
        return SGDClassifier(random_state=seed)
    elif model == "RidgeClassifier":
        return RidgeClassifier(random_state=seed)
    elif model == "LinearSVC":
        return LinearSVC(max_iter=1e8, random_state=seed)
    elif model == "SVC":
        return SVC(random_state=seed)
    elif model == "NuSVC":
        return NuSVC(random_state=seed)
    elif model == "RandomForestClassifier":
        return RandomForestClassifier(random_state=seed)
    elif model == "GradientBoostingClassifier":
        return GradientBoostingClassifier(random_state=seed)
    elif model == "AdaBoostClassifier":
        return AdaBoostClassifier(random_state=seed)
    elif model == "MultinomialNB":
        return MultinomialNB()
    elif model == "GaussianProcessClassifier":
        return GaussianProcessClassifier(random_state=seed)
    elif model == "MLPClassifier":
        return MLPClassifier(random_state=seed)
    elif model == "LGBMClassifier":
        return LGBMClassifier(random_state=seed)
    elif model == "XGBClassifier":
        return XGBClassifier(random_state=seed)
