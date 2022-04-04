from typing import List

from pydantic import BaseModel

from src import settings


class CVParams(BaseModel):
    cv: str = "TuneSearchCV"
    refit: str = "pearson_r2"
    n_trials: int = 10
    verbose: bool = False


class SelectorParams(BaseModel):
    selector: str = "SelectKBest_f_regression"
    k: List = ["range", 5, 10]


class ModelParams(BaseModel):
    model: str = "LinearRegression"
    selector_params: SelectorParams
    scalers: List[str] = ["MinMaxScaler", "StandardScaler", "RobustScaler"]
    cv_params: CVParams


class DataParams(BaseModel):
    csv_path: str
    target_col: str
    ignore_cols: List = []


class ExperimentConfig(BaseModel):
    data_params: DataParams
    out_dir: str = settings.DATA_DIR / "output"
    model_params: ModelParams


#     data_path: './data/dhs_ookla.csv'
# out_dir: './exp/exp-01/'
# model: 'LinearRegression'
# model_params:
# selector: 'SelectKBest_f_regression'
# selector_params:
#     k: ['range', 5, 10]
# scalers: ['MinMaxScaler', 'StandardScaler', 'RobustScaler']
# cv: 'TuneSearchCV'
# cv_params:
#    refit: 'pearson_r2'
#    n_trials: 10
#    verbose: 0
