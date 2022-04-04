from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src import settings


class CVParams(BaseModel):
    refit: str
    n_trials: int
    verbose: bool


class SelectorParams(BaseModel):
    k: List


class DataParams(BaseModel):
    csv_path: str
    target_col: str
    ignore_cols: List = []


class ExperimentConfig(BaseModel):
    data_params: DataParams
    out_dir: Path = settings.DATA_DIR / "output"
    model: str
    model_params: Optional[dict]
    selector: str
    selector_params: SelectorParams
    cv: str
    cv_params: CVParams
    scalers: List[str]
