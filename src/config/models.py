from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.config import settings


class DataParams(BaseModel):
    csv_path: str
    target_col: str
    ignore_cols: List[str] = []


class ExperimentConfig(BaseModel):
    data_params: DataParams
    out_dir: Path = settings.DATA_DIR / "output"
    model: str
    model_params: Optional[dict]
    selector: str
    selector_params: Optional[dict]
    cv: str
    cv_params: Optional[dict]
    scalers: List[str]
