from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.config import settings


class DataParams(BaseModel):
    csv_path: str
    target_col: str
    include_cols: List[str] = []
    ignore_cols: List[str] = []
    impute_cols: Optional[list] = None
    impute_strategy: Optional[str] = None
    balance_target_label: Optional[str] = None

    def infer_selected_features(self, full_feature_list):
        if self.include_cols:
            feature_cols = [
                feature for feature in full_feature_list if feature in self.include_cols
            ]
        else:
            feature_cols = [
                feature
                for feature in full_feature_list
                if feature not in self.ignore_cols + [self.target_col]
            ]
        return feature_cols


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
    spatial_cv_params: Optional[dict]
