import json
import os
from datetime import datetime

import click
import joblib
import pandas as pd
import shap
import yaml
from loguru import logger

from src.config import settings
from src.config.models import ExperimentConfig
from src.modelling import model_utils


@click.command()
@click.option(
    "--config-path",
    default=settings.CONFIG_DIR / "default.yaml",
    help="Path to the experiment configuration yaml file",
)
def train(config_path):

    # Experiment Configuration: Load and validate yaml #
    with open(config_path, "r") as config_file:
        raw_config_yaml = yaml.safe_load(config_file)
    config = ExperimentConfig.parse_obj(raw_config_yaml)

    # Data Preparation #

    # Read in data
    data_df = pd.read_csv(config.data_params.csv_path)
    logger.info(f"Loaded {len(data_df):,} rows from {config.data_params.csv_path}")

    # Remove any rows with nulls
    orig_count = len(data_df)
    data_df.dropna(how="any", inplace=True)
    data_df.reset_index(drop=True, inplace=True)
    null_rows = orig_count - len(data_df)
    if null_rows > 0:
        logger.warning(
            f"Dropped {null_rows:,} rows with nulls. Final data count: {len(data_df):,}"
        )

    # Prepare features and target
    target_col = config.data_params.target_col
    feature_cols = [
        feature
        for feature in data_df.columns
        if feature not in config.data_params.ignore_cols + [target_col]
    ]
    logger.info(f"Target: {target_col}, {len(feature_cols)} Features: {feature_cols}, ")

    X = data_df[feature_cols]
    y = data_df[target_col].values

    # Model Training
    nested_cv_results = model_utils.nested_cv(config.dict(), X, y)
    logger.info(f"\nNested CV results: {json.dumps(nested_cv_results, indent=4)}")

    cv = model_utils.get_cv(config.dict())
    cv.fit(X, y)

    spatial_cv_results = model_utils.spatial_cv(config.dict(), data_df, X, y)
    logger.info(f"\nSpatial CV results: {json.dumps(spatial_cv_results, indent=4)}")

    logger.info(f"Best estimator: {cv.best_estimator_}")

    # Generate feature importance
    # extracting best model from CV
    model = cv.best_estimator_[2]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values).set_axis(X.columns, axis=1)

    # Serialize Model and Results #

    # Prepare output dir
    out_dir = config.out_dir / datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save results
    with open(out_dir / "nested_cv_results.json", "w") as f:
        json.dump(nested_cv_results, f, indent=4)

    # Save results
    with open(out_dir / "spatial_cv_results.json", "w") as f:
        json.dump(spatial_cv_results, f, indent=4)

    # Save Model
    joblib.dump(cv.best_estimator_, out_dir / "best_model.pkl")
    with open(out_dir / "best_model_params.txt", "w") as f:
        print(str(cv.best_estimator_), file=f)

    # Save Feature Importance
    model_utils.get_shap(shap_df, X, out_dir / "best_model_shap.png")

    # Copy over config file so we keep track of the configuration
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(raw_config_yaml, f, default_flow_style=False)


if __name__ == "__main__":
    train()
