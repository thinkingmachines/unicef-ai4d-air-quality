import json
import os
from datetime import datetime

import click
import joblib
import pandas as pd
import yaml
from loguru import logger

from src import settings
from src.modelling import model_utils
from src.models import ExperimentConfig


@click.command()
@click.option(
    "--config-path",
    default=settings.CONFIG_DIR / "default.yaml",
    help="Path to the experiment configuration yaml file",
)
def train(config_path):

    # Experiment Configuration: Load and validate yaml #
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config = ExperimentConfig.parse_obj(config)

    # Data Preparation #

    # Read in data
    data_df = pd.read_csv(config.data_params.csv_path)
    logger.info(f"Loaded {len(data_df):,} rows from {config.data_params.csv_path}")

    # Prepare features and target
    target_col = config.data_params.target_col
    feature_cols = [
        feature
        for feature in data_df.columns
        if feature not in config.data_params.ignore_cols + [target_col]
    ]
    logger.info(f"Features: {feature_cols}, Target: {target_col}")

    X = data_df[feature_cols]
    y = data_df[target_col].values

    # Model Training
    result = model_utils.nested_cv(config.dict(), X, y)
    logger.info("\nNested CV results: {}".format(result))

    cv = model_utils.get_cv(config.dict())
    cv.fit(X, y)

    logger.info("Best estimator: {}".format(cv.best_estimator_))

    # Serialize Model and Results #

    # Prepare output dir
    out_dir = config.out_dir / datetime.today().strftime("%Y-%m-%d-%H_%M_%S")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save results
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f)

    # Save Model
    joblib.dump(cv.best_estimator_, out_dir / "best_model.pkl")


if __name__ == "__main__":
    train()
