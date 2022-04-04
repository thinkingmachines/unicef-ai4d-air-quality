import os

import click
import pandas as pd
import yaml
from loguru import logger

from src import settings
from src.models import ExperimentConfig


@click.command()
@click.option(
    "--config-path",
    default=settings.CONFIG_DIR / "default.yaml",
    help="Path to the experiment configuration yaml file",
)
def train(config_path):

    # Load and validate config file
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config = ExperimentConfig.parse_obj(config)

    # Prepare output dir
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir, exist_ok=True)

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

    # TODO: Train Model
    # result = model_utils.nested_cv(c, X, y)
    # print("\nNested CV results: {}".format(result))

    # cv = model_utils.get_cv(c)
    # cv.fit(X, y)

    # print("Best estimator: {}".format(cv.best_estimator_))

    # # Save results
    # with open(out_dir + "results.json", "w") as f:
    #     json.dump(result, f)

    # # Save Model
    # joblib.dump(cv.best_estimator_, out_dir + "best_model.pkl")


if __name__ == "__main__":
    train()
