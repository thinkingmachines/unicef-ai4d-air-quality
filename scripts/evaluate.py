from datetime import datetime

import click
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from pandas_profiling import ProfileReport

from src.config import settings
from src.modelling import eval_utils


@click.command()
@click.option(
    "--data-csv",
    default=settings.DATA_DIR / "2022-06-13-openaq-th-2022-01-rollout.csv",
    help="Path to the CSV file containing the dataset with features and ground truth.",
)
@click.option(
    "--pm25-col",
    default="pm2.5",
    help="Column of the ground truth PM2.5 value.",
)
@click.option(
    "--model-path",
    default=settings.DATA_DIR / "final_openaq_model.pkl",
    help="Path to the PM2.5 regression model.",
)
def main(
    data_csv,
    pm25_col,
    model_path,
):
    # Outputs

    run_timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    pred_col = "predicted_pm2.5"
    preds_path = settings.DATA_DIR / f"preds_{run_timestamp}.csv"

    # Read in ground truth
    data_df = pd.read_csv(data_csv)
    data_df = data_df.dropna(subset=[pm25_col])
    # data_df = data_df.dropna(how="any")
    logger.info(f"Predicting on {len(data_df)} data points...")

    # Make predictions
    logger.info("Running the model...")

    # Load Model
    model = joblib.load(model_path)

    # Filter to only the relevant columns
    keep_cols = model.feature_names  # This was saved from the train script
    ml_df = data_df[keep_cols]

    # Run model
    preds = model.predict(ml_df)

    # Evaluate
    eval_results = eval_utils.evaluate(data_df[pm25_col], preds)
    logger.info(eval_results)

    # Post-process to avoid negative values
    preds = np.clip(preds, 0, 500)

    # Join the predictions back and save the CSV file
    data_df[pred_col] = preds
    data_df.to_csv(preds_path, index=False)
    logger.info(f"Saved CSV to {preds_path}")

    profile = ProfileReport(data_df, title="Report for Project X", minimal=True)
    profile.to_file(settings.DATA_DIR / "report.html")

    fig = eval_utils.plot_actual_vs_predicted(data_df[pm25_col], preds)
    fig.savefig(settings.DATA_DIR / "scatterplot_preds.png")


if __name__ == "__main__":
    main()
