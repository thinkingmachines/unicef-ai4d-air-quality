from datetime import datetime

import click
import pandas as pd
from loguru import logger

from src.config import settings
from src.data_processing import geom_utils
from src.prediction import predict_utils


@click.command()
@click.option(
    "--locations-csv",
    help="Path to the CSV file containing the locations for which to generate data.",
)
@click.option(
    "--id-col",
    default="id",
    help="Primary Key to uniquely identify entries in the locations CSV. If ground truth CSV is provided, this should be present in that file as well.",
)
@click.option(
    "--model-path",
    default=settings.DATA_DIR / "latest_model.pkl",
    help="Path to the PM2.5 regression model.",
)
@click.option(
    "--hrsl-tif",
    default=settings.DATA_DIR / "tha_general_2020.tif",
    help="Path to the HRSL tif file containing the population counts for the country.",
)
@click.option(
    "--start-date",
    default="2021-01-01",
    help="Date to start prediction",
)
@click.option(
    "--end-date",
    default="2021-12-31",
    help="Date to end prediction",
)
@click.option(
    "--out-path",
    help="Where to save the output.",
)
@click.option(
    "--generate-bbox",
    is_flag=True,
    default=False,
    help="If true, script will augment the predictions file with the bounding box geometry for downstream purposes (e.g. viz).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="If true, will run only on 2 locations just to check if the whole script will run.",
)
def main(
    locations_csv,
    id_col,
    model_path,
    hrsl_tif,
    start_date,
    end_date,
    out_path,
    generate_bbox,
    debug,
):
    # This depends on the model. Our model is trained on agggregated features 1km x 1km around the station.
    BBOX_SIZE_KM = 1

    locations_df = pd.read_csv(locations_csv)

    if debug:
        logger.warning("Running in debug mode. Trying out on 2 locations only.")
        locations_df = locations_df[:2]

    results_df = predict_utils.predict(
        locations_df,
        start_date,
        end_date,
        id_col,
        hrsl_tif,
        model_path,
        bbox_size_km=1,
        pred_col="predicted_pm2.5",
    )

    if generate_bbox:
        logger.info(
            f"Augmenting results with the bounding boxes ({BBOX_SIZE_KM}km x {BBOX_SIZE_KM}km)"
        )
        geom_utils.generate_bboxes(results_df, BBOX_SIZE_KM)

    if not out_path:
        run_timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = settings.DATA_DIR / f"predictions_{run_timestamp}.csv"
        logger.info(f"Saved results to {out_path}")
        results_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
