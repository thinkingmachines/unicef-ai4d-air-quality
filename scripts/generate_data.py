from datetime import datetime

import click
import geopandas as gpd
import pandas as pd
from loguru import logger

from src.config import settings
from src.data_processing import admin_bounds, feature_collection_pipeline


@click.command()
@click.option(
    "--locations-csv",
    default=settings.DATA_DIR / "2022-04-29-air4thai-th-stations.csv",
    help="Path to the CSV file containing the locations for which to generate data.",
)
@click.option(
    "--ground-truth-csv",
    help="Path to the CSV file containing the locations for which to generate data.",
)
@click.option(
    "--admin-bounds-shp",
    help="If provided, the script will identify the admin bounds of the locations of interest."
    "This is used primarily for spatial cross-validation during model training and evaluation.",
)
@click.option(
    "--id-col",
    default="station_code",
    help="Primary Key to uniquely identify entries in the locations CSV. If ground truth CSV is provided, this should be present in that file as well.",
)
@click.option(
    "--hrsl-tif",
    default=settings.DATA_DIR / "tha_general_2020.tif",
    help="Path to the HRSL tif file containing the population counts for the country.",
)
@click.option(
    "--start-date",
    default="2021-01-01",
    help="Date to start collecting data",
)
@click.option(
    "--end-date",
    default="2021-12-31",
    help="Date to end collecting data",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="If true, will run only on 2 locations just to check if the whole script will run.",
)
def main(
    locations_csv,
    ground_truth_csv,
    admin_bounds_shp,
    hrsl_tif,
    id_col,
    start_date,
    end_date,
    debug,
):
    BBOX_SIZE_KM = 1

    # Read in desired AOI locations
    # Assumed that the CSV has an id column, latitude, and longitude at the minimum.
    locations_df = pd.read_csv(locations_csv)
    if debug:
        logger.warning("Running in debug mode. Trying out with 2 locations only.")
        locations_df = locations_df[:2]
    assert {id_col, "latitude", "longitude"} <= set(locations_df.columns.tolist())

    # Create base DF from the locations, date range, and features (HRSL + GEE data)
    base_df = feature_collection_pipeline.collect_features_for_locations(
        locations_df, start_date, end_date, id_col, hrsl_tif, bbox_size_km=BBOX_SIZE_KM
    )

    # Join ground truth if any
    if ground_truth_csv:
        logger.info(f"Generating dataset with ground truth from {ground_truth_csv}")
        ground_truth_df = pd.read_csv(ground_truth_csv)
        ground_truth_df["date"] = pd.to_datetime(ground_truth_df["date"]).dt.date
        base_df = base_df.merge(ground_truth_df, on=[id_col, "date"], how="left")
    else:
        ground_truth_df = None
        logger.warning("Generating dataset without ground truth.")

    # Join admin bounds if any
    if admin_bounds_shp:
        logger.info(f"Generating dataset with admin bounds from {admin_bounds_shp}")
        admin_bounds_gdf = gpd.read_file(admin_bounds_shp)
        base_df = admin_bounds.join_admin_bounds(base_df, admin_bounds_gdf)
    else:
        admin_bounds_gdf = None
        logger.warning("No admin bounds provided.")

    # Save outputs
    run_timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    out_filepath = f"generated_data_{run_timestamp}.csv"
    base_df.to_csv(settings.DATA_DIR / out_filepath, index=False)
    logger.info(
        f"Generated base table for ML modelling with {len(base_df)} rows. Saved to {out_filepath}"
    )


if __name__ == "__main__":
    main()
