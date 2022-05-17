import os
from datetime import datetime

import click
import geopandas as gpd
import pandas as pd
from loguru import logger

from src.config import settings
from src.data_processing import general_utils, hrsl
from src.data_processing.gee import aod, era5, gee_utils, ndvi


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

    # Auth with GEE
    gee_utils.gee_auth(service_acct=False)

    # Read in desired AOI locations
    # Assumed that the CSV has an id column, latitude, and longitude at the minimum.
    locations_df = pd.read_csv(locations_csv)
    if debug:
        logger.warning("Running in debug mode. Trying out with 2 locations only.")
        locations_df = locations_df[:2]
    assert {id_col, "latitude", "longitude"} <= set(locations_df.columns.tolist())

    # Read in ground truth if any
    if ground_truth_csv:
        ground_truth_df = pd.read_csv(ground_truth_csv)
        ground_truth_df["date"] = pd.to_datetime(ground_truth_df["date"]).dt.date
        logger.info(f"Generating dataset with ground truth from {ground_truth_csv}")
    else:
        ground_truth_df = None
        logger.warning("Generating dataset without ground truth.")

    # Read in admin bounds if any
    if admin_bounds_shp:
        admin_bounds_gdf = gpd.read_file(admin_bounds_shp)
        logger.info(f"Generating dataset with admin bounds from {admin_bounds_shp}")
    else:
        admin_bounds_gdf = None
        logger.warning("No admin bounds provided.")

    # Compute HRSL stats
    logger.info("Computing population sums...")
    hrsl_df = hrsl.collect_hrsl(
        locations_df, hrsl_tif, id_col=id_col, bbox_size_km=BBOX_SIZE_KM
    )

    # Collect GEE Datasets
    gee_datasets = [
        {
            "collection_id": "COPERNICUS/S5P/OFFL/L3_AER_AI",
            "bands": [
                "absorbing_aerosol_index",
            ],
            "preprocessors": [(aod.aggregate_daily_s5p_aerosol, {"id_col": id_col})],
        },
        {
            "collection_id": "ECMWF/CAMS/NRT",
            "bands": [
                # "total_aerosol_optical_depth_at_469nm_surface", # Results in errors cause it can be missing sometimes
                "total_aerosol_optical_depth_at_550nm_surface",
            ],
            "preprocessors": [
                (aod.rescale_cams_aod, {}),
                (aod.aggregate_daily_cams_aod, {"id_col": id_col}),
            ],
        },
        {
            "collection_id": "MODIS/006/MCD19A2_GRANULES",  # Aerosol Optical Depth (AOD)
            "bands": ["Optical_Depth_047", "Optical_Depth_055"],
            "preprocessors": [(aod.aggregate_daily_aod, {"id_col": id_col})],
        },
        {
            "collection_id": "MODIS/006/MOD13A2",  # Vegetation
            "bands": ["NDVI", "EVI"],
            "preprocessors": [
                (
                    ndvi.aggregate_daily_ndvi,
                    {"start_date": start_date, "end_date": end_date, "id_col": id_col},
                )
            ],
        },
        {
            "collection_id": "ECMWF/ERA5_LAND/HOURLY",  # Meteorological Variables
            "bands": [
                "dewpoint_temperature_2m",
                "temperature_2m",
                "total_precipitation_hourly",
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
                "surface_pressure",
            ],
            "preprocessors": [(era5.aggregate_daily_era5, {"id_col": id_col})],
        },
    ]

    gee_dfs = general_utils.collect_gee_datasets(
        gee_datasets, start_date, end_date, locations_df, id_col=id_col
    )

    # Save outputs
    run_timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    # Debug logs to check intermediate files
    for collection, df in gee_dfs.items():
        logger.debug(f"{collection}: {len(df)} rows")
        debug_dir = settings.DATA_DIR / "debug"
        os.makedirs(debug_dir, exist_ok=True)
        collection_name_sanitized = collection.replace("/", "_")
        df.to_csv(
            debug_dir / f"{collection_name_sanitized}_{run_timestamp}.csv", index=False
        )

    # Generate final DF and save to CSV
    logger.info("Joining all datasets together...")
    base_df = general_utils.join_datasets(
        locations_df,
        start_date,
        end_date,
        gee_dfs,
        hrsl_df,
        id_col,
        ground_truth_df=ground_truth_df,
        admin_bounds_gdf=admin_bounds_gdf,
    )
    out_filepath = f"generated_data_{run_timestamp}.csv"
    base_df.to_csv(settings.DATA_DIR / out_filepath, index=False)
    logger.info(
        f"Generated base table for ML modelling with {len(base_df)} rows. Saved to {out_filepath}"
    )


if __name__ == "__main__":
    main()
