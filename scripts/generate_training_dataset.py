import os

import click
import ee
import google.auth
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from src.config import settings
from src.data_processing import aod, era5, gee_utils, ndvi


@click.command()
@click.option(
    "--th-stations-path",
    default=settings.DATA_DIR / "air4thai_th_stations_test.csv",
    help="Path to the CSV file containing list of TH stations.",
)
@click.option(
    "--pm25-path",
    default=settings.DATA_DIR / "air4thai_daily_pm25.csv",
    help="Path to the daily PM2.5 CSV file.",
)
@click.option(
    "--start-date",
    default="2021-01-01",
    help="Date to start collecting data",
)
@click.option(
    "--end-date",
    default="2022-01-01",
    help="Date to end collecting data",
)
def main(th_stations_path, pm25_path, start_date, end_date):
    # Stations and PM2.5 values
    th_stations_df = pd.read_csv(th_stations_path)
    pm25_df = pd.read_csv(pm25_path)

    # TODO: assert that th stations and pm2.5 have intersection through station_code col.

    # Collect GEE Datasets here
    gee_datasets = [
        {
            "collection_id": "MODIS/006/MCD19A2_GRANULES",  # Aerosol Optical Depth (AOD)
            "bands": ["Optical_Depth_047", "Optical_Depth_055"],
            "preprocessors": [aod.aggregate_daily_aod],
        },
        {
            "collection_id": "MODIS/006/MOD13A2",  # Vegetation
            "bands": ["NDVI", "EVI"],
            "preprocessors": [ndvi.aggregate_daily_ndvi],
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
            "preprocessors": [era5.aggregate_daily_era5],
        },
    ]

    gee_dfs = {}

    for gee_index, gee_dataset in enumerate(gee_datasets):

        logger.info(
            f"Collecting GEE data ({gee_index+1} / {len(gee_datasets)}): {gee_dataset}"
        )

        collection_id = gee_dataset["collection_id"]
        bands = gee_dataset["bands"]
        preprocessors = gee_dataset["preprocessors"]

        # For recording all dfs before concatenating later on
        all_dfs = []

        # Iterate through stations
        for index, station in tqdm(
            th_stations_df.iterrows(), total=len(th_stations_df)
        ):

            # Generate station data
            station_gee_values_df = gee_utils.generate_aoi_tile_data(
                collection_id,
                start_date,
                end_date,
                station.latitude,
                station.longitude,
                bands=bands,
                cloud_filter=False,
            )
            # Set the station code
            station_gee_values_df["station_code"] = station.station_code

            # Pre-process
            for preprocessor in preprocessors:
                station_gee_values_df = preprocessor(station_gee_values_df)

            # Add to main df
            all_dfs.append(station_gee_values_df)

        # TODO: pre-process the DF
        gee_dfs[collection_id] = pd.concat(all_dfs, axis=0, ignore_index=True)

    # TODO: Temporary log to check results
    for collection, df in gee_dfs.items():
        logger.debug(f"{collection}: {len(df)} rows")
        debug_dir = settings.DATA_DIR / "debug"
        os.makedirs(debug_dir, exist_ok=True)
        collection_name_sanitized = collection.replace("/", "_")
        df.to_csv(debug_dir / f"{collection_name_sanitized}.csv", index=False)

    # Create reference table
    # Start with the daily pm2.5 DF
    base_table = pm25_df.copy()

    # Merge all tables
    # Station info
    base_table = base_table.merge(th_stations_df, on=["station_code"], how="left")

    # TODO: Population
    for collection, df in gee_dfs.items():
        base_table = base_table.merge(df, on=["date", "station_code"], how="left")

    # Sorting
    base_table = base_table.sort_values(by=["station_code", "date"])
    base_table.to_csv(settings.DATA_DIR / "training_dataset.csv", index=False)
    logger.info(f"Generated base table for ML modelling with {len(base_table)} rows")


if __name__ == "__main__":

    # Authenticate with GEE
    load_dotenv()
    credentials, project = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/devstorage.full_control",
        ]
    )
    ee.Initialize(credentials)

    main()
