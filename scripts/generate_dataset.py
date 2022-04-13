import os

import click
import ee
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from src.config import settings
from src.data_processing import gee_utils


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
            "preprocessors": [],
        },
        {
            "collection_id": "MODIS/006/MOD13A2",  # Vegetation
            "bands": ["NDVI", "EVI"],
            "preprocessors": [],
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
            "preprocessors": [],
        },
    ]

    collection_dfs = {}

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
            # logger.debug(
            #     f"Collecting {collection_id} data for station {index+1} / {len(th_stations_df)}"
            # )

            # Generate station data
            station_gee_values_df = gee_utils.generate_station_data(
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
            # Add to main df
            all_dfs.append(station_gee_values_df)

        # TODO: pre-process the DF
        collection_dfs[collection_id] = pd.concat(all_dfs, axis=0)

    # TODO: Temporary log to check results
    for collection, df in collection_dfs.items():
        logger.debug(f"{collection}: {len(df)} rows")


if __name__ == "__main__":

    # Authenticate with GEE
    load_dotenv()
    service_account = os.environ["SERVICE_ACCOUNT"]
    service_account_key = os.environ["SERVICE_ACCOUNT_KEY"]
    credentials = ee.ServiceAccountCredentials(service_account, service_account_key)
    ee.Initialize(credentials)

    main()
