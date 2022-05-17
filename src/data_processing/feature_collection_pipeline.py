import os
from datetime import datetime

import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from src.config import settings
from src.data_processing import hrsl
from src.data_processing.gee import aod, era5, gee_utils, ndvi

S5P_AAI_CONFIG = {
    "collection_id": "COPERNICUS/S5P/OFFL/L3_AER_AI",
    "bands": [
        "absorbing_aerosol_index",
    ],
    "preprocessors": [aod.aggregate_daily_s5p_aerosol],
}


CAMS_AOD_CONFIG = {
    "collection_id": "ECMWF/CAMS/NRT",
    "bands": [
        # "total_aerosol_optical_depth_at_469nm_surface", # Results in errors cause it can be missing sometimes
        "total_aerosol_optical_depth_at_550nm_surface",
    ],
    "preprocessors": [
        aod.rescale_cams_aod,
        aod.aggregate_daily_cams_aod,
    ],
}


MAIAC_AOD_CONFIG = {
    "collection_id": "MODIS/006/MCD19A2_GRANULES",  # Aerosol Optical Depth (AOD)
    "bands": ["Optical_Depth_047", "Optical_Depth_055"],
    "preprocessors": [aod.aggregate_daily_aod],
}


NDVI_CONFIG = {
    "collection_id": "MODIS/006/MOD13A2",  # Vegetation
    "bands": ["NDVI", "EVI"],
    "preprocessors": [ndvi.aggregate_daily_ndvi],
}

ERA5_CONFIG = {
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
}


def collect_features_for_locations(
    locations_df,
    start_date,
    end_date,
    id_col,
    hrsl_tif,
    date_col="date",
    bbox_size_km=1,
    log_gee_dfs=False,
    log_key=None,
    log_dir=settings.DATA_DIR / "debug",
    gee_datasets=[
        S5P_AAI_CONFIG,
        CAMS_AOD_CONFIG,
        MAIAC_AOD_CONFIG,
        NDVI_CONFIG,
        ERA5_CONFIG,
    ],
):
    # Auth with GEE
    gee_utils.gee_auth()

    # Compute HRSL stats
    logger.info("Computing population sums...")
    hrsl_df = hrsl.collect_hrsl(
        locations_df, hrsl_tif, id_col=id_col, bbox_size_km=bbox_size_km
    )

    # Collect GEE Datasets
    logger.info("Collecting GEE datasets...")
    gee_dfs = collect_gee_datasets(
        gee_datasets, start_date, end_date, locations_df, id_col=id_col
    )

    if log_gee_dfs:
        log_key = log_key if log_key else datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

        # Debug logs to check intermediate files
        for collection, df in gee_dfs.items():
            logger.debug(f"{collection}: {len(df)} rows")
            os.makedirs(log_dir, exist_ok=True)
            collection_name_sanitized = collection.replace("/", "_")
            df.to_csv(
                log_dir / f"{collection_name_sanitized}_{log_key}.csv",
                index=False,
            )

    # Create DF with locations + start_date, end_date
    base_df = generate_locations_with_dates_df(
        locations_df, start_date, end_date, id_col=id_col, date_col=date_col
    )

    # Merge HRSL
    # HRSL is a slow-moving feature, and so does not change depending on the date.
    base_df = base_df.merge(hrsl_df, on=[id_col], how="left")

    # Merge GEE dfs
    for _, gee_df in gee_dfs.items():
        base_df = base_df.merge(gee_df, on=[id_col, date_col], how="left")

    # Sort for easier eyebell checking
    base_df = base_df.sort_values(by=[id_col, date_col])

    return base_df


def generate_locations_with_dates_df(
    df, start_date, end_date, id_col="id", date_col="date"
):
    # We create a dummy date column just so we can use the ffill technique to construct one row for each date per station.
    df = df.copy()
    df[date_col] = pd.to_datetime(start_date, format="%Y-%m-%d")
    df = (
        df.groupby([id_col])
        .apply(
            lambda x: x.set_index(date_col)
            .reindex(pd.date_range(start=start_date, end=end_date))
            .ffill()
            .rename_axis(date_col)
            .reset_index()
        )
        .droplevel(id_col)
    )
    df[date_col] = df[date_col].dt.date
    return df


def collect_gee_datasets(gee_datasets, start_date, end_date, locations_df, id_col):
    gee_dfs = {}
    for gee_index, gee_dataset in enumerate(gee_datasets):

        logger.info(
            f"Collecting GEE data ({gee_index+1} / {len(gee_datasets)}): {gee_dataset}"
        )

        print(gee_dataset)

        collection_id = gee_dataset["collection_id"]
        bands = gee_dataset["bands"]
        preprocessors = gee_dataset["preprocessors"]

        # For recording all dfs before concatenating later on
        all_dfs = []

        # Iterate through stations
        for index, location in tqdm(locations_df.iterrows(), total=len(locations_df)):
            # Generate station data
            station_gee_values_df = gee_utils.generate_aoi_tile_data(
                collection_id,
                start_date,
                end_date,
                location.latitude,
                location.longitude,
                bands=bands,
                cloud_filter=False,
            )
            # Set the ID so we can join back the data later on
            station_gee_values_df[id_col] = location[id_col]

            # Pre-process
            params = {"start_date": start_date, "end_date": end_date, "id_col": id_col}
            for preprocessor in preprocessors:
                station_gee_values_df = preprocessor(station_gee_values_df, params)

            # Add to main df
            all_dfs.append(station_gee_values_df)

        gee_dfs[collection_id] = pd.concat(all_dfs, axis=0, ignore_index=True)

    return gee_dfs
