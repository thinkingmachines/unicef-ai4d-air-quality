import os
from datetime import datetime

import click
import geopandas as gpd
import pandas as pd
from loguru import logger
from rasterstats import zonal_stats
from shapely import wkt
from tqdm.auto import tqdm

from src.config import settings
from src.data_processing import aod, era5, gee_utils, geom_utils, ndvi


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


def collect_gee_datasets(gee_datasets, start_date, end_date, locations_df, id_col="id"):
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
            for preprocessor, params in preprocessors:
                station_gee_values_df = preprocessor(station_gee_values_df, params)

            # Add to main df
            all_dfs.append(station_gee_values_df)

        gee_dfs[collection_id] = pd.concat(all_dfs, axis=0, ignore_index=True)

    return gee_dfs


def join_datasets(
    locations_df,
    start_date,
    end_date,
    gee_dfs,
    hrsl_df,
    id_col,
    date_col="date",
    ground_truth_df=None,
):
    # TODO: Population

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

    # Ground Truth
    if ground_truth_df is not None:
        base_df = base_df.merge(ground_truth_df, on=[id_col, date_col], how="left")

    # Sorting
    base_df = base_df.sort_values(by=[id_col, date_col])

    return base_df


def load_ground_truth(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def generate_bboxes(
    locations_df,
    bbox_size_km,
    geometry_col="geometry",
):
    locations_df = locations_df.copy()
    locations_df[geometry_col] = locations_df.apply(
        lambda row: geom_utils.generate_bbox_wkt(
            row["latitude"], row["longitude"], distance_km=bbox_size_km
        ),
        axis=1,
    )
    return locations_df


def collect_hrsl(locations_df, hrsl_tif, id_col, bbox_size_km):
    locations_df = generate_bboxes(locations_df, bbox_size_km=bbox_size_km)
    # locations_df.to_csv(settings.DATA_DIR / "debug_locations.csv")
    hrsl_gdf = gpd.GeoDataFrame(
        locations_df, geometry=locations_df["geometry"].apply(wkt.loads)
    )
    hrsl_gdf["total_population"] = pd.DataFrame(
        zonal_stats(vectors=hrsl_gdf["geometry"], raster=hrsl_tif, stats="sum")
    )["sum"]

    # Retain only id_col and total_population to save on space.
    hrsl_df = pd.DataFrame(hrsl_gdf.drop(columns="geometry"))[
        [id_col, "total_population"]
    ]
    return hrsl_df


@click.command()
@click.option(
    "--locations-csv",
    default=settings.DATA_DIR / "air4thai_th_stations.csv",
    help="Path to the CSV file containing the locations for which to generate data.",
)
@click.option(
    "--ground-truth-csv",
    help="Path to the CSV file containing the locations for which to generate data.",
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
    locations_csv, ground_truth_csv, hrsl_tif, id_col, start_date, end_date, debug
):
    BBOX_SIZE_KM = 1

    # Read in desired AOI locations
    # Assumed that the CSV has an id column, latitude, and longitude at the minimum.
    locations_df = pd.read_csv(locations_csv)
    if debug:
        logger.warning("Running in debug mode. Trying out with 2 locations only.")
        locations_df = locations_df[:2]
    assert {id_col, "latitude", "longitude"} <= set(locations_df.columns.tolist())

    # Read in ground truth if any
    if ground_truth_csv:
        ground_truth_df = load_ground_truth(ground_truth_csv)
        logger.info(f"Generating dataset with ground truth from {ground_truth_csv}")
    else:
        ground_truth_df = None
        logger.warning("Generating dataset without ground truth.")

    # Compute HRSL stats
    logger.info("Computing population sums...")
    hrsl_df = collect_hrsl(
        locations_df, hrsl_tif, id_col=id_col, bbox_size_km=BBOX_SIZE_KM
    )

    # Collect GEE Datasets
    gee_datasets = [
        {
            "collection_id": "COPERNICUS/S5P/OFFL/L3_AER_AI",
            "bands": [
                "absorbing_aerosol_index",
            ],
            "preprocessors": [(aod.aggregate_daily_s5p_aerosol, {})],
        },
        {
            "collection_id": "ECMWF/CAMS/NRT",
            "bands": [
                # "total_aerosol_optical_depth_at_469nm_surface", # Results in errors cause it can be missing sometimes
                "total_aerosol_optical_depth_at_550nm_surface",
            ],
            "preprocessors": [
                (aod.rescale_cams_aod, {}),
                (aod.aggregate_daily_cams_aod, {}),
            ],
        },
        {
            "collection_id": "MODIS/006/MCD19A2_GRANULES",  # Aerosol Optical Depth (AOD)
            "bands": ["Optical_Depth_047", "Optical_Depth_055"],
            "preprocessors": [(aod.aggregate_daily_aod, {})],
        },
        {
            "collection_id": "MODIS/006/MOD13A2",  # Vegetation
            "bands": ["NDVI", "EVI"],
            "preprocessors": [
                (
                    ndvi.aggregate_daily_ndvi,
                    {"start_date": start_date, "end_date": end_date},
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
            "preprocessors": [(era5.aggregate_daily_era5, {})],
        },
    ]

    gee_utils.gee_auth(service_acct=True)
    gee_dfs = collect_gee_datasets(
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
    base_df = join_datasets(
        locations_df,
        start_date,
        end_date,
        gee_dfs,
        hrsl_df,
        id_col,
        ground_truth_df=ground_truth_df,
    )
    out_filepath = f"generated_data_{run_timestamp}.csv"
    base_df.to_csv(settings.DATA_DIR / out_filepath, index=False)
    logger.info(
        f"Generated base table for ML modelling with {len(base_df)} rows. Saved to {out_filepath}"
    )


if __name__ == "__main__":
    main()
