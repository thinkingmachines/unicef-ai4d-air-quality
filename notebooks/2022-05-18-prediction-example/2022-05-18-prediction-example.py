# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3.7.13 ('air-quality')
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook is for demonstrating how to use a trained PM2.5 prediction model to make predictions on a target area.

# %% [markdown]
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

import re
import sys

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

sys.path.append("../../")

from src.config import settings
from src.data_processing import feature_collection_pipeline

# %% [markdown]
# # Parameters

# %%

# CSV file for target roll-out location
# CENTROIDS_PATH = "data/mueang_chiang_mai_tile_centroids.csv"
CENTROIDS_PATH = "data/mueang_phuket_tile_centroids.csv"


# ID Column in the target CSV file
ID_COL = "id"

# Desired date range
START_DATE = "2019-03-01"
END_DATE = "2019-03-31"
# Model being used was trained using 1km x 1km bounding box
BBOX_SIZE_KM = 1

# Path to the model to be used for prediction
MODEL_PATH = settings.DATA_DIR / "latest_model.pkl"

# Path to the population raster (In our study, we're using the Thai 2020 HRSL raster)
HRSL_TIF = settings.DATA_DIR / "tha_general_2020.tif"

# Where to save the model predictions of  daily PM2.5 levels
OUTPUT_PREDICTIONS_FILE = "data/phuket_march2019_predictions.csv"

# What to name the column for predictions
PRED_COL = "predicted_pm2.5"


# %% [markdown]
# # Load Area of Interest

# %%
def extract_lon_lat(wkt_string_point):
    regex = r"[0-9-\.]+"
    parsed_geom = re.findall(regex, wkt_string_point)
    parsed_geom = [float(i) for i in parsed_geom]
    assert len(parsed_geom) == 2
    return parsed_geom[0], parsed_geom[1]


def load_centroids(centroids_path):
    locations_gdf = gpd.read_file(centroids_path)
    all_lon_lat = locations_gdf["WKT"].apply(lambda x: extract_lon_lat(x)).tolist()
    all_lon, all_lat = zip(*all_lon_lat)

    locations_gdf["longitude"] = all_lon
    locations_gdf["latitude"] = all_lat

    locations_df = pd.DataFrame(locations_gdf)
    locations_df.drop(["geometry", "WKT"], axis=1, inplace=True)

    return locations_df


# %%
locations_df = load_centroids(CENTROIDS_PATH)[:]
len(locations_df)

# %%
locations_df.columns

# %% [markdown]
# # Predict

# %% [markdown]
# ## Collect Features

# %%
# Create base DF from the locations (collect the necessary features)
base_df = feature_collection_pipeline.collect_features_for_locations(
    locations_df=locations_df,
    start_date=START_DATE,
    end_date=END_DATE,
    id_col=ID_COL,
    hrsl_tif=HRSL_TIF,
    bbox_size_km=BBOX_SIZE_KM,
    # Customized the list of GEE datasets because the latest model doesn't use MAIAC
    gee_datasets=[
        feature_collection_pipeline.S5P_AAI_CONFIG,
        feature_collection_pipeline.CAMS_AOD_CONFIG,
        feature_collection_pipeline.NDVI_CONFIG,
        feature_collection_pipeline.ERA5_CONFIG,
    ],
)


# %%
# Load Model
model = joblib.load(MODEL_PATH)

# Filter to only the relevant columns
keep_cols = model.feature_names  # This was saved from the train script
ml_df = base_df[keep_cols]

# Run model
preds = model.predict(ml_df)
base_df[PRED_COL] = preds
base_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)


# %% [markdown]
# # Quick checks on predictions
#
# We print out quick summary stats and plot the histogram of predicted pm2.5 values for the district of Chiang Mai and Phuket.
#
# It was known that Chiang Mai in March 2019 had poor air quality due to burning season.
#
# On the other hand, we expect the air at Phuket to be cleaner since it is far away from agricultural lands that are burned.

# %%
# Check predictions for Chiang Mai in March 2019

results_df = pd.read_csv("data/chiang_mai_march2019_predictions.csv")
results_df = gpd.GeoDataFrame(
    results_df, geometry=gpd.points_from_xy(results_df.longitude, results_df.latitude)
)
preds = results_df[PRED_COL]
print(f"Min:{min(preds):.2f} - Max:{max(preds):.2f}; Mean={np.mean(preds):.2f}")
results_df[PRED_COL].hist()

# %%
# Check predictions for Phuket in March 2019

results_df = pd.read_csv("data/phuket_march2019_predictions.csv")
results_df = gpd.GeoDataFrame(
    results_df, geometry=gpd.points_from_xy(results_df.longitude, results_df.latitude)
)
preds = results_df[PRED_COL]
print(f"Min:{min(preds):.2f} - Max:{max(preds):.2f}; Mean={np.mean(preds):.2f}")
results_df[PRED_COL].hist()

# %%
