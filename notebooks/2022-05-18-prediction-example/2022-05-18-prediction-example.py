# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook is for demonstrating how to use a trained PM2.5 prediction model to make predictions on a target area.
# There is also a script version for just generating predictoins in `scripts/predict.py`. Both utilize the same underlying prediction logic.
#
# The input is a CSV/Dataframe of locations, represented by coordinates. In our sample notebook, our goal is to predict daily PM2.5 levels for the Mueang Chiang Mai admin level 2 district for the year 2021.
#
# To visualize the input:
#
# ![Mueang Chiang Mai Tile Centroids](img/chiangmai_centroids.png)
#
# (*This was generated with QGIS and exported to a CSV file*)
#
#

# %% [markdown]
# # Imports

# %%
# %load_ext autoreload
# %autoreload 2

import re
import sys

import branca
import folium
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely import wkt

sys.path.append("../../")

from src.config import settings
from src.data_processing import feature_collection_pipeline, geom_utils
from src.prediction import predict_utils

# %% [markdown]
# # Parameters

# %%
DEBUG = True

# CSV file for target roll-out location
CENTROIDS_PATH = "data/mueang_chiang_mai_tile_centroids.csv"

# Where to save the model predictions of  daily PM2.5 levels
OUTPUT_PREDICTIONS_FILE = (
    "data/chiangmai_2021_predictions.csv" if not DEBUG else "data/debug.csv"
)

# Desired date range
START_DATE = "2021-01-01"
END_DATE = "2021-12-31"

# ID Column in the target CSV file
ID_COL = "id"

# What to name the column for predictions
PRED_COL = "predicted_pm2.5"

# Model being used was trained using 1km x 1km bounding box
BBOX_SIZE_KM = 1

# Path to the model to be used for prediction
MODEL_PATH = settings.DATA_DIR / "latest_model.pkl"

# Path to the population raster (In our study, we're using the Thai 2020 HRSL raster)
HRSL_TIF = settings.DATA_DIR / "tha_general_2020.tif"


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
locations_df = load_centroids(CENTROIDS_PATH)

if DEBUG:
    locations_df = locations_df[:2]

len(locations_df)

# %%
locations_df.columns

# %% [markdown]
# # Predict

# %%
results_df = predict_utils.predict(
    locations_df,
    start_date=START_DATE,
    end_date=END_DATE,
    id_col=ID_COL,
    hrsl_tif=HRSL_TIF,
    model_path=MODEL_PATH,
    bbox_size_km=BBOX_SIZE_KM,
    pred_col=PRED_COL,
)

# %%
results_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)


# %% [markdown]
# # EDA on Predictions

# %% [markdown] tags=[]
# ## Utility Functions

# %%
def load_results(results_path, pred_col=PRED_COL):
    df = pd.read_csv(results_path)
    # Post-processing: clip negative values to 0
    df[pred_col] = df[pred_col].apply(lambda x: x if x > 0 else 0)
    return df


# %%
def aggregate_preds(
    results_path, start_date=None, end_date=None, id_col=ID_COL, pred_col=PRED_COL
):
    # Initialize DF
    df = load_results(results_path)
    if start_date and end_date:
        # Filter according to dates
        date_mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[date_mask]

    # Initialize GDF
    df = geom_utils.generate_bboxes(
        df, bbox_size_km=BBOX_SIZE_KM, geometry_col="geometry"
    )
    gdf = gpd.GeoDataFrame(df, geometry=df["geometry"].apply(lambda x: wkt.loads(x)))

    # Get the average predicted value per tile
    gdf = gdf.dissolve(by=[ID_COL], aggfunc="mean")

    return gdf


# %%
def categorize_value(value):
    if value >= 250.5:
        return "hazardous"
    if value >= 150.5:
        return "very_unhealthy"
    if value >= 55.5:
        return "unhealthy"
    if value >= 35.5:
        return "unhealthy_for_sensitive_groups"
    if value >= 12.1:
        return "moderate"
    return "good"


def generate_pm25_cmap(min=0, max=500.0):
    lower_bounds = [0.0, 12.1, 35.5, 55.5, 150.5, 250.5]
    colors = ["green", "yellow", "orange", "red", "purple", "brown"]
    step = branca.colormap.StepColormap(
        colors, vmin=min, vmax=max, index=lower_bounds, caption="Predicted PM2.5 Levels"
    )
    return step


# %%
def viz_preds(gdf, tooltip=True, pred_col=PRED_COL):
    gdf_centroid = gdf.dissolve().geometry.centroid.values[0]

    # Categorize for the tooltip
    gdf["category"] = gdf[pred_col].apply(lambda x: categorize_value(x))

    m = folium.Map(
        location=[gdf_centroid.y, gdf_centroid.x], width=800, height=800, zoom_start=12
    )
    style_kwds = {"opacity": 0.3}

    gdf.explore(
        pred_col, m=m, style_kwds=style_kwds, tooltip=tooltip, cmap=generate_pm25_cmap()
    )
    return m


# %% [markdown]
# ## Results File to perform EDA on

# %%
results_path = "data/predictions_chiangmai_2021.csv"

# %% [markdown]
# ## Annual PM2.5 Mean

# %%
df = load_results(results_path)
df[PRED_COL].mean()

# %% [markdown]
# ## PM2.5 Levels per Month

# %%
df = load_results(results_path)
df["month"] = pd.to_datetime(df["date"]).dt.month
df["year"] = pd.to_datetime(df["date"]).dt.year
summary = df[["month", "year", PRED_COL]].groupby(["month", "year"]).describe()
summary.columns = summary.columns.droplevel()

summary

# %%
summary.reset_index().plot.bar(x="month", y="mean", rot=0)

# %% [markdown]
# Agricultural burning season in Thailand varies from place to place, but is generally in the earlier parts of the year.
#
# This [article by iqair](https://www.iqair.com/blog/air-quality/thailand-2021-burning-season) conducted an analysis and considered Jan-March as the general burning season. Our model's predictions for Chiang Mai seem to conform to this.
#
# Furthermore, the trend seems to be consistent with the monthly mean PM2.5 levels reported by IQAir where the start of the year sees the highest levels of PM2.5, which lowers starting May, then increases again towards the end of the year.
#
# ![Image taken from: https://www.iqair.com/blog/air-quality/thailand-2021-burning-season](img/chiangmai_monthly_averages_iqair.png)

# %% [markdown]
# ## Visualize Certain Months

# %%
# Infer the columns used by the model for simplicity in the viz
model = joblib.load(MODEL_PATH)
keep_cols = [
    ID_COL,
    PRED_COL,
] + model.feature_names  # This was saved from the train script

# %%
# March
gdf = aggregate_preds(results_path, start_date="2021-03-01", end_date="2021-03-31")
viz_preds(gdf, tooltip=keep_cols)

# %%
# June
gdf = aggregate_preds(results_path, start_date="2021-06-01", end_date="2021-06-30")
viz_preds(gdf, tooltip=keep_cols)

# %%
# December
gdf = aggregate_preds(results_path, start_date="2021-12-01", end_date="2021-12-31")
viz_preds(gdf, tooltip=keep_cols)

# %%
