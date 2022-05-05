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
# The purpose of this notebook is to add Thailand admin boundaries to an already existing dataset (CSV file). We will integrate this logic within the data generation script though, to avoid having to run multiple scripts/notebooks.

# %% [markdown]
# # Imports

# %%
import sys

sys.path.append("../")

import geopandas as gpd
import pandas as pd

from src.config import settings

# %% [markdown]
# # TH Admin Bounds
#
# This admin bounds file is taken from the [Humanitarian Data Exchange](https://data.humdata.org/dataset/cod-ab-tha) website.
#
# Here we're using the most granular data available, which are the admin level 3 boundaries. But the Shapefile also contains info all the way from admin level 0 to 3.

# %%
# The file has to be placed in the data folder in the project root.
TH_ADM3_SHP = (
    settings.DATA_DIR / "th_boundaries/adm3/tha_admbnda_adm3_rtsd_20220121.shp"
)

th_adm3_gdf = gpd.read_file(TH_ADM3_SHP)
th_adm3_gdf.head()

# %% [markdown]
# # Identify the admin bounds for stations
#
# In this notebook, we're identifying the admin bounds for an already existing dataset we're using for ML modelling.

# %%
# Simiarly, this file is in the data folder of the project root. Feel free to change this accordingly.
STATIONS_CSV = settings.DATA_DIR / "2022-04-29-base-table-air4thai.csv"

stations_df = pd.read_csv(STATIONS_CSV)
stations_df.head()

# %% [markdown]
# Before doing a spatial join with the admin bounds, we have to convert the stations DataFrame into a GeoDataFrame.

# %%
stations_gdf = gpd.GeoDataFrame(
    stations_df,
    geometry=gpd.points_from_xy(stations_df["longitude"], stations_df["latitude"]),
    crs="EPSG:4326",
)

stations_gdf.head()

# %% [markdown]
# We then perform the spatial join. For each station, we're interested in finding which admin bounds they fall under. This line of code finds where the station's coordinates are "within" which admin boundary.

# %%
joined_gdf = gpd.sjoin(stations_gdf, th_adm3_gdf, predicate="within")

joined_gdf.head()

# %% [markdown]
# Finally, we save the output to a CSV file.

# %%
joined_gdf.to_csv(
    settings.DATA_DIR / "2022-05-04-base-table-air4thai-with-adm3.csv", index=False
)
