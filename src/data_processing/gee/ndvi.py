import pandas as pd


def aggregate_daily_ndvi(ndvi_df, params):

    start_date = params["start_date"]
    end_date = params["end_date"]
    id_col = params["id_col"]

    # Add date column
    ndvi_df["date"] = ndvi_df["time"].dt.date

    # Generate date list
    date_list = pd.DataFrame(
        pd.date_range(start_date, end_date, freq="D").date, columns=["date"]
    )

    # Get list of stations
    station_list = ndvi_df[[id_col]].drop_duplicates()

    # Create NDVI table template
    ndvi_canvas = pd.DataFrame()
    for idx, station in station_list.iterrows():
        temp_df = date_list
        temp_df[id_col] = station[id_col]
        ndvi_canvas = pd.concat([ndvi_canvas, temp_df])

    # NDVI DF from GEE could yield multiple values per date, so aggregate
    ndvi_df = ndvi_df.groupby(["date", id_col], as_index=False, group_keys=False).agg(
        NDVI_mean=("NDVI", "mean"),
        NDVI_min=("NDVI", "min"),
        NDVI_max=("NDVI", "max"),
        NDVI_median=("NDVI", "median"),
        EVI_mean=("EVI", "mean"),
        EVI_min=("EVI", "min"),
        EVI_max=("EVI", "max"),
        EVI_median=("EVI", "median"),
    )

    # Merge and forward fill to get values for dates that don't have NDVI readings
    ndvi_filled = ndvi_df.merge(
        ndvi_canvas, on=["date", id_col], how="right"
    ).sort_values([id_col, "date"])
    ndvi_filled = ndvi_filled.fillna(method="ffill")

    # Select only relevant columns
    return ndvi_filled
