import pandas as pd


def aggregate_daily_ndvi(ndvi_df):

    # Add date column
    ndvi_df["date"] = ndvi_df["time"].dt.date

    # Generate date list
    # TODO parametrize the start/end dates
    date_list = pd.DataFrame(
        pd.date_range("2021-01-01", "2021-12-31", freq="D").date, columns=["date"]
    )

    # Get list of stations
    station_list = ndvi_df[["station_code"]].drop_duplicates()

    # Create NDVI table template
    ndvi_canvas = pd.DataFrame()
    for idx, station in station_list.iterrows():
        temp_df = date_list
        temp_df["station_code"] = station.station_code
        ndvi_canvas = pd.concat([ndvi_canvas, temp_df])

    # Merge and forward fill to get values for dates that don't have NDVI readings
    ndvi_filled = ndvi_df.merge(
        ndvi_canvas, on=["date", "station_code"], how="right"
    ).sort_values(["station_code", "date"])
    ndvi_filled = ndvi_filled.fillna(method="ffill")

    return ndvi_filled
