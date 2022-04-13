def aggregate_daily_aod(df):

    # Add date column
    df["date"] = df["time"].dt.date

    # Aggregate by date and station. For each band, get mean, min, max, and median
    df = df.groupby(["date", "station_code"], as_index=False, group_keys=False).agg(
        AOD_047_mean=("Optical_Depth_047", "mean"),
        AOD_047_min=("Optical_Depth_047", "min"),
        AOD_047_max=("Optical_Depth_047", "max"),
        AOD_047_median=("Optical_Depth_047", "median"),
        AOD_055_mean=("Optical_Depth_055", "mean"),
        AOD_055_min=("Optical_Depth_055", "min"),
        AOD_055_max=("Optical_Depth_055", "max"),
        AOD_055_median=("Optical_Depth_055", "median"),
    )

    return df
