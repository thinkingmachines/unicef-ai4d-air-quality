def aggregate_daily_aod(df, params):

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


def rescale_cams_aod(df, params):
    # This is a normalizaton step to scale the data same as MAIAC AOD
    df["total_aerosol_optical_depth_at_550nm_surface"] = df[
        "total_aerosol_optical_depth_at_550nm_surface"
    ].apply(lambda x: x * 1000)

    return df


def aggregate_daily_cams_aod(df, params):
    # Add date column
    df["date"] = df["time"].dt.date
    id_col = params["id_col"]

    # Aggregate by date and station. For each band, get mean, min, max, and median
    df = df.groupby(["date", id_col], as_index=False, group_keys=False).agg(
        # CAMS_AOD_047_mean=("total_aerosol_optical_depth_at_469nm_surface_mean", "mean"),
        # CAMS_AOD_047_min=("total_aerosol_optical_depth_at_469nm_surface_min", "min"),
        # CAMS_AOD_047_max=("total_aerosol_optical_depth_at_469nm_surface_max", "max"),
        # CAMS_AOD_047_median=(
        #     "total_aerosol_optical_depth_at_469nm_surface_median",
        #     "median",
        # ),
        CAMS_AOD_055_mean=("total_aerosol_optical_depth_at_550nm_surface", "mean"),
        CAMS_AOD_055_min=("total_aerosol_optical_depth_at_550nm_surface", "min"),
        CAMS_AOD_055_max=("total_aerosol_optical_depth_at_550nm_surface", "max"),
        CAMS_AOD_055_median=(
            "total_aerosol_optical_depth_at_550nm_surface",
            "median",
        ),
    )

    return df


def aggregate_daily_s5p_aerosol(df, params):
    # Add date column
    df["date"] = df["time"].dt.date
    id_col = params["id_col"]

    # Aggregate by date and station. For each band, get mean, min, max, and median
    df = df.groupby(["date", id_col], as_index=False, group_keys=False).agg(
        AAI_mean=("absorbing_aerosol_index", "mean"),
        AAI_min=("absorbing_aerosol_index", "min"),
        AAI_max=("absorbing_aerosol_index", "max"),
        AAI_median=(
            "absorbing_aerosol_index",
            "median",
        ),
    )

    return df
