def aggregate_daily_era5(df):

    # Add date column
    df["date"] = df["time"].dt.date

    # Aggregate by date and station
    df = df.groupby(["date", "station_code"], as_index=False, group_keys=False).agg(
        dewpoint_temperature_2m_mean=("dewpoint_temperature_2m", "mean"),
        dewpoint_temperature_2m_min=("dewpoint_temperature_2m", "min"),
        dewpoint_temperature_2m_median=("dewpoint_temperature_2m", "median"),
        dewpoint_temperature_2m_max=("dewpoint_temperature_2m", "max"),
        temperature_2m_mean=("temperature_2m", "mean"),
        temperature_2m_min=("temperature_2m", "min"),
        temperature_2m_median=("temperature_2m", "median"),
        temperature_2m_max=("temperature_2m", "max"),
        u_component_of_wind_10m_mean=("u_component_of_wind_10m", "mean"),
        u_component_of_wind_10m_min=("u_component_of_wind_10m", "min"),
        u_component_of_wind_10m_median=("u_component_of_wind_10m", "median"),
        u_component_of_wind_10m_max=("u_component_of_wind_10m", "max"),
        v_component_of_wind_10m_mean=("v_component_of_wind_10m", "mean"),
        v_component_of_wind_10m_min=("v_component_of_wind_10m", "min"),
        v_component_of_wind_10m_median=("v_component_of_wind_10m", "median"),
        v_component_of_wind_10m_max=("v_component_of_wind_10m", "max"),
        surface_pressure_mean=("surface_pressure", "mean"),
        surface_pressure_min=("surface_pressure", "min"),
        surface_pressure_median=("surface_pressure", "median"),
        surface_pressure_max=("surface_pressure", "max"),
        total_precipitation_daily=("total_precipitation_hourly", "sum"),
        mean_precipitation_hourly=("total_precipitation_hourly", "mean"),
    )

    return df
