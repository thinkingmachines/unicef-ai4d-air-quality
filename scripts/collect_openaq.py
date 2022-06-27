from datetime import datetime

import click
import pandas as pd

from src.data_collection import openaq


def preprocess_df(df):
    # Rename some columns first for better formatting and consistency.
    df = df.rename(
        columns={
            "locationId": "station_code",
            "sensorType": "sensor_type",
            "coordinates.latitude": "latitude",
            "coordinates.longitude": "longitude",
            "pm25_mean": "pm2.5",
        }
    )

    # Generate daily pm2.5 df (ground truth df)
    ground_truth_df = df.copy()
    ground_truth_df["date"] = pd.to_datetime(ground_truth_df["date.utc"]).dt.date
    ground_truth_df = ground_truth_df.groupby(
        ["date", "station_code"], as_index=False, group_keys=False
    ).agg(pm25_mean=("value", "mean"))

    # Extract list of unique stations
    station_list_df = df.copy()
    station_list_df = station_list_df.drop_duplicates(subset=["station_code"])[
        ["station_code", "location", "city", "sensor_type", "latitude", "longitude"]
    ].reset_index(drop=True)

    return ground_truth_df, station_list_df


@click.command()
@click.option(
    "--country-code",
    default="TH",
    help="2-letter country code",
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
def main(country_code, start_date, end_date):
    df = openaq.get_openaq_measurements(country_code, start_date, end_date)

    run_timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    # Save to CSV
    raw_path = f"raw_data_{run_timestamp}.csv"

    daily_pm25_path = f"daily-pm25-{run_timestamp}.csv"
    station_list_path = f"station-list-{run_timestamp}.csv"

    df.to_csv(raw_path, index=False)
    print(f"Raw data saved to {raw_path}")

    ground_truth_df, station_list_df = preprocess_df(df)
    ground_truth_df.to_csv(daily_pm25_path, index=False)
    station_list_df.to_csv(station_list_path, index=False)

    print(f"Daily PM2.5 Ground Truth saved to {daily_pm25_path}")
    print(f"Station list saved to {station_list_path}")


if __name__ == "__main__":
    main()
