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
# # OpenAQ Thailand PM 2.5 Data Collection
#
# import json
# import time
#
# import numpy as np

# %%
import pandas as pd
import requests

# %%
# Set server URL as global variable.
server_url = "https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2"  # working as of May 27, 2022


# %% [markdown]
# ## Functions

# %%
def construct_url(server_url, method, parameters):

    url_params = ""

    # Construct URL string for all parameters

    for index, value in enumerate(parameters):

        # This if clause just catches the last parameter (shouldn't have a succeeding &)
        if index == len(parameters) - 1:
            param = value + "=" + str(parameters[value])
        else:
            param = value + "=" + str(parameters[value]) + "&"

        # Append to URL parameters string
        url_params = url_params + param

    url = server_url + "/" + method + "?" + url_params
    url = url.replace(" ", "%20")

    return url


# %%
def openaq_get(url):

    # API call
    response = requests.get(url)

    # Get results specifically
    data = response.json()["results"]

    # Convert json output to DataFrame
    df = pd.json_normalize(data)

    return df


# %%
def get_measurements(
    country_id,
    start_date,
    end_date,
    parameter="pm25",
    sensor_type=None,
    limit=10000,
    include_mobile=False,
):

    method = "measurements"

    parameters = {
        "date_from": start_date,
        "date_to": end_date,
        "has_geo": True,
        "parameter": parameter,
        "country_id": country_id,
        "limit": limit,
        "isMobile": include_mobile,
    }

    # If sensor type is explicitly indicated
    if sensor_type:
        parameters["sensorType"] = sensor_type

    url = construct_url(server_url, method, parameters)
    df = openaq_get(url)

    return df


# %% [markdown]
# ## Data Collection

# %%
# Set date range
date_range = pd.date_range("2021-01-01", "2021-12-31")

# Create base table
openaq_df = pd.DataFrame()

# Iterate through date range
for d in date_range:

    start_date = d.date()
    end_date = pd.to_datetime(d) + pd.Timedelta(days=1)

    # Indicator for when data collection for a new month starts
    if d.day == 1:
        print(f"Collecting data for Month {start_date.month}...")

    t = 5
    for attempt in range(1, 100):
        try:
            # Call API
            temp_df = get_measurements("TH", start_date, end_date)
        except KeyError:
            print(
                f"API limit exceeded for {start_date}. Trying again in {t} seconds... Retry attempt: {attempt}"
            )
            time.sleep(t)
            print(f"Trying again for {start_date}.")
            t += 5
        else:
            break
    else:
        print(f"Error collecting data for {start_date}. Skipping to next date...")

    # Add to base table
    openaq_df = pd.concat([openaq_df, temp_df])

    # When the month is finished, sleep for 3 mins --workaround for API limits
    if start_date.month != end_date.month:
        print(
            f"Month {start_date.month} done. Please wait for the next month to start... Time: 3 mins"
        )
        time.sleep(180)


print("Data collection complete.")
openaq_df

# %%
# Save to CSV
openaq_df.to_csv("20220527_openaq_raw_data.csv", index=False)
print("Raw data saved as CSV.")
