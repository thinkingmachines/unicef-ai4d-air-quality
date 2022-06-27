import math
import time
import traceback

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm


def get_openaq_measurements(
    country_id,
    start_date,
    end_date,
    parameter="pm25",
    sensor_type=None,
    limit=1000,
    include_mobile=False,
    server_url="https://api.openaq.org/v2/measurements",
):

    all_records = []

    # Batch per day due to 100k total limit by the API
    date_range = pd.date_range(start_date, end_date)
    for d in date_range:
        curr_start_date = d.date()
        curr_end_date = (pd.to_datetime(d) + pd.Timedelta(days=1)).date()

        base_params = {
            "date_from": curr_start_date,
            "date_to": curr_end_date,
            "country_id": country_id,
            "limit": limit,
            "isMobile": include_mobile,
            "parameter": parameter,
            "has_geo": True,
            "page": 1,
        }

        # If sensor type is explicitly indicated
        if sensor_type:
            base_params["sensorType"] = sensor_type

        # Make the first call to get the limit and compute the expected number of pages
        api_response = requests.get(server_url, base_params)
        total_records = api_response.json()["meta"]["found"]
        num_pages = math.ceil(total_records / limit)

        logger.info(
            f"Collecting for {curr_start_date}. Total records: {total_records}, Num pages: {num_pages}"
        )

        # Collect all the records one page at a time
        all_raw_responses = []
        for page in tqdm(range(1, num_pages + 1)):

            # This is to keep re-attempting if it fails.
            attempt = 0
            while True:
                attempt += 1

                try:
                    # Construct copy of params with the right page number
                    params = base_params.copy()
                    params["page"] = page
                    api_response = requests.get(server_url, params)

                    # Collect all the raw records in a list
                    records = api_response.json()["results"]
                    all_records.extend(records)
                    all_raw_responses.append(api_response)

                    break
                except Exception:
                    traceback.print_exc()
                    logger.error(
                        f"Page {page} Attempt {attempt} response: {api_response}\n{api_response.json()}"
                    )
                    logger.error(all_raw_responses[-1].json()["meta"])
                    time.sleep(30)

    df = pd.json_normalize(all_records)

    assert len(df["country"].unique()) == 1
    assert df["country"].unique().tolist()[0] == country_id

    return df
