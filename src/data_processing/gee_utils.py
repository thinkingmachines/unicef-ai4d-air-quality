import ee
import numpy as np
import pandas as pd
from haversine import Direction, inverse_haversine


def generate_aoi_tile_data(
    collection_id,
    start_date,
    end_date,
    latitude,
    longitude,
    size_km=1,
    bands=None,
    cloud_filter=None,
):

    """
    Generates data for a given station and date range

    Parameters:
    - collection_id: ID of GEE collection
    - start_date: Start of desired date range
    - end_date: End of desired date range
    - latitude: Station latitude
    - longitude: Station longitude
    - bands: List of bands to get from GEE dataset

    Returns:
    - df: DataFrame of station data
    """

    # Generate bounding box
    bbox = generate_bbox(latitude, longitude, size_km)

    # Need to process by month to work within GEE limits
    date_range = pd.date_range(
        pd.Timestamp(start_date), pd.Timestamp(end_date), freq="MS"
    )

    all_dfs = []

    for i in range(0, len(date_range) - 1):
        date_from = date_range[i]
        date_to = date_range[i + 1]

        # Get ImageCollection
        images = get_gee_collection(
            collection_id, date_from, date_to, bands, cloud_filter
        )

        # Get table
        month_data = images.getRegion(
            bbox, 1000
        ).getInfo()  # --------- Region scale = 1000 m (defaults to WGS84)

        # Transform EE table
        df = transform_ee_array(month_data, bands)

        all_dfs.append(df)

    df = pd.concat(all_dfs)

    return df


def generate_bbox(centroid_lat, centroid_lon, distance_km, lon_lat=True):
    centroid = (centroid_lat, centroid_lon)
    top_left = inverse_haversine(
        inverse_haversine(centroid, distance_km / 2, Direction.WEST),
        distance_km / 2,
        Direction.NORTH,
    )
    bottom_right = inverse_haversine(
        inverse_haversine(centroid, distance_km / 2, Direction.EAST),
        distance_km / 2,
        Direction.SOUTH,
    )

    if lon_lat:
        bbox_coord_list = [top_left[1], top_left[0], bottom_right[1], bottom_right[0]]
    else:
        bbox_coord_list = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]

    return ee.Geometry.Rectangle(bbox_coord_list)


def get_gee_collection(
    collection_id, start_date, end_date, bands=None, cloud_filter=None
):

    # Import the collection
    collection = ee.ImageCollection(collection_id)

    # Select bands
    if bands:
        collection = collection.select(bands)

    # Filter dates
    collection = collection.filterDate(start_date, end_date)

    # Apply cloud filter
    if cloud_filter:
        collection = collection.filter(ee.Filter.eq("CLOUD_COVER", cloud_filter))

    return collection


def transform_ee_array(arr, bands=None):  # -- this is basically same as ee_array_to_df

    df = pd.DataFrame(arr)

    # --- Some steps for table formatting

    # Set first row as header
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Convert data types
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    if bands:
        for band in bands:
            df[band] = pd.to_numeric(df[band])

    # Drop rows with no data
    df.dropna(how="all", subset=bands, inplace=True)

    # Reset indices
    df.index = np.arange(1, len(df) + 1)

    # # Convert column names to lowercase
    # columns = df.columns.str.lower
    # df.columns = columns

    return df
