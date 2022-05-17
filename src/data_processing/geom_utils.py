import geopandas as gpd
from haversine import Direction, inverse_haversine


def convert_latlon_to_geometry(df, lat_col="latitude", lon_col="longitude"):
    df = df.copy()

    df["geometry"] = (
        "POINT(" + df[lon_col].astype("str") + " " + df[lat_col].astype("str") + ")"
    )
    df.drop(columns=[lat_col, lon_col], inplace=True)

    df = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))

    return df


def generate_bbox_wkt(centroid_lat, centroid_lon, distance_km):
    """Generates WKT string representing the bounding box for a given lat/lon coordinate"""
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

    tl_lat, tl_lon = top_left
    br_lat, br_lon = bottom_right

    tl_str = f"{tl_lon} {tl_lat}"
    tr_str = f"{br_lon} {tl_lat}"
    br_str = f"{br_lon} {br_lat}"
    bl_str = f"{tl_lon} {br_lat}"

    wkt_string = f"POLYGON(({tl_str}, {tr_str}, {br_str}, {bl_str}, {tl_str}))"

    return wkt_string


def generate_bboxes(
    locations_df,
    bbox_size_km,
    lat_col="latitude",
    lon_col="longitude",
    geometry_col="geometry",
):
    """Creates a bounding box geometry for a DF of lat/lon coordinates."""
    locations_df = locations_df.copy()
    locations_df[geometry_col] = locations_df.apply(
        lambda row: generate_bbox_wkt(
            row[lat_col], row[lon_col], distance_km=bbox_size_km
        ),
        axis=1,
    )
    return locations_df
