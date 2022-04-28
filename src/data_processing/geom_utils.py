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
