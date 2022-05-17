import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely import wkt

from src.data_processing import geom_utils


def collect_hrsl(locations_df, hrsl_tif, id_col, bbox_size_km, geometry_col="bbox"):
    locations_df = geom_utils.generate_bboxes(
        locations_df, bbox_size_km=bbox_size_km, geometry_col=geometry_col
    )

    locations_gdf = gpd.GeoDataFrame(
        locations_df, geometry=locations_df[geometry_col].apply(wkt.loads)
    )
    locations_gdf["total_population"] = pd.DataFrame(
        zonal_stats(vectors=locations_gdf[geometry_col], raster=hrsl_tif, stats="sum")
    )["sum"]

    # Retain only id_col and total_population to save on space.
    hrsl_df = pd.DataFrame(locations_gdf.drop(columns=geometry_col))[
        [id_col, "total_population"]
    ]

    return hrsl_df
