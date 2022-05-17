import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely import wkt

from src.data_processing.geom_utils import generate_bboxes


def collect_hrsl(locations_df, hrsl_tif, id_col, bbox_size_km):
    locations_df = generate_bboxes(locations_df, bbox_size_km=bbox_size_km)
    # locations_df.to_csv(settings.DATA_DIR / "debug_locations.csv")
    hrsl_gdf = gpd.GeoDataFrame(
        locations_df, geometry=locations_df["geometry"].apply(wkt.loads)
    )
    hrsl_gdf["total_population"] = pd.DataFrame(
        zonal_stats(vectors=hrsl_gdf["geometry"], raster=hrsl_tif, stats="sum")
    )["sum"]

    # Retain only id_col and total_population to save on space.
    hrsl_df = pd.DataFrame(hrsl_gdf.drop(columns="geometry"))[
        [id_col, "total_population"]
    ]
    return hrsl_df
