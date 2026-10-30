# shape/grid geometries from the Water Data for Texas (WDFT) API

import json
import numpy as np
import requests
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape


def query_shapefile(
        geo_type: str
):
    """
    get shapefile from wdft api

    **Input**
        geo_type : bay, basin, estuary, or watershed

    **Outputs**
        geopandas.GeoDataFrame
    """
    TWDB_BASE_URL = 'https://waterdatafortexas.org/coastal/api/geometries/'
    if geo_type not in ['bay', 'basin', 'estuary', 'watershed']:
        raise ValueError("geo_type must be one of: 'bay', 'basin', 'estuary', or 'watershed'")

    url = f"{TWDB_BASE_URL}{geo_type}"
    response = requests.get(url)
    response.raise_for_status()

    content = response.content.decode('utf-8')
    geojson = json.loads(content)

    # get features and convert to geometries and properties
    features = geojson['features']
    geometries = [shape(feature['geometry']) for feature in features]
    properties = [feature['properties'].get('extradata', {}) for feature in features]
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    return gdf