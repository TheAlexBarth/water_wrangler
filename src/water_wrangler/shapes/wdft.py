# watershed / grid geometries from the Water Data for Texas (WDFT) API

import json
import requests
import geopandas as gpd
from shapely.geometry import shape


def query_wdft_shapefile(geo_type: str):
    """
    get shapefile from wdft api

    **Input**
        geo_type : bay, basin, estuary, watershed, sub_watershed, or gauge

    **Outputs**
        geopandas.GeoDataFrame (attributes from the feature `extradata` + name/id)
    """
    TWDB_BASE_URL = "https://waterdatafortexas.org/coastal/api/geometries/"
    if geo_type not in ["bay", "basin", "estuary", "watershed", "sub_watershed", "gauge"]:
        raise ValueError(
            "geo_type must be one of: 'bay', 'basin', 'estuary', 'watershed', " \
            "'sub_watershed', or 'gauge'"
        )

    url = f"{TWDB_BASE_URL}{geo_type}"
    response = requests.get(url)
    response.raise_for_status()

    content = response.content.decode("utf-8")
    geojson = json.loads(content)

    # The API carries attributes in a top-level `extradata` (not under
    # `properties`); fold in the feature name/id too so results are labeled.
    features = geojson["features"]
    geometries = [shape(feature["geometry"]) for feature in features]
    records = []
    for feature in features:
        rec = dict(feature.get("extradata") or {})
        rec.setdefault("name", feature.get("name"))
        rec.setdefault("id", feature.get("id"))
        records.append(rec)
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    return gdf
