# watershed geometries from the USGS Watershed Boundary Dataset (WBD / HUC)

import os
import requests
import geopandas as gpd

from .geom import _as_geom

# WBD staged products: one zip per 2-digit region bundles HUC2..HUC12 shapefiles.
# Region 12 = Texas-Gulf (11 Arkansas-White-Red, 13 Rio Grande also touch TX).
WBD_BASE_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/Shape/"
)


def query_huc_watershed(level=10, region="12", intersects=None, huc=None, cache_dir="cache"):
    """
    HUC watershed polygons from the USGS Watershed Boundary Dataset (WBD).

    **Inputs**
        level : HUC level, one of 8, 10, 12 (default 10, the estuary-drainage level)
        region : 2-digit hydrologic region (default "12", Texas-Gulf)
        intersects : optional boundary (GeoDataFrame/GeoSeries/geometry); keep only
            HUCs intersecting it, e.g. the WDFT estuary polygon draining to one bay
        huc : optional HUC code prefix (or list of prefixes) to filter on, e.g. a
            HUC4/HUC6 parent like "1204" or ("1203", "1204")
        cache_dir : directory for the downloaded WBD zip (default "cache")

    **Outputs**
        geopandas.GeoDataFrame (huc code, name, geometry) in EPSG:4326
    """
    if level not in (8, 10, 12):
        raise ValueError(f"level must be 8, 10, or 12; got {level!r}")

    fname = f"WBD_{region}_HU2_Shape.zip"
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        with requests.get(WBD_BASE_URL + fname, stream=True) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    gdf = gpd.read_file(f"zip://{path}!Shape/WBDHU{level}.shp").to_crs("EPSG:4326")
    code = f"huc{level}"
    if huc is not None:
        prefixes = (huc,) if isinstance(huc, str) else tuple(huc)
        gdf = gdf[gdf[code].str.startswith(prefixes)]
    if intersects is not None:
        gdf = gdf[gdf.intersects(_as_geom(intersects))]
    return gdf[[code, "name", "geometry"]].reset_index(drop=True)
