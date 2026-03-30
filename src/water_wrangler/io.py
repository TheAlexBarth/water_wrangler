# importing from midgewater and wdft API

import datetime as dt
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm
from water_wrangler.baycast_dataset import BaycastDataset


#region \- internal fx ---------------


def _normalize_date(
    date: str | dt.date | dt.datetime | pd.Timestamp
) -> dt.datetime:
    """
    Convert date inputs to UTC datetime.
    """
    if isinstance(date, dt.datetime):
        return date.replace(tzinfo=None)
    if isinstance(date, dt.date):
        return dt.datetime(date.year, date.month, date.day)
    return pd.to_datetime(date).to_pydatetime().replace(tzinfo=None)

def _midgewater_filename(
    product: str,
    dimension: str,
    type_2d: str,
    date: dt.datetime
) -> str:
    """
    Build the expected BAYCAST filename for a given date.

    2D: {product}_{dimension}_{type_2d}_{YYYYMMDD}.nc
    3D: {product}_{dimension}_{YYYYMMDD}.nc
    """
    product = product.lower()
    dimension = dimension.upper()
    type_2d = type_2d.lower()
    datestr = date.strftime("%Y%m%d")

    if dimension == "2D":
        return f"{product}_{dimension}_{type_2d}_{datestr}.nc"
    else:
        return f"{product}_{dimension}_{datestr}.nc"
    
def _midgewater_url(
    product: str,
    dimension: str,
    type_2d: str,
    date: dt.datetime,
    base_url = "https://midgewater.twdb.texas.gov/bays_estuaries/baycast/"
) -> str:
    """Construct the full Midgewater URL for a given date."""
    fname = _midgewater_filename(product, dimension, type_2d, date)
    return f"{base_url}{fname}"
#endregion internal help fx---------------

#################
#MARK: MIDGEWATER
#################


def query_baycast(
    date: str | dt.date | dt.datetime | pd.Timestamp,
    to_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    save_dir: str = '.',
    product: str = "chem",
    dimension: str = '2D',
    type_2d: str = 'depth'
):
    """
    Download and save netCDF files from TWDB portal

    **Inputs**
    date, to_date:
        Date start time with optional to_date. Accepts strings or datetime objects
    product : {"chem", "hydro"}
        BAYCAST product
    dimension : {"2D", "3D"}
        2D or 3D product
    type_2d: {"depth", "surf"}
        if dimension = 2D, depth integrated or surface
    save_dir: where to save datasets
    
    **Outputs**
    None
    """
    date = _normalize_date(date)
    if to_date:
        to_date = _normalize_date(to_date)
    else:
        to_date = date

    # Validate product/dimension/type_2d
    product = product.lower()
    if product not in {"chem", "hydro"}:
        raise ValueError("product must be 'chem' or 'hydro'.")

    dimension = dimension.upper()
    if dimension not in {"2D", "3D"}:
        raise ValueError("dimension must be '2D' or '3D'.")

    type_2d = type_2d.lower()
    if dimension == "2D" and type_2d not in {"depth", "surf"}:
        raise ValueError("type_2d must be 'depth' or 'surf' for 2D outputs.")
    
    
    # Prepare output directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare HTTP session
    sess = requests.Session()

    # Build date sequence
    dates = pd.date_range(date, to_date, freq="D")

    downloaded_paths: List[Path] = []

    for date in tqdm(dates, desc="Downloading BAYCAST from Midgewater"):
        date_dt = date.to_pydatetime().replace(tzinfo=None)
        url = _midgewater_url(product, dimension, type_2d, date_dt)

        fname = _midgewater_filename(product, dimension, type_2d, date_dt)
        out_path = save_dir / fname

        resp = sess.get(url)
        if resp.status_code != 200:
            # Hard fail: you said you want to fail if data are not available
            raise RuntimeError(
                f"Failed to download BAYCAST file for {date_dt.date()} "
                f"from {url} (status={resp.status_code})."
            )

        out_path.write_bytes(resp.content)


###########
#MARK: LOAD BAYCAST
###########

def load_baycast(
    date: str | dt.date | dt.datetime | pd.Timestamp,
    to_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    product: str = "chem",
    dimension: str = "2D",
    type_2d: str = "depth",
    local_dir: str | Path | None = None,
    convert_time: bool = True,  # True = convert UTC ➝ US Central (America/Chicago)
    **open_kwargs,
) -> BaycastDataset:
    """
    Open or Download baycast datasets

    **Inputs**
    date, to_date:
        Date range to load, if to_date empty only will use date
    product, dimension, type_2d :
        BAYCAST product selection. Must match Midgewater filename scheme.
    local_dir : str or Path or None
        - If not None: Local mode (read-only).
        - If None: Remote mode (temp-only).
    convert_time : bool, default True
        If True, convert time coord from UTC to US Central (America/Chicago)
        and drop timezone info for xarray friendliness.
    **open_kwargs :
        Extra keyword arguments passed to `xarray.open_mfdataset`.
    **Outputs**
    BaycastDataset
    """

    # Normalize and validate date range
    start = _normalize_date(date)
    end = _normalize_date(to_date or date)
    
    # Build expected filename for each date in the range
    dates = pd.date_range(start, end, freq="D")
    expected_files = {
        date.to_pydatetime().replace(tzinfo=None): _midgewater_filename(
            product, dimension, type_2d, date.to_pydatetime()
        )
        for date in dates
    }

    if local_dir is not None:
        local_dir_path = Path(local_dir)
        file_paths: List[Path] = []
        missing: List[str] = []

        for dt_val, fname in tqdm(expected_files.items()):
                candidate = local_dir_path / fname
                if candidate.exists():
                    file_paths.append(candidate)
                else:
                    missing.append(f"{dt_val.date()} ({fname})")
        if len(missing) > 0:
            raise FileNotFoundError(
                "Missing BAYCAST files in local_dir for the requested range: "
                + ", ".join(missing)
            )
        if not file_paths:
            raise FileNotFoundError(
                "No BAYCAST files found in local_dir for the requested range."
            )

        ds = xr.open_mfdataset(file_paths, data_vars='minimal', compat = 'no_conflicts', **open_kwargs)
    
    
    temp_files: List[str] = []
    sess = requests.Session()
    try:
        for dt_val in tqdm(expected_files.keys()):
            url = _midgewater_url(product, dimension, type_2d, dt_val)
            resp = sess.get(url)
            if resp.status_code != 200:
                for tmp in temp_files:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"Could not downlaod BAYCAST for {dt_val.date()}"
                        f"from {url}"
                    )
                
            with tempfile.NamedTemporaryFile(delete=False, suffix = '.nc') as tmp:
                tmp.write(resp.content)
                temp_files.append(tmp.name)

        if not temp_files:
            raise FileNotFoundError(
                "No BAYCAST files could be downloaded for the requested range."
            )
        
        ds = xr.open_mfdataset(temp_files, data_vars="minimal", compat = 'no_conflicts', **open_kwargs)

        if convert_time and "time" in ds:
            time_utc = pd.DatetimeIndex(ds["time"].values).tz_localize('UTC')
            time_central = time_utc.tz_convert("America/Chicago").tz_localize(None)
            ds = ds.assign_coords(time=("time", time_central))
        
    finally:
        for tmp in temp_files:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return BaycastDataset(ds)


######################
#MARK: WDFT
######################

# def query_station_meta(
        
# )

# def query_station_data(
#         station_id: str,
#         parameter_code: str,
#         start_date: str | dt.date | dt.datetime | pd.Timestamp = None,
#         end_date: str | dt.date | dt.datetime | pd.Timestamp = None,
# )


import json
import geopandas as gpd
from shapely.geometry import shape, box, Point
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

def _make_bbox(extent: list | gpd.GeoDataFrame):
    """
    Make a bounding box around an extent

    **Inputs**
        extent : either [minx, miny, maxx, maxy] or a geopandas dataframe
    **Output**
        bbox : list
    """
    if isinstance(extent, gpd.GeoDataFrame):
        bbox = extent.total_bounds
    elif isinstance(extent, list) or isinstance(extent, np.ndarray):
        if len(extent) != 4:
            raise ValueError("List must be form [minx, miny, maxx, maxy]")
        bbox = extent
    return bbox


import rasterio
from rasterio.features import rasterize
def _make_rast(shp, rast_res = 0.001):
    bbox = _make_bbox(shp)
    xmin, ymin, xmax, ymax = bbox
    transform = rasterio.transform.from_bounds(
        *bbox,
        width=int((xmax - xmin) / rast_res),
        height=int((ymax - ymin) / rast_res)
    )
    out_shape = (
        int((ymax - ymin) / rast_res),
        int((xmax - xmin) / rast_res)
    )
    raster = rasterize(
        [(geom, 1) for geom in shp.geometry],
        out_shape=out_shape, transform=transform,
        fill=0, dtype='uint8'
    )
    return {
        "profile": {
            "driver": "GTiff",
            "transform": transform,
            'dtype': 'uint8', 
            'count': 1,
            'width': int(out_shape[1]),
            'height': int(out_shape[0]),
            "crs": shp.crs,
            'compress': 'lzw'
        },
        "raster": raster,
    }
        
