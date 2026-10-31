# TWDB TxRR freshwater-inflow timeseries from the Water Data for Texas coastal API.
# Generalized from ../temp_nwm_code/examples/compare_estuary_txrr.py.

import io
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape

TWDB_BASE_URL = "https://waterdatafortexas.org/coastal/api"

# Columns served by the TxRR geography timeseries that characterize the inflow:
# gauged, modeled ungaged, and the reported return / diversion terms.
INFLOW_COLS = ["gauge", "model", "return", "diversion"]

# The watershed-polygon layers are plain WDFT geometry fetches -- pull those with
# shapes.query_wdft_shapefile("watershed" / "sub_watershed") and filter on the
# `watershed` column. The gauge layer below is kept here because it needs the
# feature-level `name` (the USGS site_no), which query_wdft_shapefile hides behind
# the descriptive `extradata.name`.


#region \- geographies ---------------
def gauges(watershed: str | None = None) -> gpd.GeoDataFrame:
    """
    TWDB freshwater-inflow gauges (USGS) as points, optionally filtered by watershed.

    These are the gauges TWDB considers for the gauged inflow term.

    **Inputs**
        watershed : TWDB watershed/basin name (extradata.basin); None = all gauges
    **Outputs**
        GeoDataFrame (site_no, basin, name, source, geometry)
    """
    resp = requests.get(f"{TWDB_BASE_URL}/geometries/gauge")
    resp.raise_for_status()
    rows, geoms = [], []
    for f in resp.json()["features"]:
        ex = f.get("extradata") or {}
        if watershed is not None and ex.get("basin") != watershed:
            continue
        rows.append(
            {
                "site_no": f.get("name"),
                "basin": ex.get("basin"),
                "name": ex.get("name"),
                "source": ex.get("source"),
            }
        )
        geoms.append(shape(f["geometry"]))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


#endregion geographies ---------------


#region \- timeseries ---------------
def _geography_csv(session: requests.Session, gid) -> pd.DataFrame | None:
    """Fetch one geography's CSV, keeping only the inflow columns."""
    resp = session.get(
        f"{TWDB_BASE_URL}/hydrology/geography/{gid}/timeseries",
        params={"output_format": "csv"},
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["date"]).set_index("date")
    cols = [c for c in INFLOW_COLS if c in df.columns]
    return df[cols] if cols else None


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Clip a date-indexed frame to [start, end] (anything pandas parses)."""
    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def geography_timeseries(gid, start=None, end=None) -> pd.DataFrame:
    """
    Inflow series for a single geography (e.g. the watershed-level MONTHLY series).

    **Inputs**
        gid : TWDB geography id
        start, end : optional date range
    **Outputs**
        date-indexed DataFrame of the available inflow columns
    """
    with requests.Session() as session:
        df = _geography_csv(session, gid)
    if df is None:
        return pd.DataFrame(columns=INFLOW_COLS)
    return _slice(df.sort_index(), start, end)


def inflow_timeseries(geo_ids, start=None, end=None, max_workers=12) -> pd.DataFrame:
    """
    Daily TxRR freshwater inflow summed across geographies (acre-ft/day).

    Characterization columns (present where the API serves them): `gauge`
    (gauged), `model` (modeled ungaged, TxRR), `return` + `diversion` (reported).
    The API serves each sub-watershed's full period of record (no server-side date
    filter), so the per-geography CSVs are fetched concurrently and sliced here. For
    a much faster MONTHLY series use geography_timeseries on the watershed-level id.

    **Inputs**
        geo_ids : iterable of TWDB geography ids
        start, end : optional date range (anything pandas parses)
        max_workers : concurrent requests for the per-geography downloads
    **Outputs**
        date-indexed DataFrame of the available inflow columns
    """
    geo_ids = list(geo_ids)
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(geo_ids) or 1)) as ex:
            frames = [
                f
                for f in ex.map(lambda g: _geography_csv(session, g), geo_ids)
                if f is not None
            ]

    if not frames:
        return pd.DataFrame(columns=INFLOW_COLS)

    # Sum rows sharing a date across all sub-geographies (missing cols treated as 0).
    out = pd.concat(frames).groupby(level=0).sum().sort_index()
    return _slice(out, start, end)


#endregion timeseries ---------------
