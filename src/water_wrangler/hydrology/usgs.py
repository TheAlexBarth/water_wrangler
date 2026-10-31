# USGS NWIS streamflow gauges: discovery within a boundary + observed discharge.
# Observed counterpart to the modeled nwm.py streamflow; all NWIS access lives here.

import geopandas as gpd
import pandas as pd

from .. import shapes

# USGS discharge is served in cubic feet per second; keep everything in m3/s to
# match nwm.fetch_streamflow (1 cfs = 0.0283168 m3/s).
CFS_TO_CMS = 0.0283168

# USGS parameter code for discharge / streamflow.
DISCHARGE_PCODE = "00060"


#region \- gauge discovery ---------------
def find_gauges(boundary, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Active USGS discharge gauges within a boundary polygon.

    Queries the NWIS site service over the boundary's bounding box for stream
    sites carrying discharge (00060), then clips to the polygon itself (the bbox
    query can pull in neighboring-basin sites near a loose boundary).

    **Inputs**
        boundary : GeoDataFrame / GeoSeries / shapely geometry of the drainage
        crs : CRS of the boundary
    **Outputs**
        GeoDataFrame (site_no, station_nm, drain_area_va, geometry)
    """
    from dataretrieval import nwis  # lazy [hydrology] dep

    geom = shapes._as_geom(boundary)
    west, south, east, north = geom.bounds
    df, _ = nwis.get_info(
        bBox=f"{west:.6f},{south:.6f},{east:.6f},{north:.6f}",
        parameterCd=DISCHARGE_PCODE,
        siteType="ST",
        siteStatus="active",
    )
    if df is None or df.empty:
        return gpd.GeoDataFrame(
            columns=["site_no", "station_nm", "drain_area_va", "geometry"],
            geometry="geometry",
            crs=crs,
        )
    keep = [c for c in ("site_no", "station_nm", "drain_area_va") if c in df.columns]
    out = gpd.GeoDataFrame(
        df[keep].copy(),
        geometry=gpd.points_from_xy(df["dec_long_va"], df["dec_lat_va"]),
        crs=crs,
    )
    out["site_no"] = out["site_no"].astype(str)
    # Clip to the polygon (bbox query is rectangular).
    within = out.geometry.representative_point().within(geom)
    return out[within].drop_duplicates(subset="site_no").reset_index(drop=True)


#endregion gauge discovery ---------------


#region \- discharge ---------------
def _site_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Pull the 00060 discharge column from an NWIS frame, cfs -> m3/s, tz-naive.

    Returns None when the frame is empty or carries no discharge column -- the
    NWIS site service lists stream sites that have no daily/instantaneous
    discharge for the requested window, and those come back without a 00060 column.
    """
    if df is None or df.empty:
        return None
    cols = [c for c in df.columns if DISCHARGE_PCODE in c]
    if not cols:
        return None
    series = df[cols[0]].astype(float) * CFS_TO_CMS
    series.index = pd.to_datetime(series.index).tz_localize(None)
    return series


def fetch_discharge(sites, start=None, end=None, resolution="day") -> pd.DataFrame:
    """
    Observed USGS discharge (m3/s) for the given sites over a date range.

    Sites with no discharge data for the window (the NWIS bbox query can surface
    stream sites that don't report 00060) are skipped rather than erroring, so the
    returned frame holds only the columns that actually have data.

    **Inputs**
        sites : iterable of USGS site numbers
        start, end : optional date range (anything pandas parses)
        resolution : "day" for the daily mean (get_dv) or "hour" for the hourly
            series (get_iv instantaneous values resampled to the hourly mean)
    **Outputs**
        DataFrame indexed by date (daily) or timestamp (hourly), columns = site_no
    """
    if resolution not in ("day", "hour"):
        raise ValueError(f"usgs resolution must be 'day' or 'hour'; got {resolution!r}")
    from dataretrieval import nwis  # lazy [hydrology] dep

    out = {}
    for s in sites:
        s = str(s)
        try:
            if resolution == "day":
                df, _ = nwis.get_dv(sites=s, parameterCd=DISCHARGE_PCODE, start=start, end=end)
                series = _site_series(df)
            else:
                df, _ = nwis.get_iv(sites=s, parameterCd=DISCHARGE_PCODE, start=start, end=end)
                series = _site_series(df)
                if series is not None:
                    series = series.resample("1h").mean()
        except (ValueError, KeyError):
            # dataretrieval raises when a site returns no records for the window.
            series = None
        if series is not None:
            out[s] = series
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


#endregion discharge ---------------
