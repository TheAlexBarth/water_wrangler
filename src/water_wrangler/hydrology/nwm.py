# National Water Model (NWM) streamflow + NHDPlus reach handling.
# tools largely handled in dataset.py

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from .. import shapes

# daily-mean m3/s is kept as-is here; unit conversion happens in the dataset.

#region \- source constants ---------------
# NWM v3.0 CONUS Retrospective (anonymous S3 Zarr, 1979-02 .. 2023-01, hourly).
RETRO = {
    "zarr_uri": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
    "streamflow_var": "streamflow",  # m3 s-1 (scale_factor auto-applied)
    "reach_dim": "feature_id",  # == NHDPlusV2 COMID
    "gage_var": "gage_id",  # built-in reach -> USGS gage crosswalk
    "start": "1979-02-01",  # reanalysis coverage; drives auto source selection
    "end": "2023-02-01",  # past this, auto hands off to the operational analysis
}

# Operational extended Analysis & Assimilation (runs 16Z, tm00..tm23 -> 24 hourly).
OPERATIONAL = {
    "product": "analysis_assim_extend",
    "cycle": "t16z",
    "tm_steps": 24,
    "filename_tmpl": "nwm.{cycle}.{product}.channel_rt.tm{tm:02d}.conus.nc",
}

# Medium-Range forecast (hourly, member 1 main GFS-forced run).
FORECAST = {
    "product_tmpl": "medium_range_mem{member}",
    "filename_tmpl": "nwm.t{cycle:02d}z.medium_range.channel_rt_{member}.f{fhour:03d}.conus.nc",
    "member": 1,
    "cycle": 0,
    "horizon_days": 6,
}

# Anonymous NODD mirrors. AWS keeps only a short rolling window; GCS carries the
# archive continuously from 2018-09. We always pick per day: AWS when it is inside
# its window (fresher), else the GCS archive.
BACKENDS = {
    "gcs": {
        "protocol": "gcs",
        "bucket": "national-water-model",
        "storage_options": {"token": "anon"},
    },
    "aws": {
        "protocol": "s3",
        "bucket": "noaa-nwm-pds",
        "storage_options": {"anon": True},
    },
}
_AWS_FLOOR = pd.Timestamp("2025-01-01")  # AWS NODD rolling-window start
#endregion source constants ---------------


#region \- internal fx ---------------


def _open_retro():
    """Open the retrospective Zarr store anonymously."""
    return xr.open_zarr(
        RETRO["zarr_uri"],
        storage_options={"anon": True},
        consolidated=True,
        chunks={},
    )


#endregion internal fx ---------------


#region \- reach selection ---------------
def classify(
    flw: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Label each flowline contributing/terminal and derive downstream connectivity.


    **Inputs**
        flw : NHDPlus flowline network GeoDataFrame
    **Outputs**
        GeoDataFrame: feature_id, role, to_comid, tidal, gnis_name, streamorde, geometry
    """
    df = flw.copy()
    df.columns = [c.lower() for c in df.columns]
    df["comid"] = df["comid"].astype("int64")

    order = pd.to_numeric(df.get("streamorde"), errors="coerce").fillna(-9)
    df = df[order >= 1].copy()

    hs = pd.to_numeric(df["hydroseq"], errors="coerce")
    dn = pd.to_numeric(df["dnhydroseq"], errors="coerce").fillna(0)
    to_comid = dn.map(dict(zip(hs, df["comid"]))).fillna(0).astype("int64")
    tidal = pd.to_numeric(df.get("tidal"), errors="coerce").fillna(0).astype(int)

    termfl = (
        pd.to_numeric(df.get("terminalfl"), errors="coerce").fillna(0).astype(int)
    )
    # true network ends
    is_terminal = (termfl == 1) | (dn == 0)
  

    out = gpd.GeoDataFrame(
        {
            "feature_id": df["comid"].values,
            "role": np.where(is_terminal, "terminal", "contributing"),
            "to_comid": to_comid.values,
            "tidal": tidal.values,
            "gnis_name": df["gnis_name"].values if "gnis_name" in df.columns else "",
            "streamorde": (
                df["streamorde"].values if "streamorde" in df.columns else np.nan
            ),
            "geometry": df.geometry.values,
        },
        crs=flw.crs,
    )
    return out.drop_duplicates(subset="feature_id").reset_index(drop=True)


def attach_gauges(reaches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Crosswalk `gage_id`/`gaged` onto reaches from the retrospective Zarr gage var.
    """
    ds = _open_retro()
    all_fids = ds[RETRO["reach_dim"]].values
    # gage_id is a fixed-width bytes array (|S15); decode + strip padding.
    gage_ids = np.char.strip(np.char.decode(ds[RETRO["gage_var"]].values, "ascii"))
    gmap = pd.Series(gage_ids, index=all_fids)
    out = reaches.copy()
    out["gage_id"] = out["feature_id"].map(gmap).fillna("").astype(str)
    out["gaged"] = (out["gage_id"].str.len() > 0).astype(int)
    return out


def _coastline_rows(flw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Coastline features (shoreline, non-flow) as role='coastline' rows."""
    coast = flw[flw.get("ftype") == "Coastline"]
    if not len(coast):
        return gpd.GeoDataFrame(columns=["feature_id", "role", "geometry"], crs=flw.crs)
    return gpd.GeoDataFrame(
        {
            "feature_id": coast["comid"].astype("int64").values,
            "role": "coastline",
            "to_comid": 0,
            "tidal": 0,
            "gnis_name": (
                coast["gnis_name"].values if "gnis_name" in coast.columns else ""
            ),
            "streamorde": np.nan,
            "gage_id": "",
            "gaged": 0,
            "geometry": coast.geometry.values,
        },
        crs=flw.crs,
    )


def select_coastal_reaches(
    boundary,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    NHDPlus flowlines within a boundary, classified and (optionally) gage-crosswalked.

    Select reaches that are at a network terminus to the tidal bounary

    **Inputs**
        boundary : GeoDataFrame / GeoSeries / shapely geometry of the drainage
        crs : CRS of the boundary
        gages : attach the retrospective gage crosswalk
        coastline : also return NHDPlus Coastline features (role='coastline') for
            estuary delineation on maps; excluded from the inflow math
    **Outputs**
        GeoDataFrame of reaches (feature_id, role, to_comid, geometry, gage_id, gaged)
    """
    from pynhd import WaterData  # lazy [hydrology] dep

    geom = shapes._as_geom(boundary)
    flw = WaterData("nhdflowline_network").bygeom(geom, geo_crs=crs)
    flw.columns = [c.lower() for c in flw.columns]
    reaches = classify(flw)
    
    # Keep reaches whose midpoint falls inside the drainage boundary (so we do not
    # pick up a neighboring estuary's reaches near a loose boundary).
    within = reaches.geometry.representative_point().within(geom)
    reaches = reaches[within].reset_index(drop=True)

    # Keep reaches that flow leaves boundary AND is either tidally influenced
    # or a stream order >= 3. This willd drop minor streams that end inside boundary
    within_ids = set(reaches["feature_id"].astype("int64"))
    exits = ~reaches["to_comid"].astype("int64").isin(within_ids)
    order = pd.to_numeric(reaches["streamorde"], errors="coerce").fillna(0)
    is_terminal = exits & ((reaches["tidal"] == 1) | (order >= 3))
    reaches["role"] = np.where(is_terminal, "terminal", "contributing")
    reaches = attach_gauges(reaches)

    coast = _coastline_rows(flw)
    if len(coast):
        reaches = gpd.GeoDataFrame(
            pd.concat([reaches, coast], ignore_index=True), crs=flw.crs
        )
    return reaches


#endregion reach selection ---------------


#region \- streamflow fetch ---------------
def fetch_streamflow(feature_ids, start=None, end=None):
    """
    Daily-mean streamflow (m3/s) for the given reaches over a date range.

    **Inputs**
        feature_ids : iterable of NWM feature_ids (NHDPlus COMIDs)
        start, end : date range (defaults: reanalysis start .. today)
    **Outputs**
        DataFrame indexed by date, columns = feature_id
    """
    feature_ids = np.unique(np.asarray(list(feature_ids), dtype="int64"))
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = pd.Timestamp(start) if start is not None else pd.Timestamp(RETRO["start"])
    end = pd.Timestamp(end) if end is not None else today
    if end < start:
        raise ValueError(f"end {end.date()} precedes start {start.date()}")
    retro_end = pd.Timestamp(RETRO["end"])

    frames = []
    if start <= retro_end:  # reanalysis where available
        frames.append(_read_retro(feature_ids, start, min(end, retro_end)))
    op_start, op_end = max(start, retro_end + pd.Timedelta(days=1)), min(end, today)
    if op_end >= op_start:  # operational analysis past the reanalysis, up to today
        frames.append(_read_files("operational", feature_ids, op_start, op_end))
    if end > today:  # forecast into the future
        frames.append(_read_files("forecast", feature_ids, today, end))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    # Stitch: drop the day shared by adjacent sources, keep the first (preferred) one.
    out = pd.concat(frames)
    return out[~out.index.duplicated("first")].sort_index()


def _read_retro(feature_ids, start, end):
    """Slice the retrospective Zarr store for the requested reaches and window."""
    ds = _open_retro()
    present = np.intersect1d(feature_ids, ds[RETRO["reach_dim"]].values)
    if len(present) == 0:
        return pd.DataFrame()
    # The store is hourly; floor start / ceil end to whole days so a bare-date
    # endpoint still includes that full day's 24 hours (a plain Timestamp end would
    # otherwise clip the slice at 00:00 and leave a one-hour mean for the last day).
    lo = None if start is None else pd.Timestamp(start).normalize()
    hi = (
        None
        if end is None
        else pd.Timestamp(end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    )
    q = (
        ds[RETRO["streamflow_var"]]
        .sel({RETRO["reach_dim"]: present})
        .sel(time=slice(lo, hi))
        .load()
    )
    return q.resample(time="1D").mean().to_pandas()


def _read_files(source, feature_ids, start, end):
    """
    Daily-mean streamflow from the per-file channel_rt NetCDFs.

    The two file-based products read the same variable and differ only in how the
    files enumerate: operational walks (day x tm) hourly analysis slabs, forecast
    walks forecast hours from one init cycle (member 1, horizon spanning the
    window). The NODD mirror is chosen per file from its date -- AWS inside its
    rolling window, else the GCS archive.
    """
    import fsspec  # lazy [hydrology] dep

    files = []  # (uri, mirror_cfg)
    if source == "operational":
        for day in pd.date_range(start, end, freq="D"):
            be = BACKENDS["aws"] if day >= _AWS_FLOOR else BACKENDS["gcs"]
            base = f"{be['bucket']}/nwm.{day:%Y%m%d}/{OPERATIONAL['product']}"
            for tm in range(OPERATIONAL["tm_steps"]):
                fname = OPERATIONAL["filename_tmpl"].format(
                    cycle=OPERATIONAL["cycle"], product=OPERATIONAL["product"], tm=tm
                )
                files.append((f"{base}/{fname}", be))
    else:  # forecast from one init cycle, horizon clipped to the window
        init, member = pd.Timestamp(start), FORECAST["member"]
        horizon = min(max((pd.Timestamp(end) - init).days + 1, 1), FORECAST["horizon_days"])
        be = BACKENDS["aws"] if init >= _AWS_FLOOR else BACKENDS["gcs"]
        base = f"{be['bucket']}/nwm.{init:%Y%m%d}/{FORECAST['product_tmpl'].format(member=member)}"
        for fhour in range(1, 24 * horizon + 1):
            fname = FORECAST["filename_tmpl"].format(
                cycle=FORECAST["cycle"], member=member, fhour=fhour
            )
            files.append((f"{base}/{fname}", be))

    fs_cache: dict = {}
    slabs = []
    for uri, be in files:
        fs = fs_cache.setdefault(
            be["protocol"], fsspec.filesystem(be["protocol"], **be["storage_options"])
        )
        if not fs.exists(uri):
            continue
        with fs.open(uri) as fh:
            ds = xr.open_dataset(fh, engine="h5netcdf")
            fids = ds["feature_id"].values
            idx = np.searchsorted(fids, feature_ids)
            ok = fids[np.clip(idx, 0, len(fids) - 1)] == feature_ids
            da = ds["streamflow"].isel(feature_id=idx[ok])
            da = da.assign_coords(feature_id=feature_ids[ok])
            # streamflow carries only feature_id; attach the valid time so concat
            # builds a proper datetime axis.
            slabs.append(da.expand_dims(time=ds["time"].values).load())
    if not slabs:
        return pd.DataFrame()
    out = xr.concat(slabs, dim="time").sortby("time")
    _, keep = np.unique(out["time"].values, return_index=True)
    return out.isel(time=np.sort(keep)).resample(time="1D").mean().to_pandas()


#endregion streamflow fetch ---------------
