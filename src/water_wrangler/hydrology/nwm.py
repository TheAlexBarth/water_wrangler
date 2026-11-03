# National Water Model (NWM) streamflow + NHDPlus reach handling.
# tools largely handled in dataset.py

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from .. import shapes

# streamflow m3/s is kept as-is here (daily-mean or hourly per `resolution`);
# unit conversion happens in the dataset.

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

# Medium-Range forecast (hourly, member 1 main GFS-forced run). The init cycle is
# picked at fetch time as the latest posted one (see _latest_forecast_init), not fixed.
FORECAST = {
    "product_tmpl": "medium_range_mem{member}",
    "filename_tmpl": "nwm.t{cycle:02d}z.medium_range.channel_rt_{member}.f{fhour:03d}.conus.nc",
    "member": 1,
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

# Per-file channel_rt NetCDF reads are open-overhead bound (HDF5 metadata over HTTP).
# We fan them out across threads and cache whole files to local disk so reruns /
# incremental date extensions are near-free.
_MAX_WORKERS = 16  # ThreadPoolExecutor width for per-file opens
_CACHE_DIR = os.environ.get(
    "WATER_WRANGLER_NWM_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "water_wrangler", "nwm"),
)
_FETCH_ATTEMPTS = 3  # per-file fetch retries before giving up on a transient error
_FETCH_BACKOFF = 1.0  # seconds; doubled each retry
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

    # Drop coded/sub-stream reaches (streamorde < 1)
    order = pd.to_numeric(df.get("streamorde"), errors="coerce").fillna(-9)
    # keep terminal reaches and head waters
    _termfl = pd.to_numeric(df.get("terminalfl"), errors="coerce").fillna(0).astype(int)
    _dn = pd.to_numeric(df.get("dnhydroseq"), errors="coerce").fillna(0)
    df = df[(order >= 1) | (_termfl == 1) | (_dn == 0)].copy()

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


def downstream_most(feature_ids, to_comid_map) -> list:
    """
    Keep only the downstream-most reach in each chain.

    Walks the `feature_id -> downstream feature_id` graph and drops any reach that
    is nested upstream of another reach in the set, so a gauged reach's flow is not
    counted (or subtracted) twice. Shared by the nwm and usgs gauge logic.

    **Inputs**
        feature_ids : iterable of reach feature_ids (NHDPlus COMIDs)
        to_comid_map : dict feature_id -> downstream feature_id (0 for a terminus)
    **Outputs**
        list of the downstream-most feature_ids
    """
    gaged = set(int(g) for g in feature_ids)
    keep = []
    for g in gaged:
        others = gaged - {g}
        c, seen, nested = to_comid_map.get(g, 0), {g}, False
        while c and c not in seen:
            if c in others:
                nested = True
                break
            seen.add(c)
            c = to_comid_map.get(c, 0)
        if not nested:
            keep.append(g)
    return keep


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
    if "ftype" not in flw.columns:
        return gpd.GeoDataFrame(columns=["feature_id", "role", "geometry"], crs=flw.crs)
    coast = flw[flw["ftype"] == "Coastline"]
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
    NHDPlus flowlines within a boundary, classified and gage-crosswalked.

    Select reaches that are at a network terminus to the tidal boundary.

    **Inputs**
        boundary : GeoDataFrame / GeoSeries / shapely geometry of the drainage
        crs : CRS of the boundary
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
    # or a stream order >= 3. This will drop minor streams that end inside boundary
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
def fetch_streamflow(
    feature_ids,
    start=None,
    end=None,
    resolution="day",
    workers=_MAX_WORKERS,
    cache_dir=None,
    store=None,
    retain_raw=False
):
    """
    Streamflow (m3/s) for the given reaches over a date range.

    **Inputs**
        feature_ids : iterable of NWM feature_ids (NHDPlus COMIDs)
        start, end : date range (defaults: reanalysis start .. today)
        resolution : "day" for the daily mean (default) or "hour" for the hourly frame
        workers : thread-pool width for the per-file operational/forecast reads
        cache_dir : on-disk whole-file cache for the per-file reads (used when retain_raw)
        store : optional directory to store parquets for analysis window (<= today)
        retain_raw : if True keep the whole CONUS file in `cache_dir`; if False (default),
            stream + subset + discard (no bulky on-disk copy).
    **Outputs**
        DataFrame indexed by date (daily) or timestamp (hourly), columns = feature_id
    """
    feature_ids = np.unique(np.asarray(list(feature_ids), dtype="int64"))
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = pd.Timestamp(start) if start is not None else pd.Timestamp(RETRO["start"])
    end = pd.Timestamp(end) if end is not None else today
    if end < start:
        raise ValueError(f"end {end.date()} precedes start {start.date()}")
    retro_end = pd.Timestamp(RETRO["end"])

    if store is not None:
        return _fetch_store(
            feature_ids, start, end, resolution, workers, cache_dir, store,
            retain_raw, today,
        )

    frames = []
    if start <= retro_end:  # reanalysis where available
        frames.append(_read_retro(feature_ids, start, min(end, retro_end), resolution))
    op_start, op_end = max(start, retro_end + pd.Timedelta(days=1)), min(end, today)
    if op_end >= op_start:  # operational analysis past the reanalysis, up to today
        frames.append(
            _read_files(
                "operational", feature_ids, op_start, op_end, resolution,
                workers=workers, cache_dir=cache_dir, retain_raw=retain_raw,
            )
        )
    if end > today:  # forecast into the future
        frames.append(
            _read_files(
                "forecast", feature_ids, today, end, resolution,
                workers=workers, cache_dir=cache_dir, retain_raw=retain_raw,
            )
        )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    # drop shared if stiching across sources
    out = pd.concat(frames)
    return out[~out.index.duplicated("first")].sort_index()


def _fetch_store(
    feature_ids, start, end, resolution, workers, cache_dir, store, retain_raw, today
):
    """
    fetch_streamflow from stored period of analysis window (useful in running multiple)
    """
    path = os.path.join(store, f"streamflow_{resolution}.parquet")
    have = pd.DataFrame()
    if os.path.exists(path):
        have = pd.read_parquet(path)
        have.columns = have.columns.astype("int64")
    stored_ids = set(int(c) for c in have.columns)

    a_end = min(end, today)  # cacheable analysis end
    analysis = pd.DataFrame()
    if a_end >= start:
        req_days = pd.date_range(start.normalize(), a_end.normalize(), freq="D")
        have_days = (
            set(pd.DatetimeIndex(have.index).normalize()) if not have.empty else set()
        )
        if any(f not in stored_ids for f in feature_ids):
            missing = req_days  # a newly-requested reach -> refetch the requested days
        else:
            missing = pd.DatetimeIndex([d for d in req_days if d not in have_days])
        if len(missing):
            fetch_ids = (
                np.unique(np.concatenate([feature_ids, np.fromiter(stored_ids, "int64")]))
                if stored_ids
                else feature_ids
            )
            fresh = fetch_streamflow(
                fetch_ids, start=missing.min(), end=missing.max(), resolution=resolution,
                workers=workers, cache_dir=cache_dir, retain_raw=retain_raw,
            )
            have = fresh if have.empty else fresh.combine_first(have)
            have = have.sort_index()
            os.makedirs(store, exist_ok=True)
            out = have.copy()
            out.columns = out.columns.astype(str)  # pyarrow needs string column names
            out.to_parquet(path)
        cols = [c for c in feature_ids if c in set(have.columns)]
        hi = a_end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        analysis = have.loc[(have.index >= start.normalize()) & (have.index <= hi), cols]

    forecast = pd.DataFrame()
    # still needs to fetch forecast if requested
    if end > today:
        forecast = fetch_streamflow(
            feature_ids, start=today, end=end, resolution=resolution,
            workers=workers, cache_dir=cache_dir, retain_raw=retain_raw,
        )
        forecast = forecast[pd.DatetimeIndex(forecast.index).normalize() > today]

    frames = [f for f in (analysis, forecast) if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out[~out.index.duplicated("first")].sort_index()


def _read_retro(feature_ids, start, end, resolution="day"):
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
    if resolution == "hour":
        return q.to_pandas()
    return q.resample(time="1D").mean().to_pandas()


def _latest_forecast_init(end, lookback_days=3):
    """Most recent medium-range init cycle whose files are actually posted.

    The forecast is reissued every 6 h (cycles 00/06/12/18Z) with a posting lag, so
    `today`/t00z is often not up yet (early UTC or during the lag). Probe candidate
    inits newest-first over the last few days and return the first whose f001
    channel_rt file exists, as (init Timestamp, backend cfg); (None, None) if none.
    """
    import fsspec  # lazy [hydrology] dep

    now = pd.Timestamp.now("UTC").tz_localize(None)
    member = FORECAST["member"]
    product = FORECAST["product_tmpl"].format(member=member)
    fs_cache = {}
    for d in range(lookback_days + 1):
        day = now.normalize() - pd.Timedelta(days=d)
        for cyc in (18, 12, 6, 0):
            init = day + pd.Timedelta(hours=cyc)
            if init > now:
                continue  # cycle not reached yet
            be = BACKENDS["aws"] if init.normalize() >= _AWS_FLOOR else BACKENDS["gcs"]
            proto = be["protocol"]
            fs = fs_cache.get(proto)
            if fs is None:
                fs = fs_cache[proto] = fsspec.filesystem(proto, **be["storage_options"])
            fname = FORECAST["filename_tmpl"].format(cycle=cyc, member=member, fhour=1)
            uri = f"{be['bucket']}/nwm.{init:%Y%m%d}/{product}/{fname}"
            try:
                if fs.exists(uri):
                    return init, be
            except Exception:
                continue
    return None, None


def _read_files(
    source, feature_ids, start, end, resolution="day",
    workers=_MAX_WORKERS, cache_dir=None, retain_raw=True,
):
    """
    Streamflow from the per-file channel_rt NetCDFs (daily mean or hourly).

    The two file-based products read the same variable and differ only in how the
    files enumerate: operational walks (day x tm) hourly analysis slabs, forecast
    walks forecast hours from one init cycle (member 1, horizon spanning the
    window). The NODD mirror is chosen per file from its date -- AWS inside its
    rolling window, else the GCS archive.

    Files are opened concurrently (`workers` threads) and whole-file cached to
    `cache_dir`
    """
    import io
    import threading
    import time
    import warnings
    import fsspec  # lazy [hydrology] dep
    from concurrent.futures import ThreadPoolExecutor

    def _cat_file(fs, uri):
        """Fetch a remote file, retrying transient errors with exponential backoff.

        A missing object (FileNotFoundError) is skipped other errors are tried
        up to `_FETCH_ATTEMPTS` which is 3
        """
        for attempt in range(_FETCH_ATTEMPTS):
            try:
                return fs.cat_file(uri)
            except FileNotFoundError:
                raise
            except Exception as exc:  # transient network/server error
                if attempt + 1 >= _FETCH_ATTEMPTS:
                    raise
                warnings.warn(
                    f"transient error fetching {uri} "
                    f"({attempt + 1}/{_FETCH_ATTEMPTS}): {exc}; retrying"
                )
                time.sleep(_FETCH_BACKOFF * (2 ** attempt))

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
    else:  # forecast from the latest *available* init cycle, horizon clipped to window
        member = FORECAST["member"]
        init, be = _latest_forecast_init(end)
        if init is not None:  # (else: nothing posted yet -> empty forecast frame)
            horizon = min(
                max((pd.Timestamp(end).normalize() - init.normalize()).days + 1, 1),
                FORECAST["horizon_days"],
            )
            base = f"{be['bucket']}/nwm.{init:%Y%m%d}/{FORECAST['product_tmpl'].format(member=member)}"
            for fhour in range(1, 24 * horizon + 1):
                fname = FORECAST["filename_tmpl"].format(
                    cycle=init.hour, member=member, fhour=fhour
                )
                files.append((f"{base}/{fname}", be))

    # One whole-file-caching filesystem per backend protocol in play, built up front
    # so the worker threads only ever read from (never mutate) the mapping.
    cache_root = cache_dir or _CACHE_DIR
    fs_by_proto = {
        proto: fsspec.filesystem(proto, **be["storage_options"])
        for proto, be in {be["protocol"]: be for _, be in files}.items()
    }

    def _cache_path(uri, proto):
        return os.path.join(cache_root, proto, uri.replace("://", "__").replace("/", "_"))

    def _subset(ds):
        fids = ds["feature_id"].values
        idx = np.searchsorted(fids, feature_ids)
        ok = fids[np.clip(idx, 0, len(fids) - 1)] == feature_ids
        da = ds["streamflow"].isel(feature_id=idx[ok])
        da = da.assign_coords(feature_id=feature_ids[ok])
        # streamflow carries only feature_id; attach the valid time so concat
        # builds a proper datetime axis.
        return da.expand_dims(time=ds["time"].values).load()

    def _one(item):
        uri, be = item
        proto = be["protocol"]
        if retain_raw:
            # Whole-file disk cache (fast reruns/backfill, but ~14 MB/file of CONUS grid).
            local = _cache_path(uri, proto)
            if not os.path.exists(local):
                try:
                    data = _cat_file(fs_by_proto[proto], uri)
                except FileNotFoundError:
                    # Absent object (e.g. an AWS rolling-window edge) -- skip, as before.
                    return None
                os.makedirs(os.path.dirname(local), exist_ok=True)
                tmp = f"{local}.{os.getpid()}.{threading.get_ident()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, local)  # atomic publish; each uri is its own file -> no races
            with open(local, "rb") as fh:
                return _subset(xr.open_dataset(fh, engine="h5netcdf"))
        # retain_raw=False: stream the file to memory, subset, discard -- keeps no bulky
        # whole-CONUS copy on disk (the storage-bounded path for operational/forecast use).
        try:
            data = _cat_file(fs_by_proto[proto], uri)
        except FileNotFoundError:
            return None
        return _subset(xr.open_dataset(io.BytesIO(data), engine="h5netcdf"))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        slabs = [s for s in ex.map(_one, files) if s is not None]
    if not slabs:
        return pd.DataFrame()
    out = xr.concat(slabs, dim="time").sortby("time")
    _, keep = np.unique(out["time"].values, return_index=True)
    out = out.isel(time=np.sort(keep))
    if resolution == "hour":
        return out.to_pandas()
    return out.resample(time="1D").mean().to_pandas()


#endregion streamflow fetch ---------------
