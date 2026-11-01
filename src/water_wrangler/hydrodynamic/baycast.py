# BAYCAST (SCHISM) model adapter for the TWDB Midgewater portal.
# Provides DISPLAY_SETTINGS + download() + open_dataset() consumed by
# HydrodynamicDataset via the model registry.

import datetime as dt
import os
import tempfile
from pathlib import Path
from typing import List
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm


DISPLAY_SETTINGS = {
    'bathy' : {
        'cmap':'cmo.deep_r', 'vmin':-25, 'vmax':0, 'facecolor':'whitesmoke',
        'show_bnd':True, 'bnd_alpha':1, 'label':'Bathymetry Elevation', 'units':'m NAPGD2022'},
    'salinity' : {
        'cmap':'cmo.haline', 'vmin':0, 'vmax':36, 'facecolor':'whitesmoke',
        'show_bnd':True, 'bnd_alpha':1, 'label':'Salinity', 'units':'ppt',
        'quiv_cmap':'viridis_r', 'qmin':10, 'qmax':36},
    'temperature' : {
        'cmap':'RdBu_r', 'vmin':10, 'vmax':32, 'facecolor':'slategrey',
        'show_bnd':True, 'bnd_alpha':1, 'label':'Water Temperature', 'units':r'$^\circ C$',
        'quiv_cmap':'RdBu_r', 'qmin':10, 'qmax':32},
    'wse' : {
        'cmap':'cmo.ice', 'vmin':-0.4, 'vmax':0.6, 'facecolor':'gainsboro',
        'show_bnd':True, 'bnd_alpha':1, 'label':'Water Surface Elevation', 'units':'m MSL',
        'quiv_cmap':'cmo.ice_r', 'qmin':-0.4, 'qmax':0.6},
    'currents' : {
        'cmap':'magma', 'vmin':0, 'vmax':0.75, 'facecolor':'gainsboro',
        'show_bnd':False, 'bnd_alpha':1, 'label':'Current Speed', 'units':'m/s',
        'quiv_cmap':'binary', 'qmin':0.2, 'qmax':0.75},
    'w' : {
        'cmap':'seismic', 'vmin':-1e-3, 'vmax':1e-3, 'facecolor':'gainsboro',
        'show_bnd':True, 'bnd_alpha':0.2, 'label':'Vertical Velocity', 'units':'m/s'},
    'u' : {
        'cmap':'seismic', 'vmin':-0.75, 'vmax':0.75, 'facecolor':'silver',
        'show_bnd':True, 'bnd_alpha':1, 'label':'Eastward (X) Velocity', 'units':'m/s'},
    'v' : {
        'cmap':'seismic', 'vmin':-0.75, 'vmax':0.75, 'facecolor':'silver',
        'show_bnd':True, 'bnd_alpha':1, 'label':'Northward (Y) Velocity', 'units':'m/s'}
}


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


def download(
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
#MARK: OPEN DATASET
###########

def open_dataset(
    date: str | dt.date | dt.datetime | pd.Timestamp,
    to_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
    product: str = "chem",
    dimension: str = "2D",
    type_2d: str = "depth",
    local_dir: str | Path | None = None,

    **open_kwargs,
) -> xr.Dataset:
    """
    Open or download BAYCAST datasets as an xarray.Dataset

    **Inputs**
    date, to_date:
        Date range to load, if to_date empty only will use date
    product, dimension, type_2d :
        BAYCAST product selection. Must match Midgewater filename scheme.
    local_dir : str or Path or None
        - If not None: Local mode (read-only).
        - If None: Remote mode (temp-only).
    **open_kwargs :
        Extra keyword arguments passed to `xarray.open_mfdataset`.
    **Outputs**
    xarray.Dataset
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
    else:
        temp_files: List[str] = []
        sess = requests.Session()
        try:
            for dt_val in tqdm(expected_files.keys()):
                url = _midgewater_url(product, dimension, type_2d, dt_val)
                resp = sess.get(url)
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Could not download BAYCAST for {dt_val.date()} from {url}"
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix = '.nc') as tmp:
                    tmp.write(resp.content)
                    temp_files.append(tmp.name)

            if not temp_files:
                raise FileNotFoundError(
                    "No BAYCAST files could be downloaded for the requested range."
                )

            ds = xr.open_mfdataset(temp_files, data_vars="minimal", compat = 'no_conflicts', **open_kwargs)

        except Exception:
            for tmp in temp_files:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise
    return ds
