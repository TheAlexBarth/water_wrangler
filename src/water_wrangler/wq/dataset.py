# Observed water-quality (sonde) timeseries from the Water Data for Texas (WDFT) API.

import datetime as dt
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import shape
from ..core.env_timeseries import EnvTimeseries

WDFT_BASE_URL = "https://waterdatafortexas.org/coastal/api/"


#region \- internal fx ---------------

def _fmt_date(date: str | dt.date | dt.datetime | pd.Timestamp | None) -> str | None:
    """Format a date as YYYY-MM-DD for the WDFT API."""
    if date is None:
        return None
    return pd.to_datetime(date).strftime("%Y-%m-%d")
#endregion internal help fx---------------


class SondeTimeseries(EnvTimeseries):
    """
    sonde/deployment timeseries from WDfT
    """

    #region \- loaders ---------------
    @classmethod
    def load(
            cls,
            station_id: str,
            parameter_code: str,
            start_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
            end_date: str | dt.date | dt.datetime | pd.Timestamp | None = None,
            binning: str | None = None,
    ) -> "SondeTimeseries":
        """
        Load an observed water-quality timeseries for a station + parameter.

        **Inputs**
            station_id : WDFT station code
            parameter_code : parameter code (see list_stations / WDFT docs)
            start_date, end_date : optional date range
            binning : optional {"mon", "day", "hour"} aggregation
        **Outputs**
            SondeTimeseries (datetime-indexed 'value' series)
        """
        params = {
            "start_date": _fmt_date(start_date),
            "end_date": _fmt_date(end_date),
            "binning": binning,
        }
        params = {k: v for k, v in params.items() if v is not None}

        url = f"{WDFT_BASE_URL}stations/{station_id}/data/{parameter_code}"
        resp = requests.get(url, params=params)
        resp.raise_for_status()

        df = pd.DataFrame(resp.json())
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime_utc"])
            df = df.drop(columns="datetime_utc").set_index("datetime")
        data = df["value"] if "value" in df else df

        meta = {
            "station_id": station_id,
            "parameter_code": parameter_code,
            "label": parameter_code,
            "title": f"{station_id} {parameter_code}",
        }
        return cls(data, meta)

    @classmethod
    def list_stations(cls, bbox: list | None = None, all_stations: bool = False) -> gpd.GeoDataFrame:
        """
        List WDFT in-situ monitoring stations (to discover station codes).

        **Inputs**
            bbox : optional [minx, miny, maxx, maxy] spatial filter
            all_stations : include retired/inactive stations
        **Outputs**
            geopandas.GeoDataFrame of stations
        """
        params = {}
        if bbox is not None:
            params["bbox"] = bbox
        if all_stations:
            params["all_stations"] = "true"

        resp = requests.get(f"{WDFT_BASE_URL}stations", params=params)
        resp.raise_for_status()
        stations = resp.json()

        geometries = [shape(s["the_geom"]) for s in stations]
        return gpd.GeoDataFrame(stations, geometry=geometries, crs="EPSG:4326")
    #endregion loaders ---------------
