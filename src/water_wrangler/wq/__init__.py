from .dataset import SondeTimeseries


def query_wqtimeseries(station_id: str, parameter_code: str, **kwargs):
    """
    Query an observed water-quality (sonde) timeseries from Water Data for Texas.
    Use SondeTimeseries.list_stations() to discover station codes.

    **Inputs**
        station_id : WDFT station code
        parameter_code : parameter code
        **kwargs : passed to SondeTimeseries.load (start_date, end_date, binning)
    **Outputs**
        SondeTimeseries
    """
    return SondeTimeseries.load(station_id, parameter_code, **kwargs)


__all__ = ["SondeTimeseries", "query_wqtimeseries"]
