from .dataset import HydrologyDataset


def query_hydrology(source: str = "nwm", **kwargs) -> "HydrologyDataset":
    """
    Query hydrologic flows as a HydrologyDataset.

    **Inputs**
        source : {"nwm", "twdb", "usgs"}
        **kwargs : passed to HydrologyDataset.load (boundary, start, end,
            resolution, ...; usgs also takes most_downstream)
    **Outputs**
        HydrologyDataset
    """
    if source == "nwm" or source == "usgs":
        if "boundary" in kwargs:
            HydrologyDataset.load(source=source, **kwargs)
        elif "boundaries" in kwargs:
            HydrologyDataset.load_many(kwargs.pop("boundaries"), source=source, **kwargs)
        else:
            raise ValueError("Must provide either `boundary` or `boundaries`")