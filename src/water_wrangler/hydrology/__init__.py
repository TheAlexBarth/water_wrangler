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
    return HydrologyDataset.load(source=source, **kwargs)


__all__ = ["HydrologyDataset", "query_hydrology"]
