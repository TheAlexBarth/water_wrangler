from .dataset import HydrologyDataset


def query_hydrology(source: str = "nwm", **kwargs) -> "HydrologyDataset":
    """
    Query hydrologic flows as a HydrologyDataset.

    **Inputs**
        source : {"nwm", "twdb"}
        **kwargs : passed to HydrologyDataset.load (boundary, start, end, ...)
    **Outputs**
        HydrologyDataset
    """
    return HydrologyDataset.load(source=source, **kwargs)


__all__ = ["HydrologyDataset", "query_hydrology"]
