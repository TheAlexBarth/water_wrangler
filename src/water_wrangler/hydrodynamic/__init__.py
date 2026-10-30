from .dataset import HydrodynamicDataset


#region \- verbs ---------------

def query_hydrodynamic(model: str = "baycast", **kwargs):
    """
    Download hydrodynamic model output to disk.

    **Inputs**
        model : {"baycast"} may add more in future
        **kwargs : passed to the model (date, to_date, save_dir, ...)
    **Outputs**
        None
    """
    return HydrodynamicDataset.download(model=model, **kwargs)

def load_hydrodynamic(model: str = "baycast", **kwargs):
    """
    Open hydrodynamic model output as a HydrodynamicDataset.

    **Inputs**
        model : {"baycast"}
        **kwargs : passed to the model loader (date, to_date, local_dir, ...)
    **Outputs**
        HydrodynamicDataset
    """
    return HydrodynamicDataset.load(model=model, **kwargs)
#endregion verbs ---------------


__all__ = ["HydrodynamicDataset", "query_hydrodynamic", "load_hydrodynamic"]
