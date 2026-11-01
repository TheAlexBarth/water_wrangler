import pandas as pd
import matplotlib.pyplot as plt


class EnvTimeseries:
    """
    Base wrapper for an environmental timeseries (datetime-indexed values) with
    minor formatting and visualization.
    """

    def __init__(self, data: pd.DataFrame | pd.Series, meta: dict | None = None):
        self.data = data
        self.meta = meta or {}

    #region Set up -----------------------------------------------

    # Delegate attribute access to the underlying pandas object
    def __getattr__(self, name):
        if name == "data":
            raise AttributeError(name)
        try:
            return getattr(self.data, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getitem__(self, key):
        return self.data[key]

    def to_frame(self):
        return self.data
    #endregion setup

    #region Formatting ----------------------------------
    def resample_time(self, freq: str, how: str = "mean"):
        """
        Resample onto a new frequency.

        **Inputs**
            freq : pandas offset alias (e.g. 'D', 'MS')
            how : reduction to apply ('mean', 'sum', ...)
        **Outputs**
            same type, resampled
        """
        out = getattr(self.data.resample(freq), how)()
        return type(self)(out, self.meta)
    #endregion formatting

    #region Plotting ----------------------------------
    def plot(self, ax=None, **kwargs):
        """
        Line plot of the timeseries. Uses meta['label'], meta['units'], meta['title'].
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3))
        self.data.plot(ax=ax, **kwargs)
        if self.meta.get("label"):
            ax.set_ylabel(f"{self.meta['label']} ({self.meta.get('units', '')})".strip())
        if self.meta.get("title"):
            ax.set_title(self.meta["title"])
        return ax
    #endregion plotting
