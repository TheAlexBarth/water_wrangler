# Hydrology dataset: freshwater flows within a spatial boundary.
# Holds the spatial state (reaches / watershed / gauges) and collapses to a line
# series only at a single node/gauge/watershed-exit via to_env_timeseries.
# Reference pattern for the NWM/TWDB internals: ../temp_nwm_code.

import numpy as np
import pandas as pd

from ..core.env_timeseries import EnvTimeseries
from . import nwm, twdb, usgs

# m3/s -> acre-ft (1 acre-ft = 1233.48 m3)
_CMS_TO_AFD = 86400.0 / 1233.48  # daily-mean m3/s -> acre-ft/day
_CMS_TO_AFH = 3600.0 / 1233.48  # hourly m3/s -> acre-ft/hour

_SOURCES = ("nwm", "twdb", "usgs")


class HydrologyDataset:
    """
    Freshwater-flow dataset for a watershed/estuary. `nwm` holds the NHDPlus reach
    network + per-reach NWM streamflow; `twdb` holds the TWDB TxRR inflow series
    (gauged / modeled-ungaged / reported). It is NOT an EnvTimeseries -- it becomes
    one only at a single point via `to_env_timeseries`.
    """

    def __init__(
        self, source, boundary=None, reaches=None, flow=None, gauges=None, meta=None
    ):
        self.source = source
        self.boundary = boundary  # GeoDataFrame / GeoSeries / shapely geometry
        self.reaches = reaches  # GeoDataFrame (nwm) or None (twdb)
        self.flow = flow  # nwm: date x feature_id (m3/s); twdb: inflow cols
        self.gauges = gauges  # GeoDataFrame of gauges considered (nwm or twdb)
        self.meta = meta or {}

    #region \- loaders ---------------
    @classmethod
    def load(
        cls,
        source: str = "nwm",
        boundary=None,
        start=None,
        end=None,
        **kwargs,
    ) -> "HydrologyDataset":
        """
        Load hydrologic flows for a boundary.

        **Inputs**
            source : {"nwm", "twdb", "usgs"}
            boundary : drainage boundary for NWM / USGS gauge selection (GeoDataFrame
                / GeoSeries / shapely geometry); generic (WDFT watershed, HUC8, ...)
            start, end : date range (nwm source auto-selects reanalysis / operational
                / forecast per the window and stitches across boundaries)
            resolution : "day" (default) or "hour"
            **kwargs : source kwargs nwm: crs; twdb: watershed, geo_ids;
                usgs: crs, most_downstream
        **Outputs**
            HydrologyDataset
        """
        if source not in _SOURCES:
            raise ValueError(
                f"unknown source '{source}'; choose 'nwm', 'twdb', or 'usgs'"
            )
        if source == "twdb":
            return cls._load_twdb(boundary, start, end, **kwargs)
        if source == "usgs":
            return cls._load_usgs(boundary, start, end, **kwargs)
        return cls._load_nwm(boundary, start, end, **kwargs)

    @classmethod
    def _load_nwm(
        cls,
        boundary,
        start,
        end,
        crs="EPSG:4326",
        resolution="day",
        workers=nwm._MAX_WORKERS,
        cache_dir=None,
        store=None,
        retain_raw=True,
    ):
        if boundary is None:
            raise ValueError("nwm source needs a `boundary` (GeoDataFrame or geometry)")
        if resolution not in ("day", "hour"):
            raise ValueError(f"nwm resolution must be 'day' or 'hour'; got {resolution!r}")
        reaches = nwm.select_coastal_reaches(boundary, crs=crs)
        # Fetch streamflow for flow reaches only (coastline rows carry no flow).
        flow_ids = reaches.loc[reaches["role"] != "coastline", "feature_id"].values
        flow = nwm.fetch_streamflow(
            flow_ids, start=start, end=end, resolution=resolution,
            workers=workers, cache_dir=cache_dir, store=store, retain_raw=retain_raw,
        )
        gauges = reaches.loc[
            reaches["gaged"] == 1, ["feature_id", "gage_id", "geometry"]
        ].reset_index(drop=True)
        meta = {"label": "streamflow", "units": "m³/s", "resolution": resolution}
        return cls(
            "nwm",
            boundary=boundary,
            reaches=reaches,
            flow=flow,
            gauges=gauges,
            meta=meta,
        )

    @classmethod
    def load_many(
        cls,
        boundaries,
        source: str = "nwm",
        start=None,
        end=None,
        crs="EPSG:4326",
        resolution="day",
        workers=nwm._MAX_WORKERS,
        cache_dir=None,
        store=None,
        retain_raw=True,
    ) -> dict:
        """
        Load several NWM boundaries that share one streamflow fetch.

        Selects reaches per boundary, then fetches the *union* of their feature_ids in a
        single pass - this is an I/O speed up if boundaries are provided

        NOTE: each returned dataset is independent any may contain overlapping reaches
        if the boundaries themselves overlap.

        **Inputs**
            boundaries : mapping {key: boundary} or an iterable of boundaries
            source : only "nwm" is supported (the per-file fetch is NWM-specific)
            start, end, crs, resolution, workers, cache_dir : as `load()`
        **Outputs**
            dict {key: HydrologyDataset}  (keys are 0..n-1 if an iterable was given)
        """
        if source != "nwm":
            raise ValueError("load_many is only implemented for the nwm source")
        items = list(boundaries.items()) if isinstance(boundaries, dict) else list(enumerate(boundaries))

        # identify reaches per boundary (cheap NHDPlus query); collect the union of ids
        reaches_by_key = {}
        union = set()
        for key, boundary in items:
            if boundary is None:
                raise ValueError("nwm source needs a `boundary` for every entry")
            reaches = nwm.select_coastal_reaches(boundary, crs=crs)
            reaches_by_key[key] = reaches
            union.update(
                int(i) for i in reaches.loc[reaches["role"] != "coastline", "feature_id"]
            )

        # fetch over full area
        flow = nwm.fetch_streamflow(
            np.array(sorted(union), dtype="int64"),
            start=start, end=end, resolution=resolution,
            workers=workers, cache_dir=cache_dir, store=store, retain_raw=retain_raw,
        )
        meta = {"label": "streamflow", "units": "m³/s", "resolution": resolution}

        # slice the flow into each boundary's dataset
        out = {}
        for key, boundary in items:
            reaches = reaches_by_key[key]
            ids = set(int(i) for i in reaches.loc[reaches["role"] != "coastline", "feature_id"])
            cols = [c for c in flow.columns if c in ids]
            gauges = reaches.loc[
                reaches["gaged"] == 1, ["feature_id", "gage_id", "geometry"]
            ].reset_index(drop=True)
            out[key] = cls(
                "nwm", boundary=boundary, reaches=reaches,
                flow=flow[cols], gauges=gauges, meta=dict(meta),
            )
        return out

    @classmethod
    def _load_twdb(
        cls,
        boundary,
        start,
        end,
        watershed=None,
        geo_ids=None,
        resolution="day",
    ):
        from ..shapes import query_wdft_shapefile

        # Map the unified resolution onto the TWDB data frequency.
        if resolution == "day":
            resolution = "daily"
        if resolution not in ("daily", "hour", "monthly"):
            raise ValueError(
                f"twdb resolution must be 'day', 'hour', or 'monthly'; got {resolution!r}"
            )
        if resolution == "monthly" and geo_ids is not None:
            # The monthly series is the watershed-level geography;
            raise ValueError("twdb 'monthly' is the watershed-level series; omit `geo_ids`")

        gauges = None
        if resolution == "monthly":
            if watershed is None:
                raise ValueError("twdb monthly needs a `watershed` name")
            # One fast request at the watershed level (daily needs the 24 sub-basins).
            sheds = query_wdft_shapefile("watershed")
            row = sheds[sheds["watershed"] == watershed]
            if row.empty:
                raise ValueError(f"no watershed named '{watershed}'")
            gid, poly = row["id"].iloc[0], row.geometry.union_all()
            flow = twdb.geography_timeseries(gid, start=start, end=end)
            gauges = twdb.gauges(watershed)
            if boundary is None:
                boundary = poly
            units = "acre-ft/month"
        else:
            if geo_ids is None:
                if watershed is None:
                    raise ValueError(
                        "twdb source needs `watershed` (name) or `geo_ids` if resolution != 'monthly'"
                    )
                # Sub-watershed geography ids (per-basin series) + union drainage polygon.
                subs = query_wdft_shapefile("sub_watershed")
                subs = subs[subs["watershed"] == watershed]
                if subs.empty:
                    raise ValueError(f"no sub-watersheds tagged '{watershed}'")
                geo_ids, poly = subs["id"].tolist(), subs.geometry.union_all()
                gauges = twdb.gauges(watershed)
                if boundary is None:
                    boundary = poly
            flow = twdb.inflow_timeseries(geo_ids, start=start, end=end)
            units = "acre-ft/day"
            if resolution == "hour":
                print(
                    "Warning: TWDB daily inflow spread evenly across hours "
                    "(constant-rate assumption; not a measured hourly value)."
                )
                units = "acre-ft/hour"
                # Spread each daily value evenly over its 24 hours: forward-fill the
                # daily series onto an hourly index, then divide every column by 24.
                if not flow.empty:
                    hourly_idx = pd.date_range(
                        flow.index.min(),
                        flow.index.max() + pd.Timedelta(hours=23),
                        freq="h",
                    )
                    flow = flow.reindex(hourly_idx, method="ffill") / 24.0
        meta = {
            "label": "freshwater inflow",
            "units": units,
            "title": str(watershed) if watershed else "TWDB TxRR",
            "resolution": resolution,
            "geo_ids": geo_ids,
        }
        return cls(
            "twdb", boundary=boundary, reaches=None, flow=flow, gauges=gauges, meta=meta
        )

    @classmethod
    def _load_usgs(
        cls,
        boundary,
        start,
        end,
        crs="EPSG:4326",
        resolution="day",
        most_downstream=True,
    ):
        """
        Observed USGS discharge for the gauges within a boundary.

        Discovers active discharge gauges in the boundary (NWIS site service) and,
        when `most_downstream`, keeps only the downstream-most gauge in each chain.
        Topology comes from the NWM NHDPlus.
        """
        if boundary is None:
            raise ValueError("usgs source needs a `boundary` (GeoDataFrame or geometry)")
        if resolution not in ("day", "hour"):
            raise ValueError(
                f"usgs resolution must be 'day' or 'hour'; got {resolution!r}"
            )
        reaches = nwm.select_coastal_reaches(boundary, crs=crs)
        found = usgs.find_gauges(boundary, crs=crs)

        # Downstream-most among the found sites, via the NWM gage crosswalk/topology.
        found_sites = set(found["site_no"].astype(str))
        gaged = reaches[(reaches["gaged"] == 1) & (reaches["gage_id"].astype(str).isin(found_sites))]
        dmap = dict(
            zip(
                reaches["feature_id"].astype("int64"),
                reaches["to_comid"].astype("int64"),
            )
        )
        outlets = set(nwm.downstream_most(gaged["feature_id"].astype("int64"), dmap))
        downstream_sites = set(
            gaged.loc[gaged["feature_id"].astype("int64").isin(outlets), "gage_id"].astype(str)
        )
        # Sites NWIS found but the crosswalk does not know have unknown topology; keep them.
        crosswalked = set(gaged["gage_id"].astype(str))
        downstream_sites |= found_sites - crosswalked
        found = found.copy()
        found["downstream_most"] = found["site_no"].astype(str).isin(downstream_sites)

        sites = sorted(downstream_sites) if most_downstream else sorted(found_sites)
        flow = usgs.fetch_discharge(sites, start=start, end=end, resolution=resolution)
        meta = {"label": "streamflow", "units": "m³/s", "resolution": resolution}
        return cls(
            "usgs",
            boundary=boundary,
            reaches=reaches,
            flow=flow,
            gauges=found,
            meta=meta,
        )

    #endregion loaders ---------------

    #region \- processing ---------------
    def _sum_flow(self, feature_ids) -> pd.Series:
        """Sum daily streamflow (m3/s) over the given reaches."""
        cols = [c for c in self.flow.columns if c in set(feature_ids)]
        if not cols:
            return pd.Series(0.0, index=self.flow.index)
        return self.flow[cols].sum(axis=1)

    def _downstream_most(self, gaged):
        """Drop reaches nested upstream of another in the set (keep downstream-most)."""
        dmap = dict(
            zip(
                self.reaches["feature_id"].astype("int64"),
                self.reaches["to_comid"].astype("int64"),
            )
        )
        return nwm.downstream_most(gaged, dmap)

    def gaged_outlets(self, subtract=None):
        """
        Gaged feature_ids subtracted when computing ungaged inflow -- the
        downstream-most gaged reaches only (nested upstream gages are dropped so a
        gaged reach's flow is not subtracted twice).

        **Inputs**
            subtract : optional iterable of USGS site numbers to restrict to (e.g. the
                gauges a hydrodynamic model ingests); default = all crosswalked gauges
        **Outputs**
            list of feature_ids
        """
        gaged = self.reaches[self.reaches["gaged"] == 1]
        if subtract is not None:
            sites = set(str(s) for s in subtract)
            gaged = gaged[gaged["gage_id"].astype(str).isin(sites)]
        return self._downstream_most(gaged["feature_id"].astype("int64"))

    def inflow_reaches(self, subtract=None) -> "pd.DataFrame":
        """
        Reach table behind the NWM ungaged calc, for review.

        Returns the flow reaches (coastline excluded) with a `subtracted` column
        flagging the gauges removed from the terminal total. Ungaged inflow is
        Σ(terminal flow) − Σ(flow at `subtracted` reaches).

        **Inputs**
            subtract : optional iterable of USGS site numbers (see gaged_outlets)
        **Outputs**
            GeoDataFrame (feature_id, role, gaged, gage_id, subtracted, geometry, ...)
        """
        if self.source != "nwm":
            raise ValueError("inflow_reaches is only available for the nwm source")
        out = self.reaches[self.reaches["role"] != "coastline"].copy()
        outlets = set(self.gaged_outlets(subtract=subtract))
        out["subtracted"] = out["feature_id"].isin(outlets)
        return out.reset_index(drop=True)

    def ungaged_inflow(self, subtract=None) -> pd.Series:
        """
        Ungaged freshwater inflow (acre-ft per resolution).

        nwm  : Σ(terminal) − Σ(subtracted gauges) streamflow, converted per resolution.
        twdb : the TxRR `model` column (already modeled ungaged, in the loaded units).

        **Inputs**
            subtract : optional iterable of USGS site numbers to subtract instead of all
                crosswalked gauges (e.g. the set your hydrodynamic model ingests)
        **Outputs**
            date/timestamp-indexed pandas Series
        """
        if self.source == "twdb":
            return self.flow.get("model", pd.Series(dtype=float)).rename("ungaged")
        factor = _CMS_TO_AFH if self.meta.get("resolution") == "hour" else _CMS_TO_AFD
        terminal = self.reaches.loc[self.reaches["role"] == "terminal", "feature_id"]
        total = self._sum_flow(terminal) * factor
        gaged = self._sum_flow(self.gaged_outlets(subtract=subtract)) * factor
        return (total - gaged).clip(lower=0).rename("ungaged")

    def adjust_with_gauges(self, add=None, subtract=None) -> "EnvTimeseries":
        """
        Blend observed USGS gauge flows into the ungaged watershed-exit series.

        **Inputs**
            add, subtract : iterables of USGS site numbers to add / subtract
        **Outputs**
            EnvTimeseries of the adjusted inflow (acre-ft per resolution)
        """
        hourly = self.meta.get("resolution") == "hour"
        factor = _CMS_TO_AFH if hourly else _CMS_TO_AFD
        res = "hour" if hourly else "day"
        base = self.ungaged_inflow()
        sites = list(add or []) + list(subtract or [])
        obs = self._observed_discharge(sites, base.index, resolution=res)
        for s in add or []:
            base = base.add(obs[str(s)] * factor, fill_value=0)
        for s in subtract or []:
            base = base.sub(obs[str(s)] * factor, fill_value=0)
        meta = {
            "label": "freshwater inflow (gauge-adjusted)",
            "units": "acre-ft/hour" if hourly else "acre-ft/day",
            "title": self.meta.get("title", ""),
        }
        return EnvTimeseries(base.clip(lower=0).rename("ungaged_adj"), meta)

    def _observed_discharge(self, sites, index, resolution="day"):
        """
        Observed USGS discharge (m3/s) for the sites, aligned to `index`.

        """
        start = index.min().strftime("%Y-%m-%d") if len(index) else None
        end = index.max().strftime("%Y-%m-%d") if len(index) else None
        obs = usgs.fetch_discharge(sites, start=start, end=end, resolution=resolution)
        for s in sites:
            # fetch_discharge drops sites with no data for the window; treat an
            # absent column as fully missing so the error below is informative.
            col = obs[str(s)] if str(s) in obs.columns else pd.Series(dtype=float)
            missing = int(col.reindex(index).isna().sum())
            if missing:
                raise ValueError(
                    f"gauge {s} is missing {missing} USGS observation(s) over the "
                    "window; cannot use observed data without silently dropping flow"
                )
        return obs

    #endregion processing ---------------

    #region \- env timeseries bridge ---------------
    def to_env_timeseries(
        self, kind: str, id=None, subtract=None, gauge_source="modeled"
    ) -> "EnvTimeseries":
        """
        Collapse to a single-point line-series as an EnvTimeseries.

        **Inputs**
            kind : {"total", "gauge", "reach"/"node"}
            id   : gauge id (kind="gauge") or feature_id (kind="reach"/"node")
            subtract : for kind="total" (nwm), optional USGS site numbers to
                subtract instead of all crosswalked gauges (see ungaged_inflow)
            gauge_source : for kind="total" (nwm), {"modeled", "observed"}. Defaults to NWM but can
                replace with USGS obs
        **Outputs**
            EnvTimeseries
        """
        if gauge_source not in ("modeled", "observed"):
            raise ValueError(
                f"gauge_source must be 'modeled' or 'observed'; got {gauge_source!r}"
            )
        if kind == "total":
            if self.source == "twdb":
                meta = {
                    "label": "freshwater inflow",
                    "units": self.meta.get("units", "acre-ft/day"),
                    "title": self.meta.get("title", "TWDB TxRR inflow"),
                }
                data = self.flow
            elif self.source == "usgs":
                # Observed inflow = sum of the downstream-most gauges' discharge.
                factor = (
                    _CMS_TO_AFH if self.meta.get("resolution") == "hour" else _CMS_TO_AFD
                )
                dsm = self.gauges.loc[self.gauges["downstream_most"], "site_no"].astype(str)
                cols = [c for c in self.flow.columns if str(c) in set(dsm)]
                total = (
                    self.flow[cols].sum(axis=1) * factor
                    if cols
                    else pd.Series(dtype=float)
                )
                units = "acre-ft/hour" if self.meta.get("resolution") == "hour" else "acre-ft/day"
                data = total.rename("total")
                meta = {
                    "label": "freshwater inflow",
                    "units": units,
                    "title": "USGS observed inflow at gauges",
                }
            else:
                factor = (
                    _CMS_TO_AFH if self.meta.get("resolution") == "hour" else _CMS_TO_AFD
                )
                terminal = self.reaches.loc[self.reaches["role"] == "terminal", "feature_id"]
                total = self._sum_flow(terminal) * factor
                if gauge_source == "observed":
                    # Substitute observed USGS discharge at the gaged outlets: the gauged
                    # term is the real gauge reading, total = modeled ungaged + observed.
                    outlets = self.gaged_outlets(subtract=subtract)
                    g2s = dict(
                        zip(
                            self.reaches["feature_id"].astype("int64"),
                            self.reaches["gage_id"].astype(str),
                        )
                    )
                    sites = sorted({g2s[int(o)] for o in outlets if g2s.get(int(o))})
                    res = "hour" if self.meta.get("resolution") == "hour" else "day"
                    ungaged = (total - self._sum_flow(outlets) * factor).clip(lower=0)
                    if sites:
                        obs = self._observed_discharge(sites, self.flow.index, resolution=res)
                        gaged = obs.reindex(self.flow.index)[sites].sum(axis=1) * factor
                    else:
                        gaged = pd.Series(0.0, index=self.flow.index)
                    total = ungaged + gaged
                    title = "NWM inflow at river mouths (observed gauges)"
                else:
                    gaged = self._sum_flow(self.gaged_outlets(subtract=subtract)) * factor
                    # Gauged inflow cannot exceed the total reaching the mouths; capping it
                    # keeps gauged + ungauged == total exactly (a day's gauge mean can briefly
                    # top the routed terminal flow). ungaged == max(total - gauged, 0) either way.
                    gaged = gaged.clip(upper=total)
                    ungaged = total - gaged
                    title = "NWM inflow at river mouths"
                data = pd.DataFrame(
                    {"total": total, "gaged": gaged, "ungaged": ungaged}
                )
                units = "acre-ft/hour" if self.meta.get("resolution") == "hour" else "acre-ft/day"
                meta = {
                    "label": "freshwater inflow",
                    "units": units,
                    "title": title,
                }
        elif kind == "gauge":
            if self.source != "nwm":
                raise ValueError("kind='gauge' is only available for the nwm source")
            match = self.reaches.loc[
                self.reaches["gage_id"].astype(str) == str(id), "feature_id"
            ]
            if match.empty:
                raise ValueError(f"no gaged reach with gage_id '{id}'")
            meta = {"label": "streamflow", "units": "m³/s", "title": f"gauge {id}"}
            data = self._sum_flow(match).rename(str(id))
        elif kind in ("reach", "node"):
            if self.source != "nwm":
                raise ValueError("kind='reach' is only available for the nwm source")
            fid = int(id)
            if fid not in self.flow.columns:
                raise ValueError(f"feature_id {fid} not in fetched streamflow")
            meta = {"label": "streamflow", "units": "m³/s", "title": f"reach {fid}"}
            data = self.flow[fid].rename(fid)
        else:
            raise ValueError(f"unknown kind '{kind}'")
        return EnvTimeseries(data, meta)

    # short helper
    def total_inflow(self, gauge_source="modeled", subtract=None):
        """
        Total freshwater inflow as an EnvTimeseries.

        For the nwm source, `gauge_source="observed"` substitutes real USGS
        discharge at the gaged outlets into the total (see to_env_timeseries).
        """
        return self.to_env_timeseries(
            "total", subtract=subtract, gauge_source=gauge_source
        )

    #endregion env timeseries bridge ---------------

    #region \- spatial plots ---------------
    def _basemap(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        return ax

    def plot_watershed(self, ax=None, **kwargs):
        """Plot the drainage boundary polygon."""
        import geopandas as gpd

        ax = self._basemap(ax)
        b = self.boundary
        if isinstance(b, gpd.GeoDataFrame):
            gs = b.geometry
        elif isinstance(b, gpd.GeoSeries):
            gs = b
        else:
            gs = gpd.GeoSeries([b], crs="EPSG:4326")
        gs.plot(
            ax=ax,
            color=kwargs.pop("color", "#f0f0e0"),
            edgecolor=kwargs.pop("edgecolor", "#ccccaa"),
            lw=kwargs.pop("lw", 0.5),
            **kwargs,
        )
        return ax

    def plot_coastline(self, ax=None, **kwargs):
        """Plot the NHDPlus coastline (shoreline) for estuary delineation."""
        if self.reaches is None:
            return self._basemap(ax)
        ax = self._basemap(ax)
        coast = self.reaches[self.reaches["role"] == "coastline"]
        if len(coast):
            coast.plot(
                ax=ax,
                color=kwargs.pop("color", "0.5"),
                lw=kwargs.pop("lw", 0.8),
                label="coastline",
                **kwargs,
            )
        return ax

    def plot_reaches(self, ax=None, **kwargs):
        """Plot flowlines with a small dot marking each terminal pour-point."""
        if self.reaches is None:
            raise ValueError("reach plots require the nwm source")
        ax = self._basemap(ax)
        flow = self.reaches[self.reaches["role"].isin(["contributing", "terminal"])]
        flow.plot(ax=ax, color="#9ecae1", lw=0.4, alpha=0.7)
        term = flow[flow["role"] == "terminal"]
        if len(term):
            pts = term.geometry.representative_point()
            ax.scatter(
                pts.x,
                pts.y,
                s=kwargs.pop("s", 12),
                c="#2171b5",
                zorder=5,
                label="river mouth",
                **kwargs,
            )
        return ax

    def plot_gauges(self, ax=None, annotate=True, **kwargs):
        """Plot the gauges considered (nwm crosswalk or twdb inflow gauges)."""
        ax = self._basemap(ax)
        g = self.gauges
        if g is None or len(g) == 0:
            return ax
        p = g.geometry.representative_point()
        ax.scatter(
            p.x,
            p.y,
            s=kwargs.pop("s", 40),
            marker="o",
            c="#cb181d",
            edgecolor="k",
            lw=0.5,
            zorder=6,
            label="gauge",
            **kwargs,
        )
        if annotate:
            idcol = next((c for c in ("site_no", "gage_id") if c in g.columns), None)
            if idcol:
                for pt, val in zip(p, g[idcol]):
                    if str(val):
                        ax.annotate(
                            str(val),
                            (pt.x, pt.y),
                            fontsize=7,
                            xytext=(3, 3),
                            textcoords="offset points",
                        )
        return ax

    def plot_map(self, ax=None, gauges=False):
        """
        Compose watershed + coastline + reaches on one axis.

        **Inputs**
            gauges : also overlay the gauge dots (off by default)
        """
        ax = self.plot_watershed(ax=ax)
        if self.reaches is not None:
            self.plot_coastline(ax=ax)
            self.plot_reaches(ax=ax)
        if gauges:
            self.plot_gauges(ax=ax)
        ax.legend(loc="lower left", fontsize=8)
        return ax

    #endregion spatial plots ---------------
