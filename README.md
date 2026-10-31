# Water Wrangler

Water Wrangler is a Python toolkit for working with TWDB Coastal Science outputs, primarily accessing and handling BAYCAST datasets. The design is to act both as an API handler for TWDB Coastal data AND provide user-friendly features for handling BAYCAST output without requiring the complete SCHISM cofiguration or baycast package.

## BAYCAST

Water Wrangler accesses BAYCAST output, which is expected to be stored as single-day output files
following a standard convention. These are all stored in the TWDB Midgewater access site for public access.

```text
https://midgewater.twdb.texas.gov/bays_estuaries/baycast/
```

Expected filename patterns:

```text
2D: {product}_{dimension}_{type_2d}_{YYYYMMDD}.nc
3D: {product}_{dimension}_{YYYYMMDD}.nc
```

Supported BAYCAST options:

```python
product = "chem" or "hydro"
dimension = "2D" or "3D"
type_2d = "depth" or "surf"
```

---


### Load BAYCAST Datasets

```python
from water_wrangler import load_hydrodynamic

baycast = load_hydrodynamic(
    model="baycast",
    date="2026-01-01",
    to_date="2026-01-07",
    product="chem",
    dimension="2D",
    type_2d="depth",
)
```

If files are available locally, use `local_dir` which will check that directory for BAYCAST output files
If `local_dir=None` (default) files are downloaded to temporary `.nc` files. Note if you want to download and *store* BAYCAST output use `query_hydrodynamic(model="baycast")`
Alternatively use `query_hydrodynamic()` to download and store locally.

---

### BAYCAST DATASETS

Once loaded, in python water_wranger constructs an object from the HydrodynamicDataset class. This object has several convenient features.

#### Mesh Preparation

Mesh topology is prepared lazily with `HydrodynamicDataset._ensure_mesh()`. This will load nv to have mesh features for plotting and formatting - like subsets or interpolation.

---
#### Subsetting Meshes
`HydrodynamicDataset.subset_mesh()`
---
#### Triangular interpolation
`HydrodynamicDataset.interpolate_to_points()`
---

#### Plotting
[need to add details here]
---

## Hydrology
### Load Hydrology
Hydrology can be loaded using `query_hydrology()` with either 'nwm' or 'twdb' for source. If `source='twdb'` the boundary can be either a watershed character or gpd/geometry. `source='nwm'` requires a geometry input for the watershed limit.

## Shapefiles
### Water Data for Texas

Use `query_wdft_shapefile()` to retrieve coastal geometry layers from Water Data for Texas.

```python
from water_wrangler import query_wdft_shapefile

bays = query_wdft_shapefile("bay")
basins = query_wdft_shapefile("basin")
estuaries = query_wdft_shapefile("estuary")
watersheds = query_wdft_shapefile("watershed")
```

Use `query_wqtimeseries()` to pull observed water-quality (sonde) timeseries from Water Data for
Texas. It returns a `SondeTimeseries` (an `EnvTimeseries`, so it has `.plot()` and
`.resample_time()`). Use `SondeTimeseries.list_stations()` to discover station codes.

```python
from water_wrangler import query_wqtimeseries, SondeTimeseries

stations = SondeTimeseries.list_stations()                 # station metadata (GeoDataFrame)
salinity = query_wqtimeseries(
    "6980", "seawater_salinity",
    start_date="2018-08-01", end_date="2018-10-01",
)                                                           # SondeTimeseries
salinity.plot()
```
---




# To add in future:
- TX Land Cover
- ngofs2
- 