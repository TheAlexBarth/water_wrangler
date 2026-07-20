# Water Wrangler

Water Wrangler is a Python toolkit for working with TWDB Coastal Science outputs, primarily accessing and handling BAYCAST datasets. The design is to act both as an API handler for TWDB Coastal data AND provide user-friendly features for handling BAYCAST output without requiring the complete SCHISM cofiguration or baycast package.

Water Wrangler supports workflows for BAYCAST-focused work including:
- Downloading BAYCAST NetCDF files from TWDB Midgewater
- Loading local or remote BAYCAST datasets
- Wrapping BAYCAST output in `BaycastDataset`
- Preparing SCHISM-style mesh topology
- Supporting mixed triangle / quadrilateral meshes
- Handling disconnected mesh fragments
- Plotting mesh-based datasets
- Interpolating over unstructured meshes
- Managing boundary topology
- Rasterizing coastal geometry layers
And querying data from Water Data for Texas including:
- coastal geometries
- (coming soon) continuous water quality 
- (coming soon) freshwater inflows datasets'


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

Examples:

```text
chem_2D_depth_20260101.nc
chem_3D_20260101.nc
```

Supported BAYCAST options:

```python
product = "chem" or "hydro"
dimension = "2D" or "3D"
type_2d = "depth" or "surf"
```

---


### Load BAYCAST Datasets

Use `load_baycast()` to open BAYCAST datasets and return a `BaycastDataset`.

```python
from water_wrangler.io import load_baycast

baycast = load_baycast(
    date="2026-01-01",
    to_date="2026-01-07",
    product="chem",
    dimension="2D",
    type_2d="depth",
)
```

If files are available locally, use `local_dir` which will check that directory for BAYCAST output files
- Missing files raise `FileNotFoundError`
- Files are opened with `xarray.open_mfdataset`


If `local_dir=None` (default) files are downloaded to temporary `.nc` files. Note if you want to download and *store* BAYCAST output use `query_baycast()`

Additional keyword arguments are passed to `xarray.open_mfdataset`:

```python
baycast = load_baycast(
    date="2026-01-01",
    to_date="2026-01-07",
    chunks={"time": 1},
)
```

---

### BAYCAST DATASETS

Once loaded, in python water_wranger constructs an object from the BaycastDataset class. This object has several convenient features.

#### Mesh Preparation

Mesh topology is prepared lazily with `BaycastDataset._ensure_mesh()`. This allows for triangulation features reading the nv connectivity matrix and formatting so SCHISM 1-based indexing is converted to Python-compatible 0-based indexing.

---

#### Subsetting Meshes
`BaycastDataset.subset_mesh()`

---

#### Triangular inter
`BaycastDataset.interpolate_to_points()`
---

#### Plotting

---

## Data Access


### Download BAYCAST Files

Use `query_baycast()` to download BAYCAST NetCDF files and store them locally.

```python
from water_wrangler.io import query_baycast

query_baycast(
    date="2026-01-01",
    to_date="2026-01-07",
    save_dir="./baycast",
    product="chem",
    dimension="2D",
    type_2d="depth",
)
```

If `to_date` is omitted, only `date` is downloaded.

Date inputs may be:

```python
str
datetime.date
datetime.datetime
pandas.Timestamp
```
---

### Water Data for Texas Geometries

Use `query_shapefile()` to retrieve coastal geometry layers from Water Data for Texas.

```python
from water_wrangler.io import query_shapefile

bays = query_shapefile("bay")
basins = query_shapefile("basin")
estuaries = query_shapefile("estuary")
watersheds = query_shapefile("watershed")
```

Allowed geometry types:

```python
"bay"
"basin"
"estuary"
"watershed"
```

Returned data are `geopandas.GeoDataFrame` objects using:

```text
EPSG:4326
```

---