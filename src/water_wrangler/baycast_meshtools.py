from zipfile import Path

import rasterio
from scipy.spatial import cKDTree
from scipy import ndimage
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import numpy as np
from importlib.resources import files, as_file
from pathlib import Path
from contextlib import contextmanager
import xarray as xr

import math

class MeshMixin:
    
    #region \- subset to presets -------------------------------------
    def subset_to_area(self, 
                       area: str, 
                       use_mask = True,
                       mask_kwargs = None
                       ):
        """
        Subset to a pre-defined area bounding box

        **Inputs**
            area (str) : name of subregion for filtering
        **Outputs**
            None - reduces dataset to smaller region
        """
        if mask_kwargs is None:
            mask_kwargs = {}
        ndist = mask_kwargs.get('neighbor_dist', 200)
        if use_mask:
            poss_areas = {
                'texas': '00-full_estuary.tif',
                'sabine-neches':  '00-sabine_neches.tif',
                'trinity-sanjacinto': '00-trinity_sanjacinto.tif',
                'colorado-lavaca': "00-colorado_lavaca.tif",
                'guadalupe': "00-guadalupe.tif",
                'mission-aransas': '00-mission_aransas.tif',
                'nueces': "00-nueces.tif",
                'upper-lagunamadre': '00-upper_laguna_madre.tif',
                'lower-lagunamadre': '00-lower_laguna_madre.tif'
            }
            if area not in poss_areas.keys():
                raise ValueError(f"{area} not in preset regions; {[a for a in poss_areas.keys()]}")
            else:
                mask_name = poss_areas[area]
                with self._data_resource_path(mask_name) as mask_path:
                    return self.subset_with_mask(mask_file = str(mask_path), neighbor_dist=ndist)
        else:
            poss_areas = {
                'texas': [-97.85, 25.8, -93.5, 30.2],
                'sabine-neches':  [-94.2, 29.5, -93.6, 30.2],
                'trinity-sanjacinto': [-95.3, 28.95, -94.3, 29.95],
                'colorado-lavaca': [-96.75, 28.2, -95.6, 28.9],
                'guadalupe': [-97, 28.05, -96.25, 28.7],
                'mission-aransas': [-97.3, 27.77, -96.8, 28.3],
                'nueces': [-97.6, 27.55, -96.85, 28.05],
                'upper-lagunamadre': [-97.82, 26.9, -97.1, 27.75],
                'lower-lagunamadre': [-97.55, 25.9, -97.05, 26.85]
            }
            if area not in poss_areas.keys():
                raise ValueError(f"{area} not in preset regions; {[a for a in poss_areas.keys()]}")
            else:
                return self.subset_to_bbox(bbox = poss_areas[area])
    #endregion subset presets -----------------------------------------------

    #region \- subset engines ----------------------------------------------------
    def subset_with_mask(self, 
                         mask_file: str = None, 
                         mask: dict = None,
                         neighbor_dist: int = 50
                         ):
        """
        Subset Baycast Dataset with a mask. Can provide a mask file or live object
        However, both must have a height, width, and trans (transformation) to proper coords

        **Inputs**
            mask_file (str) : path to
            mask (dict) : dictionary with keys 'height', 'width', 'trans', and 'raster' (2D array of mask values)
            neighbor_dist (int) : distance to include outside mask boundary (useful for catching boundary nodes in unstructured mesh)
        """
        if mask_file:
            with rasterio.open(mask_file) as src:
                height = src.height
                width = src.width
                trans = src.transform
                raster = src.read(1)            
                crs = src.crs
        elif mask:
            if "profile" in mask.keys():
                height = mask['profile']['height']
                width = mask['profile']['width']
                trans = mask['profile']['transform']
                raster = mask['raster']            
                crs = mask["profile"].get("crs", None)
            else:
                height = mask['height']
                width = mask['width']
                trans = mask['trans']
                raster = mask['raster']
                crs = None
        else:
            raise ValueError('must provide mask_file or mask')
        
        xs = np.asarray(self.lon.values)
        ys = np.asarray(self.lat.values)
        
        # rasterio expects (x, y) == (lon, lat) for EPSG:4326
        rows, cols = rasterio.transform.rowcol(trans, xs, ys)

        rows = np.asarray(rows)
        cols = np.asarray(cols)

        keep = np.zeros(xs.size, dtype=bool)
        
   
        pw = trans.a * math.cos(math.radians(trans.f)) * math.pi / 180 * 6371008.8 #pixel width

        pixel_dist = int(neighbor_dist / pw)
        
        if pixel_dist > 0:
            print(f"Using soft mask with {neighbor_dist}m inclusion distance")
            R = int(pixel_dist)

            # binary mask
            m = (raster > 0)

            # Use a disk footprint (closer to "distance" than a square)
            yy, xx = np.ogrid[-R:R+1, -R:R+1]
            footprint = (xx*xx + yy*yy) <= (R*R)

            
            m = (raster > 0)
            # distance in pixels from each pixel to nearest True pixel
            dist = ndimage.distance_transform_edt(~m)  # distance to mask
            m_soft = dist <= R

            # sample once for all nodes (with bounds check)
            inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            keep = np.zeros(xs.size, dtype=bool)
            keep[inside] = m_soft[rows[inside], cols[inside]]
        else:
            # hard mask (no neighbor expansion)
            inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            keep = np.zeros(xs.size, dtype=bool)
            keep[inside] = raster[rows[inside], cols[inside]] > 0


        idx = np.where(keep)[0]

        return self._subset_to_idx(idx)

    

    def subset_to_bbox(self, bbox: list):
        """
        From a bounding box, subset only nodes within

        **Inputs**
            bbox (list): [xmin, ymin, xmax, ymax]
        **Outputs**
            None - updates self to smaller mesh area
        """
        xmin, ymin, xmax, ymax = bbox
        in_box = np.ones_like(self.lat.values, dtype='int')==1
        in_box[self.lat.values < ymin] = False
        in_box[self.lat.values > ymax] = False
        in_box[self.lon.values < xmin] = False
        in_box[self.lon.values > xmax] = False

        in_box = np.where(in_box)[0]

        return self._subset_to_idx(in_box)
    #endregion subset engines ------------------------------
    #region \- Subset driver ------------------------------------------
    def _subset_to_idx(self, 
                       idx: np.ndarray,
                       drop_loners: bool = True):
        """
        Similar to baycast's subset_mesh

        **Inputs**
            drop_loner (bool): if any nodes are alone drop them (defaul True)
        **Returns**
            Updates object to have less nodes
        """
        ncdf = self.ds

        def _sub_idx_recursive(sub_ncdf, idx):
            
            # get connectivity matrix
            mesh_name, mesh = self._get_mesh_var(sub_ncdf)
            nconn, nelem = mesh.dims
            conn = mesh.values
            
            # reindex to python index for matching node idx
            conn, valid = self._reidx_conn(conn)
            
            # mask for all elements
            in_keep = np.isin(conn, idx)
            elem_mask = np.all((~valid) | in_keep, axis = 0)
            conn = conn[:, elem_mask]

            # map indices old to new
            n_nodes = sub_ncdf.sizes['node']
            node_map = -np.ones(n_nodes, dtype = 'int')
            node_map[idx] = np.arange(len(idx)) # set keep nodes to new indexing
            conn_new = conn.copy()
            valid_new = conn_new >= 0
            conn_new[valid_new] = node_map[conn_new[valid_new]]

            pre_isel_ds = sub_ncdf

            sub_ncdf = sub_ncdf.isel({"node": idx})
            sub_ncdf[mesh_name] = ((nconn, nelem), conn_new)
            sub_ncdf = self._subset_bnd(pre_isel_ds, sub_ncdf, node_map)

            if drop_loners:
                kept_nodes = np.unique(conn_new[conn_new >= 0])
                all_nodes = np.arange(sub_ncdf.sizes['node'])
                if kept_nodes.size < all_nodes.size:
                    return _sub_idx_recursive(sub_ncdf, kept_nodes)
            return sub_ncdf
                
        sub_ncdf = _sub_idx_recursive(ncdf, idx)
        new_obj = self._spawn(sub_ncdf)
        return new_obj
    
    def _subset_bnd(self, pre_isel_ds, post_isel_ds, node_map):
        """
        pre_isel_ds : dataset before node isel
        post_isel_ds: dataset after node isel
        node_map    : old->new node index map (len = pre_isel_ds.node)
        """
        if "bnd" not in pre_isel_ds.data_vars:
            return post_isel_ds

        bnd = pre_isel_ds["bnd"].values.copy()

        # convert to 0-based if needed
        if bnd[:, 0].min() == 1:
            bnd[:, 0:2] -= 1

        n1 = bnd[:, 0].astype(int)
        n2 = bnd[:, 1].astype(int)

        keep = (node_map[n1] >= 0) & (node_map[n2] >= 0)
        if not np.any(keep):
            # no surviving segments
            return post_isel_ds.drop_vars("bnd")

        bnd_new = bnd[keep].copy()
        bnd_new[:, 0] = node_map[bnd_new[:, 0].astype(int)]
        bnd_new[:, 1] = node_map[bnd_new[:, 1].astype(int)]

        # preserve dims if present, else use ('nbnd','four')
        if "bnd" in post_isel_ds.data_vars:
            dims = post_isel_ds["bnd"].dims
        else:
            dims = ("nbnd", "four")

        return post_isel_ds.assign(bnd=(dims, bnd_new))
    
    #endregion subset driver
    #region \- Mesh Helpers ------------------------------------------
    def _get_mesh_var(self, ds=None):
        if ds is None:
            ds = self.ds
        
        poss_names = ['nv', 'mesh']
        for name in poss_names:
            if name in ds.data_vars and ds[name].ndim == 2:
                return name, ds[name]
        raise KeyError("cannot fine connectivity var named with 'nv' or 'mesh'")

    def _reidx_conn(self, conn):
        """
        Check connectivity array for 0-based indexing

        ***Inputs***
            conn (array) : connectivity array ('nv' or 'mesh')
        ***Outputs***
            conn (array) : reindexed if neded
            valid (array) : boolean set of valid conn nodes
        """
        # reindex to python index for matching node idx
        valid = conn >= 0
        if not np.any(valid):
            raise ValueError('Connectivity array has no valid indices')
        min_valid = conn[valid].min()
        if min_valid == 1:
            conn = conn.copy()
            conn[valid] -= 1
        return conn, valid
    
    def _match_nearest(self, coords1, coords2, unique = True):
        """
        Helper function used to match set 1 of coordinates to the nearest
        points in set 2. Presumes 2 > 1 so that all points in 2 are assigned nearest in 1

        **Input**
            coords1 (np.array, [:,2] lon,lat set) : coordinates which to set match
            coords2 (np.array, [:,2] lon,lat set) : coordinates to assign to set 1
            (optional):
            unique (boolean) : if true, will only return one instance of idx
        **Output**
            idx (np.array,]) : index of coords1
        """

        # match to nearest
        mesh_tree = cKDTree(coords1)
        _, idx = mesh_tree.query(
            coords2, k=1
        )
        if unique:
            idx = np.unique(idx)
        return idx

    # endregion meshhelpers

    # region \- mesh structures --------------------------------------
    def _prep_mesh(self):
        """
        Build mesh-derived topology objects from
        connectivity stored in self.ds.
        """

        self.x = self.lon.values
        self.y = self.lat.values

        _, mesh = self._get_mesh_var()

        conn = mesh.values
        conn, valid = self._reidx_conn(conn)

        nconn_dim, nelem = mesh.dims

        nconn_elem = valid.sum(axis=0)
        i34 = np.clip(nconn_elem, 3, 4).astype(int)

        tris1 = conn[:3, :].T

        is_quad = (i34 == 4)

        if nconn_dim == "four" and np.any(is_quad):

            conn_quad = conn[:, is_quad]

            tris2 = np.column_stack([
                conn_quad[0],
                conn_quad[2],
                conn_quad[3]
            ])

            trimesh = np.vstack([tris1, tris2])

        else:
            trimesh = tris1

        self.nn = len(self.x)
        self.ne = nelem

        self.conn = conn
        self.i34 = i34
        self.trimesh = trimesh

    def _prep_bnd(self):
        """
        Ensure boundary information exists and
        prepare plotting segments.
        """

        if "bnd" not in self.data_vars:
            with self._data_resource_path("bnd.nc") as bnd_file:
                bnd_ds = xr.open_dataset(bnd_file)
            self.ds = self.ds.assign(
                bnd=bnd_ds["bnd"]
            )

        bnd = self.ds["bnd"].values.copy()

        if bnd[:, 0].min() == 1:
            bnd[:, 0:2] -= 1

        n1 = bnd[:, 0].astype(int)
        n2 = bnd[:, 1].astype(int)

        x1 = self.x[n1]
        y1 = self.y[n1]

        x2 = self.x[n2]
        y2 = self.y[n2]

        dx = (x2 - x1) * np.cos(np.deg2rad(0.5 * (y1 + y2)))
        dy = y2 - y1

        dist_km = 111.0 * np.sqrt(dx**2 + dy**2)

        keep = dist_km <= 20

        self.bnd = bnd

        self.bnd_segments = np.column_stack([
            n1[keep],
            n2[keep]
        ])
        self._bnd_prepped = True

    def _prep_bathy(self):
        """
        Ensure bathymetry information exists and
        prepare plotting segments.
        """

        if "bathy" not in self.data_vars:

            with self._data_resource_path("bathy.nc") as bathy_file:
                bthy_ds = xr.open_dataset(bathy_file)

            self.ds = self.ds.assign(
                bathy=bthy_ds["bathy"]
            )

        self.bathy = self.ds["bathy"].values
    
    # endregion mesh structures -------------------------------

    # region \- ensure mesh structures --------------------------------------
    def _ensure_mesh(self):
        if not getattr(self, '_mesh_prepped', False):
            self._prep_mesh()
            self._mesh_prepped = True

    def _ensure_bnd(self):
        self._ensure_mesh()
        if not hasattr(self, 'bnd_segments'):
            self._prep_bnd()

    def _ensure_bathy(self):
        if not hasattr(self, 'bathy'):
            self._prep_bathy()


    # endregion mesh structures
    def _spawn(self, ds):
        obj = type(self)(ds, tz = self.tz)
        return obj
    

    @contextmanager
    def _data_resource_path(self, filename: str):
        
        """
        Return a context manager yielding a real filesystem Path to a packaged data file.

        This is required because wheels may store package data inside a zip-like
        structure, and rasterio expects a real on-disk path.
        """
        rsc = files('water_wrangler') / "data" /filename
        with as_file(rsc) as p:
            yield Path(p)
            
    #endregion MeshTool -------------------------------
