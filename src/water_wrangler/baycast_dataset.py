import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from .baycast_meshtools import MeshMixin
from .baycast_viztools import VizMixin

class BaycastDataset(MeshMixin, VizMixin):
    """
    A BAYCAST specific wrapper for xarray.Dataset with BAYCAST mesh tools and plotting
    """

    def __init__(self, ds: xr.Dataset, assign_bathy = False, assign_tri = False, convert_time = False, tz = 'UTC'):

        # Underlying actual Dataset
        self.ds = ds

        # BAYCAST display settings
        self.display_settings = {
            'bathy' : {
                'cmap':'cmo.deep_r', 'vmin':-25, 'vmax':0, 'facecolor':'whitesmoke',
                'show_bnd':True, 'bnd_alpha':1, 'label':'Bathymetry Elevation', 'units':'m NAPGD2022'},
            'salinity' : {
                'cmap':'cmo.haline', 'vmin':0, 'vmax':36, 'facecolor':'whitesmoke',
                'show_bnd':True, 'bnd_alpha':1, 'label':'Salinity', 'units':'ppt',
                'quiv_cmap':'viridis_r', 'qmin':10, 'qmax':36},
            'temperature' : {
                'cmap':'RdBu_r', 'vmin':10, 'vmax':32, 'facecolor':'slategrey',
                'show_bnd':True, 'bnd_alpha':1, 'label':'Water Temperature', 'units':r'$^\circ C$',
                'quiv_cmap':'RdBu_r', 'qmin':10, 'qmax':32},
            'wse' : {
                'cmap':'cmo.ice', 'vmin':-0.4, 'vmax':0.6, 'facecolor':'gainsboro',
                'show_bnd':True, 'bnd_alpha':1, 'label':'Water Surface Elevation', 'units':'m MSL',
                'quiv_cmap':'cmo.ice_r', 'qmin':-0.4, 'qmax':0.6},
            'currents' : {
                'cmap':'magma', 'vmin':0, 'vmax':0.75, 'facecolor':'gainsboro',
                'show_bnd':False, 'bnd_alpha':1, 'label':'Current Speed', 'units':'m/s',
                'quiv_cmap':'binary', 'qmin':0.2, 'qmax':0.75},
            'w' : {
                'cmap':'seismic', 'vmin':-1e-3, 'vmax':1e-3, 'facecolor':'gainsboro',
                'show_bnd':True, 'bnd_alpha':0.2, 'label':'Vertical Velocity', 'units':'m/s'},
            'u' : {
                'cmap':'seismic', 'vmin':-0.75, 'vmax':0.75, 'facecolor':'silver',
                'show_bnd':True, 'bnd_alpha':1, 'label':'Eastward (X) Velocity', 'units':'m/s'},
            'v' : {
                'cmap':'seismic', 'vmin':-0.75, 'vmax':0.75, 'facecolor':'silver',
                'show_bnd':True, 'bnd_alpha':1, 'label':'Northward (Y) Velocity', 'units':'m/s'}
        }

        if 'sigma' in self.coords:
            self.dim = '3D'
        else:
            self.dim = '2D'
        
        if assign_bathy:
            self.has_bathy = True
            self._assign_bathybnd()
        else:
            self.has_bathy = False
        if assign_tri:
            self._assign_tri_attrs()
            self.has_tri = True
        else:
            self.has_tri = False

        if convert_time:
            self.tz = tz
            self._convert_time()
        else:
            self.tz = tz

    def _convert_time(self, tz = "America/Chicago"):
        if self.tz == tz:
            return
        else:
            time_idx = pd.DatetimeIndex(self.ds.time.values)
            self.ds = self.ds.assign_coords(
                        time=(
                            time_idx
                            .tz_localize(self.tz)
                            .tz_convert(tz)
                            .tz_localize(None)
                        )
                    )
            self.tz = tz


    #region Set up -----------------------------------------------    
    
    # Delegate all attribute access to the underlying Dataset
    def __getattr__(self, name):
        try:
            return getattr(self.ds, name)
        except AttributeError:
            raise AttributeError(f"'BaycastDataset' has no attribute '{name}'")

    def __getitem__(self, key):
        return self.ds[key]

    def __setitem__(self, key, value):
        self.ds[key] = value

    def copy(self, deep=False):
        return type(self)(self.ds.copy(deep=deep))

    def to_xarray(self):
        return self.ds


    #endregion setup 

    #region Formatting ----------------------------------
    """

    Various tools used for formatting the dataset
    Includes
        - average time
        - interpolation to points
        - io features

    """
    #region \- average time -------------------------
    def time_avg(self):
        """
        Construct new ds with mean along time axis
        """
        ds_mean = self.ds.mean(dim='time', keep_attrs=True)
        return type(self)(ds_mean)

    #endregion avgtime --------------------------------

    # region \- interpolator -----------------------------------------
    def interpolate_to_points(self, 
                              variable: str, 
                              lon: pd.Series | np.ndarray = None, 
                              lat: pd.Series | np.ndarray = None,
                              z: pd.Series | np.ndarray = None,
                              coords: np.ndarray = None,
                              coord_names: list = None):
        """
        Use triangular interpolation to match. Note that if a 3D product is used,
        z must be specified otherwise it will default to surface.
        
        ***Inputs***
            variable (str) : baycast variable which to interpolate
            lon (pd.Series or np.array) : lon
            lat (pd.Series or np.array) : lat
            coords (np array shape [,2]) : lon, lat pairs in column form OR

        """
        
        if coords is not None:
            if coords.shape[1] == 3:
                lon = coords[:,0]
                lat = coords[:, 1]
                z = coords[:,2]
            elif coords.shape[1] == 2:
                lon = coords[:,0]
                lat = coords[:, 1]
            else:
                raise ValueError("Coords must be (npoints, 2) or (npoints, 3) shaped")
        else:
            if lon is None or lat is None:
                raise ValueError("Need to specify lon and lat")
            if self.dim == '3D':
                if len(lon) != len(lat):
                        raise ValueError("mismatch in point coords")
                if z is None:
                    print('No depth dimension provided on 3D product, using surfacae')

        n_points = len(lon)

        
        var = self[variable]
            
        has_time = 'time' in var.dims
        has_nvrt = 'nvrt' in var.dims

        # Standardize dimension order for easier indexing
        if has_time and has_nvrt:
            var = var.transpose('time', 'node', 'nvrt')
        elif has_time:
            var = var.transpose('time', 'node')
        elif has_nvrt:
            var = var.transpose('node', 'nvrt')
        else:
            var = var.transpose('node')

        nt = var.sizes['time'] if has_time else 1
        nvert = var.sizes['nvrt'] if has_nvrt else 1
        
        if self.has_tri:
            triang = tri.Triangulation(self.x, self.y, self.trimesh)
        else:
            triang = tri.Triangulation(self['lon'].values, self['lat'].values, self.nv.T)
        use_surf = has_nvrt and (z is None)
        if has_nvrt and not use_surf:
            raise NotImplementedError("Depth-specific integration not built yet")
            sigma_lvl = self.ds['sigma'].values    
            z_arr = np.asarray(z)
            if z_arr.ndim == 0:
                z_arr = np.full(n_points, float(z_arr))
            elif z_arr.ndim == 1:
                if z_arr.size != n_points:
                    raise ValueError("If z is 1D, it must have length == number of points.")

            

        # unified shape
        if has_time and has_nvrt:
            data = var.values  # (time, node, nvrt)
        elif has_time and not has_nvrt:
            data = var.values[:, :, None]  # (time, node) -> (time, node, 1)
        elif not has_time and has_nvrt:
            data = var.values[None, :, :]  # (node, nvrt) -> (1, node, nvrt)
        else:
            data = var.values[None, :, None]  # (node,) -> (1, node, 1)

        nt, nnode, nvert = data.shape


        # interp loop
        all_data = np.zeros((nt, n_points), dtype='float')
       
        for ti in range(nt):
            field_t = data[ti]  # (nnode, nvert)

            if not has_nvrt:
                # Purely 2D variable: just use the single "layer"
                interp2d = tri.LinearTriInterpolator(triang, field_t[:, 0])
                vals = interp2d(lon, lat).data  # (n_points,)

            else:
                if use_surf:
                    # Surface-only: take last vertical level (consistent with show_var)
                    surface_field = field_t[:, -1]  # (nnode,)
                    interp2d = tri.LinearTriInterpolator(triang, surface_field)
                    vals = interp2d(lon, lat).data  # (n_points,)

                else:
                    # Full 3D: interpolate horizontally at each sigma, then vertically
                    vals_layers = np.empty((n_points, nvert), dtype=float)

                    for k in range(nvert):
                        layer_field = field_t[:, k]  # (nnode,)
                        interp_layer = tri.LinearTriInterpolator(triang, layer_field)
                        vals_layers[:, k] = interp_layer(lon, lat).data  # (n_points,)

                    # Now vertical interpolation at each point in sigma space
                    vals = np.empty(n_points, dtype=float)
                    for j in range(n_points):
                        vals[j] = np.interp(
                            z_arr[j],
                            sigma_lvl,
                            vals_layers[j, :],
                        )
            all_data[ti, :] = vals

        out_df = pd.DataFrame(all_data, columns = coord_names)
        keep_cols = []
        for i in range(out_df.shape[1]):
            if not np.all(np.isnan(out_df.iloc[:,i])):
                keep_cols.append(i)
        out_df = out_df.iloc[:, keep_cols]
        dropped_pts = np.arange(0,len(lon))[[i not in keep_cols for i in range(len(lon))]]
        if len(dropped_pts) > 0:
            print("Some points outside model boundary and were automatically removed")
        if has_time:
            out_df['datetime'] = self.time.values
        return out_df
    #endregion interpolate -------------------------------------------------