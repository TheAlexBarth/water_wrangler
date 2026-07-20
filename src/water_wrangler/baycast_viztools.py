import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
import os
import imageio.v2 as imageio
import pandas as pd
import cmocean

class VizMixin:

    #region Viz -------------------
    """

    Plotting tools:
        Userside:
        - show variable
        - show mesh
        - show quiver
        Hidden:



    """
    #region \- Full Plot --------------------
    def show_var(self, variable, tind=-1, time=None,
                     vlayer=-1,
                     show_mesh = True,
                     show_bnd = True,
                     title_prefix = "",
                     custom_title = None,
                     figsize = (10,10),
                     dpi = 300,
                     mesh_kwargs = {},
                     bnd_kwargs = {},
                     **kwargs):
        """
        Function that standardizes the process of displaying model outputs using the
        pre-defined display settings for each variable type.

        **Inputs**
            variable (str) : The data variable to be displayed. String should match one
                of the keys in the 'display_settings' attribute. 
            tind (int) : Index of the time axis to be displayed (default = last). If 'time'
                is specified, 'tind' is ignored.
            time (str) : Specific time in UTC to be displayed; format should match expectations
                of pandas.to_datetime, preferably '%Y-%m-%d %H:%M:%S'. Nearest time value returned,
                there is no interpolation performed. If specified, 'tind' is ignored.
            vlayer (int) : IF data are 3D, Index (0-46) of the vertical layer to be displayed (default = surface)
        **Outputs**
            Displays the requested model data and returns the matplotlib.pyplot.axes object.
        """
        self._ensure_mesh()

        # need to update here to reflect 
        # that it has already read in a data structure and not pulling from
        # a baycast configuration
        data = self._select_datavar(variable, tind, time, vlayer)
        
        # Generate plot
        if len(plt.get_fignums()) < 1:
            # Create new figure axes if none are open
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        else:
            ax = plt.gca()  # Otherwise grab existing

        # Display the data
        bth = ax.tripcolor(
            self.x, self.y, self.trimesh, data, zorder=2, shading='gouraud',
            vmin=self.display_settings[variable]['vmin'], 
            vmax=self.display_settings[variable]['vmax'], 
            cmap=self.display_settings[variable]['cmap'])

        # Apply figure settings
        plt.axis('scaled')
        
        ax.set_facecolor(self.display_settings[variable]['facecolor'])
        self._show_colorbar(ax, bth, **kwargs)
        if show_mesh:
            self.show_mesh(ax = ax, zorder = 3, **mesh_kwargs)
        if show_bnd:
            self._ensure_bnd()
            self.show_boundary(ax = ax, zorder = 4, **bnd_kwargs)

        # Generate an appropriate title for the image
        if custom_title is None:
            titlestr = self._create_title(variable, title_prefix, tind)
        else:
            titlestr = custom_title
        title = ax.set_title(titlestr)
        return ax
    #endregion fullplot-------------------------------------------------

    #region \- shade -------------------------------------------------
    def show_mesh(self,
                  ax = None, 
                  zorder = 3, 
                  figsize = (10,10),
                  shade = True,
                  title = None,
                  mesh_kwargs=None
                  ):
        
        self._ensure_mesh()
        self._ensure_bathy()

        if mesh_kwargs is None:
            mesh_kwargs = {}

        shade = mesh_kwargs.pop('shade', shade)
        if title is None:
            mesh_kwargs.pop('title', None)

        if ax is None:
            # Generate plot
            if len(plt.get_fignums()) < 1:
                # Create new figure axes if none are open
                fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
            else:
                ax = plt.gca()  # Otherwise grab existing
        
        if shade:
            azimuth = mesh_kwargs.pop('azimuth', 45)
            angle_altitude = mesh_kwargs.pop('angle_altitude', 70)
            alpha = mesh_kwargs.pop('alpha', 0.5)
    
            shade_area = self._hillshade(azimuth = azimuth, angle_altitude=angle_altitude)
            if shade_area is None:
                print('No bathy')
                return
                
            ax.tripcolor(self.x, self.y, self.trimesh, shade_area, zorder=zorder,
                        cmap='Greys', alpha=alpha)
            ax.set_facecolor('whitesmoke') # Add subtle background hue
        
        else:
            ax.triplot(self.x, self.y, self.trimesh, color='k', lw=0.25, zorder=zorder)

        plt.axis('scaled')
        if title is not None:
            ax.set_title(title)
        return ax


    def _hillshade(self, azimuth=315, angle_altitude=45):
        """
        Function for computing hillshade on an unstructured mesh.
        For nice-looking plots, layer it on top other variables with an alpha=0.5.
        
        Function adapted (and modified to work on an unstructured mesh) from here:
        https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py
        
        **Inputs**
            azimuth (float, optional) : Azimuth of the light source, in degrees
            angle_altitude (float, optional) : Angle altitude of the light source,
                in degrees
        **Outputs**
            shaded (np.ndarray) : Hillshade array of shape (1, number of nodes)
        """
        import warnings # Suppress MaskedArray warning
        warnings.filterwarnings("ignore", category=UserWarning)
        azimuth = 360-azimuth+90

        # Corrects for vertical-to-horizontal scale difference and adds ledge to land boundaries
        if self.bathy is None:
            return None
    
        modified_bathy = self.bathy*2e-3
      
        # Create interpolation function to get the gradient of the bathymetry
        triang = tri.Triangulation(self.x, self.y, self.trimesh)
        tri_interp = tri.LinearTriInterpolator(triang, modified_bathy)

        # Find centroid coordinates of the elements
        ele_x = (self.x[self.trimesh[:,0]] + self.x[self.trimesh[:,1]] + self.x[self.trimesh[:,2]])/3
        ele_y = (self.y[self.trimesh[:,0]] + self.y[self.trimesh[:,1]] + self.y[self.trimesh[:,2]])/3

        # Compute gradient in each element
        dzdx, dzdy = tri_interp.gradient(ele_x, ele_y)
        slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        slope = np.clip(slope, 0, np.deg2rad(20))

        # Determine aspect, with several conditionals
        aspect = np.zeros_like(dzdx)
        aspect[dzdx!=0] = np.arctan2(dzdy, -dzdx)[dzdx!=0]
        aspect[aspect<0] += 2*np.pi
        aspect[(dzdx==0)&(dzdy>0)] = np.pi/2
        aspect[(dzdx==0)&(dzdy<0)] = np.pi*2 - np.pi/2

        # Other variables
        azimuthrad = azimuth*np.pi/180
        altituderad = angle_altitude*np.pi/180

        # Compute hillshade
        shaded = ((np.cos(altituderad)*np.cos(slope)) + \
            (np.sin(altituderad)*np.sin(slope)*np.cos(azimuthrad - aspect)))
        
        ne = self.trimesh.shape[0]
        nn = self.x.size

        # element -> node
        node_vals = np.zeros(nn)
        node_counts = np.zeros(nn)

        for e in range(ne):
            i, j, k = self.trimesh[e]
            val = shaded[e]
            node_vals[i] += val
            node_vals[j] += val
            node_vals[k] += val
            node_counts[i] += 1
            node_counts[j] += 1
            node_counts[k] += 1

        valid = node_counts > 0
        node_vals[valid] /= node_counts[valid]

        # node -> element
        smoothed = np.zeros_like(shaded)
        for e in range(ne):
            i, j, k = self.trimesh[e]
            smoothed[e] = (node_vals[i] + node_vals[j] + node_vals[k]) / 3.0

        shaded = smoothed
        shaded = np.clip(shaded, 0.0, 0.85)
        return shaded




    #endregion shade

    # region \- boundary -------------------------------
    def show_boundary(self, ax=None, zorder=4, **kwargs):
        """
        Plot boundary over figure area
        """

        if ax is None:
            # Generate plot
            if len(plt.get_fignums()) < 1:
                # Create new figure axes if none are open
                fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
            else:
                ax = plt.gca()  # Otherwise grab existing

        if self.has_bnd:       
            lines = np.stack(
                [
                    np.column_stack((self.x[self.bnd_segments[:, 0]], self.y[self.bnd_segments[:, 0]])),
                    np.column_stack((self.x[self.bnd_segments[:, 1]], self.y[self.bnd_segments[:, 1]])),
                ],
                axis=1,
            )

            lc_land = LineCollection(
                lines,
                linewidths=1,
                alpha=0.9,
                zorder=zorder,
                color = 'k',
                **kwargs
            )
            ax.add_collection(lc_land)
            plt.axis('scaled')
            return ax
        else:
            return

    #endregion boundary

    #region \- quiver --------------------------------------------------
    def show_quiver(self, 
                    tind=-1,
                    custom_title = None,
                    ax=None,
                    space_factor=100, scale=22, width=0.002, headwidth=4,
                    pivot='tail', alpha=0.8,
                    currents = True,
                    cmap=None,
                    vmin=None,
                    vmax=None,
                    add_colorbar=True,
                    cbar_loc="upper left",
                    cbar_kwargs={},
                    show_bnd = True,
                    bnd_kwargs = {},
                    **kwargs
                    ):
        """
        Plot velocity components over figure area


        """
        if ax is None:
            # Generate plot
            if len(plt.get_fignums()) < 1:
                # Create new figure axes if none are open
                fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
            else:
                ax = plt.gca()  # Otherwise grab existing

        bbox = [self.x.min()+0.01, 
                self.y.min()+0.01,
                self.x.max()-0.01,
                self.y.max()-0.01]

        xmin, ymin, xmax, ymax = bbox
        x = np.linspace(xmin, xmax, space_factor)
        y = np.linspace(ymin, ymax, space_factor)

        if len(x) != len(y):
            min_len = np.min([len(x), len(y)])
            x = x[0:min_len]
            y = y[0:min_len]
         # Determine quiver magnitude
        
        if "time" in self.coords:
            plot_bds = type(self)(self.ds.isel(time = tind))
        else:
            plot_bds = self

        xx,yy = np.meshgrid(x,y)
        U = plot_bds.interpolate_to_points('u', xx.ravel(), yy.ravel())
        V = plot_bds.interpolate_to_points('v', xx.ravel(), yy.ravel())

        u_names = U.columns.values
        v_names = V.columns.values

        if not np.all(u_names == v_names):
            print('Interpolation did not perform evenly across u and v')
            if len(u_names) > len(v_names):
                U.isel[:,v_names]
            else:
                V.isel[:, u_names]
            u_names = U.columns.values
            v_names = V.columns.values
            if not np.all(u_names == v_names):
                raise ValueError("Unable to fix u and v mismatch")
        
        x = xx.ravel()[u_names]
        y = yy.ravel()[u_names]
        u = U.iloc[0,:]
        v = V.iloc[0,:]

        # set up coloring
        if currents:
            speed = np.hypot(u, v)
            if cmap is None:
                cmap = self.display_settings['currents']['cmap']
            if vmin is None:
                vmin = self.display_settings['currents']['vmin']
            if vmax is None:
                vmax = self.display_settings['currents']['vmax']
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            q = ax.quiver(x,y,u,v,
                      speed, cmap=cmap, norm=norm, 
                      zorder = 3, 
                      scale = scale,width = width, headwidth=headwidth,
                      pivot=pivot,alpha=alpha, **kwargs)
            
            try:
                self._show_colorbar(ax, q, loc=cbar_loc, **cbar_kwargs)
            except Exception:
                # fallback: standard colorbar
                plt.colorbar(q, ax=ax, fraction=0.03, pad=0.02, label="Current speed (m/s)")
        else:
            q=ax.quiver(x,y,u,v, 
                      zorder = 3, 
                      scale = scale,width = width, headwidth=headwidth,
                      pivot=pivot,alpha=alpha, **kwargs)
        
        if show_bnd:
            self._ensure_bnd()
            self.show_boundary(ax = ax, zorder = 4, **bnd_kwargs)
        plt.axis('scaled')
        if custom_title is not None:
            ax.set_title(custom_title)
        return ax
         
    #endregion quiver -----------------------------------------------
    

    #region \- plot helpers ----------------------------------------------

   


    def _show_colorbar(self, ax, mappable, loc = 'upper left', **kwargs):
        """
        Helper function to add an inset colorbar
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Generate a small colorbar inset
        if 'right' in loc:
            axins1 = inset_axes(ax, width="3%", height="20%",
                                loc=loc, borderpad = 1)
            cbar = plt.colorbar(mappable, cax=axins1)
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.yaxis.set_label_position('left')
        else:
            axins1 = inset_axes(ax, width="3%", height="20%",
                                loc=loc, borderpad = 1)
            cbar = plt.colorbar(mappable, cax=axins1)
        return
    
    def _create_title(self,
                      variable,
                      title_prefix,
                      tind):
        """
        Make a nice title for plot
        """
        titlestr = r"{title_prefix}{label} [{units}]{time}"
        
        timestr = ""
        if 'time' in self.dims:
            timestr = ' ({time})'.format(time=self['time'][tind].dt.strftime('%Y-%m-%d-H%H').values)
        if variable == 'bathy':
            timestr = ''

        titlestr = titlestr.format(
            title_prefix = title_prefix,
            label = self.display_settings[variable]['label'],
            units = self.display_settings[variable]['units'],
            time = timestr)
        return titlestr


    def _select_datavar(self, 
                        variable, 
                        tind=-1, 
                        time = None, 
                        vlayer=-1, 
                        depth_avg = False):
        """
        Helper function to slice data for plotting

        ***Inputs**
            variable (str) : one of the variables in baycastdataset
            tind (int) : index on time axis (default = last). Ingored if time provided
            time (str) : Specific time IN MODEL TIME - BE AWARE of auto converts
            vlayer (int) : Index of vertical layer to display (default = surface). Ignored if depth_avg = True
            depth_avg (bool) : if true average the depth
        ***Outputs***
            data (np.ndarray) : slice of dataarray to specific time index
        """
        has_time = 'time' in self.dims

        if has_time and time is not None:
            tind = int(np.argmin(abs(pd.to_datetime(self['time'].values)-pd.to_datetime(time))))
        
        if variable == 'currents':
            if 'nvrt' in self.dims:
                if has_time:
                    data = np.hypot(self['u'][tind,:,vlayer], self['v'][tind,:,vlayer])
                else:
                    data = np.hypot(self['u'][:, vlayer], self['v'][:, vlayer])
            else:
                if has_time:
                    data = np.hypot(self['u'][tind,:], self['v'][tind,:])
                else:
                    data = np.hypot(self['u'], self['u'])
        elif variable == 'wse':
            if has_time:
                data = self[variable][tind,:]
            else:
                data = self[variable]
        elif variable == 'bathy':
            data = self[variable]
        else:
            if 'nvrt' in self.dims:
                if has_time:
                    data = self[variable][tind,:, vlayer]
                else:
                    data = self[variable][:, vlayer]
            else:
                if has_time:
                    data = self[variable][tind, :]
                else:
                    data = self[variable]
        return data
  

    # region \- animate  -------------------------------
    def _iter_tinds(self, tind=None, interval=1):
        """Helper: normalize time indices."""
        if tind is None:
            if "time" not in self.dims:
                return [None]
            return range(0, self.sizes["time"], interval)
        # allow int, list, range, np array
        if isinstance(tind, int):
            return [tind]
        return tind


    def _gif_output_targets(self, save_gif, display, folder, gif_filename):
        """
        Decide where GIF should be written:
        - if save_gif: write to folder/gif_filename
        - elif display: write to a temporary file (lowest memory) and return path
        - else: no gif output target (None)
        """
        import os
        import tempfile

        tmpdir = None
        gif_path = None

        if save_gif:
            if folder is None:
                raise ValueError("folder must be provided when save_gif=True")
            os.makedirs(folder, exist_ok=True)
            if gif_filename is None:
                gif_filename = "animation.gif"
            gif_path = os.path.join(folder, gif_filename)
            return gif_path, tmpdir

        if display:
            tmpdir = tempfile.TemporaryDirectory()
            gif_path = os.path.join(tmpdir.name, gif_filename or "animation.gif")
            return gif_path, tmpdir

        return None, None


    def _maybe_display_gif(self, gif_path, display):
        """Display a GIF inline if running in an interactive environment."""
        if not display or gif_path is None:
            return
        try:
            from IPython.display import Image, display as ipy_display
            ipy_display(Image(filename=gif_path))
        except Exception:
            # Non-notebook environment; silently ignore
            pass

        
    def _animate_core(
        self,
        mode,
        variable=None,
        tind=None,
        interval=1,
        duration=200,         # ms per frame
        dpi=300,
        save_frames=False,
        save_gif=True,
        display=True,
        folder=".",
        gif_filename=None,
        custom_title = None,
        frame_prefix="frame",
        # kwargs passthrough
        plot_kwargs=None,
        quiver_kwargs=None,
        savefig_kwargs=None,
    ):
        """
        Core animation routine. Streams frames to GIF writer (lowest memory).
        mode:
        - "var": uses self.show_var(variable, ...)
        - "quiver": uses self.show_quiver(...)
        """
        if plot_kwargs is None:
            plot_kwargs = {}
        if quiver_kwargs is None:
            quiver_kwargs = {}
        if savefig_kwargs is None:
            savefig_kwargs = {"bbox_inches": "tight"}

        if mode not in ("var", "quiver"):
            raise ValueError("mode must be 'var' or 'quiver'")
        if mode == "var" and variable is None:
            raise ValueError("variable must be provided when mode='var'")

        # Determine where GIF will be written (real path or temp path)
        gif_path, tmpdir = self._gif_output_targets(save_gif, display, folder, gif_filename)

        # Determine frame directory (only needed if saving frames OR if using disk as intermediate)
        # For lowest-memory, we do NOT keep frames in RAM; we write a PNG, append to GIF, delete PNG unless saving.
        if save_frames:
            if folder is None:
                raise ValueError("folder must be provided when save_frames=True")
            os.makedirs(folder, exist_ok=True)
            frame_dir = folder
        else:
            # Use temp dir for intermediate PNGs if not saving frames
            # If gif_path is in a temp dir already, reuse that; else create a temp dir
            if tmpdir is not None:
                frame_dir = tmpdir.name
            else:
                import tempfile
                tmp_png_dir = tempfile.TemporaryDirectory()
                frame_dir = tmp_png_dir.name

        # Streaming writer (one frame at a time)
        writer = None
        if gif_path is not None:
            writer = imageio.get_writer(gif_path, mode="I", duration=duration / 1000.0, loop=0)

        # Iterate times
        tinds = self._iter_tinds(tind=tind, interval=interval)

        try:
            for i, ti in enumerate(tinds):
                plt.close("all")

                # --- render frame ---
                if mode == "var":
                    ax = self.show_var(variable, tind=ti, dpi=dpi, custom_title=custom_title **plot_kwargs)

                elif mode == "quiver":
                    # Build a consistent canvas
                    fig, ax = plt.subplots(1, 1, figsize=plot_kwargs.get("figsize", (10, 10)), dpi=dpi, **plot_kwargs)

                    # Now the actual thing requested: pass through show_quiver
                    ax = self.show_quiver(tind=ti if ti is not None else -1, ax=ax, custom_title=custom_title, **quiver_kwargs)

                # --- write intermediate PNG ---
                png_path = os.path.join(frame_dir, f"{frame_prefix}_{i:05d}.png")
                plt.savefig(png_path, dpi=dpi, **savefig_kwargs)
                plt.close("all")

                # --- append to gif ---
                if writer is not None:
                    writer.append_data(imageio.imread(png_path))

                # --- cleanup frame unless keeping ---
                if not save_frames:
                    try:
                        os.remove(png_path)
                    except OSError:
                        pass

        finally:
            if writer is not None:
                writer.close()

            # If we created a temp gif for display-only, display then clean up
            if display and gif_path is not None and not save_gif:
                self._maybe_display_gif(gif_path, display=True)

            if tmpdir is not None:
                tmpdir.cleanup()

        # If saved, optionally display, then return path
        if save_gif and gif_path is not None:
            self._maybe_display_gif(gif_path, display=display)
            return gif_path

        # If not saved and not displayed, nothing to return
        return None
    
        
    def animate_var(
        self,
        variable,
        tind=None,
        interval=1,
        duration=200,
        dpi=200,
        folder=None,
        gif_filename=None,
        save_frames=False,
        save_gif=True,
        display=True,
        plot_kwargs=None,
    ):
        """
        Animate a scalar field by calling self.show_var(...) for each frame.
        Mirrors the BAYCAST 'animate_outputs' spirit but streams frames (low memory).
        """
        return self._animate_core(
            mode="var",
            variable=variable,
            tind=tind,
            interval=interval,
            duration=duration,
            dpi=dpi,
            folder=folder,
            gif_filename=gif_filename or f"{variable}.gif",
            save_frames=save_frames,
            save_gif=save_gif,
            display=display,
            plot_kwargs=plot_kwargs,
        )


    def animate_quiver(
        self,
        tind=None,
        interval=1,
        duration=200,
        dpi=200,
        folder=None,
        gif_filename="quiver.gif",
        save_frames=False,
        save_gif=True,
        display=True,
        custom_title = None,
        quiver_kwargs=None,
        **plot_kwargs,
    ):
        """
        Animate currents by calling self.show_quiver(...) for each frame.
        Optionally draw a lightweight background ('mesh' or 'bathy') for context.
        """
        return self._animate_core(
            mode="quiver",
            variable=None,
            tind=tind,
            interval=interval,
            duration=duration,
            dpi=dpi,
            folder=folder,
            gif_filename=gif_filename,
            save_frames=save_frames,
            save_gif=save_gif,
            display=display,
            custom_title=custom_title,
            quiver_kwargs=quiver_kwargs or {},
            plot_kwargs=plot_kwargs,
        )

    #endregion animate outputs

    #endregion Viz