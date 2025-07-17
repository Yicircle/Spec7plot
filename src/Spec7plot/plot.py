import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import colorsys
import seaborn as sns
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import astropy.wcs as wcs
from astropy.wcs import WCSSUB_LONGITUDE, WCSSUB_LATITUDE

class plot:
    def __init__(self):
        """
        Initialize the plot class.
        This class is used to create plots for spectral data.
        """
        try:
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
        except:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
    
    @staticmethod
    def find_rec(N):
        # Start from the square root of N and work downwards
        num_found = False
        while ~num_found:
            for k in range(int(N ** 0.5), 0, -1):
                if N % k == 0:  # k must divide N
                    l = N // k  # Calculate l
                    # Check the condition that neither exceeds twice the other
                    if k <= 2 * l and l <= 2 * k:
                        num_found = True
                        return k, l
            N = N + 1
        return None, None  # Return None if no valid pair is found
    
    @staticmethod
    def makeSpecColors(n: int = 40, 
                       palette:str = 'Spectral_r'
                       ) -> list:
        
        """
        Create a list of colors for plotting spectra.

        Parameters
        ----------
        n : int, optional
            Number of colors to generate. Default is 40.
        palette : str, optional
            Name of the color palette. Default is 'Spectral_r'.

        Returns
        -------
        clist : list
            List of colors for plotting.
        """
        
        # Color palette
        cmap = sns.color_palette(palette, as_cmap=True)

        # List of colors
        clist = [cmap(i / (n - 1)) for i in range(n)]
        return clist
    
    @staticmethod
    def lighten_hls(color, amount=0.8):
        """
        Convert RGB to HLS, increase lightness by amount, convert back.
        amount: fraction of remaining lightness to add [0,1].
        """
        # 1) Seperate RGB (ignore alpha)
        r, g, b, a = color
        # 2) RGB → HLS
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # 3) Increase Lightness (cap at 1.0)
        l_l = min(1.0, l + amount * (1 - l))
        # 4) HLS → RGB
        r_l, g_l, b_l = colorsys.hls_to_rgb(h, l_l, s)
        return (r_l, g_l, b_l, a)
    
    @staticmethod
    def makeSepcCmaps(n:int = 40,
                      lightness:str = 'dark'
                      ) -> list:
        colours = [plot.lighten_hls(c, amount=0.95) for c in plot.makeSpecColors(n)]
        colours = [matplotlib.colors.rgb2hex(tuple(c), keep_alpha=True) for c in colours]
        
        if (lightness != 'light') & (lightness != 'dark'):
            raise ValueError("Lightness should be either 'light' or 'dark'.")

        cmaps = [sns.color_palette(f"{lightness}:{c}", as_cmap=True) for c in colours]
        return cmaps

    @classmethod
    def SED(self,
            cube: str | Path,
            output_dir: str | Path = None,
            output_name: str = None,
            pixel_loc: tuple | list | np.ndarray = None,
            sky_loc: tuple | list | np.ndarray = None,
            aperture: float | None = None,
            ax: plt.Axes | None = None
            ) -> None:
        """
        Extract the SED from a given pixel location in a spectral cube.
        Parameters
        ----------
        cube : str or Path
            Path to the spectral cube file.
        output_dir : str or Path, optional
            Directory to save the extracted SED plot. If None, no output is saved.
        pixel_loc : tuple or list
            The row and column indices of the pixel.
        sky_loc : tuple or list
            The row and column indices of the sky pixel.
        aperture : float, optional
            Aperture radius in pixels for the extraction. If None, no aperture is applied.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            The axis to plot the SED. If None, a new figure and axis are created.
        Returns
        -------
        None
        """
        
        # Check the type of cube (npy or fits)
        if isinstance(cube, (str, Path)):
            if cube.endswith('.npy'):
                data = np.load(cube)
            elif cube.endswith('.fits'):
                from astropy.io import fits
                with fits.open(cube) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
            else:
                raise ValueError("Unsupported file format. Use .npy or .fits files.")
        else:
            raise TypeError("Cube must be a string or Path object pointing to a file.")
        
        loc_string = None
        # Check the type of pixel_loc and sky_loc
        if pixel_loc is None and sky_loc is None:
            raise ValueError("Both pixel_loc and sky_loc cannot be None. At least one must be provided.")
        elif pixel_loc is not None and sky_loc is not None:
            raise ValueError("Only one of pixel_loc or sky_loc should be provided.")
        
        # In case of sky_loc, transform it into pixel_loc and aperture
        if sky_loc is not None:
            if aperture is None:
                raise ValueError("Aperture must be provided when using sky_loc.")
            # Convert sky_loc to pixel_loc based on the WCS information
            from astropy.wcs import WCS
            try:
                wcs2d = WCS(header).celestial
            except RuntimeError:    
                wcs2d = WCS(header).sub([WCSSUB_LONGITUDE | WCSSUB_LATITUDE])
                
            print(f"Using WCS header: {wcs2d}")
            pixel_loc = wcs2d.all_world2pix(sky_loc[0], sky_loc[1], 0)
            aperture = np.abs(int(aperture / (wcs2d.pixel_scale_matrix[0, 0] * 3600)))  # Convert aperture to pixel scale
            loc_string = f'{sky_loc[0]:.3f}, {sky_loc[1]:.3f}'
        else:
            loc_string = f'{pixel_loc[0]:.1f}, {pixel_loc[1]:.1f}'
        
        # Extract the SED from the location
        k, l = map(float, pixel_loc)
        if not isinstance(k, (int, float)) or not isinstance(l, (int, float)):
            print(f"Pixel location provided: {pixel_loc}")
            raise ValueError("Invalid pixel location provided. Ensure pixel_loc contains valid indices.")        
        
        if aperture is not None:
            # Extract a circular aperture around the pixel
            from photutils.aperture import CircularAperture
            aperture = CircularAperture(pixel_loc, r=aperture)
            mask = aperture.to_mask().to_image(data[0].shape)
            mask_3d = np.broadcast_to(mask, data.shape)
            sed = np.sum(data * mask_3d, axis=(1, 2))
        else:
            sed = data[:, k, l]
            
        # Extract the wavelength information from the header
        if 'CDELT3' in header or 'CD3_3' in header:
            delta_wav = header.get('CDELT3', header.get('CD3_3', 1.0)) * 10
            ref_wav = header.get('CRVAL3', 0.0) * 10
            n_files = data.shape[0]
            piv_lambs = ref_wav + np.arange(n_files) * delta_wav
        else:
            raise ValueError("Wavelength information not found in the header. Ensure CDELT3 or CD3_3 is present.")
            
        # Create a plot of the SED
        if ax is None and output_dir is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        else:
            fig = ax.figure
                
        # Load colour palette
        mcolors = np.array(self.makeSpecColors(len(piv_lambs)))
        
        for i, (lamb, flux) in enumerate(zip(piv_lambs, sed)):
            label = '7DT' if i == 0 else None
            marker = 'o' if flux > 0 else 'v'
            ax.errorbar(
                [lamb], [flux], yerr=[flux * 0.0] , xerr=[250/2],
                marker=marker, lw=0, elinewidth=1, capsize=2,
                c=mcolors[i], label=label, mec='#333333', mew='0.3'
                )
        del_y = sed.max() - sed.min()
        y_max = sed.max() + del_y * 0.1
        y_min = sed.min() - del_y * 0.1
        ax.set_xlim(3600, 9150)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
        ax.set_ylabel('Flux [mJy]')
        ax.legend(loc='best')
        ax.grid(True, which='both', color='#666666', linestyle='--', alpha=0.5)
        ax.set_title('7DT SED\nLocation ('+loc_string+')', fontdict={'fontsize':14})
        fig.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'sed_pixel_{k}_{l}.png' if output_name is None else output_dir / output_name
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
           return ax
       
    @classmethod
    def Slice(self,
              cube: str | Path,
              index: int,
              output_dir: str | Path = None,
              output_name: str = None,
              ax: plt.Axes | None = None,
              cmap: str | matplotlib.colors.Colormap = "bone",
              colorbar: bool = False,
              scale: str = 'Zscale'
              ) -> None:
        
        from astropy.visualization import (
            ZScaleInterval, AsymmetricPercentileInterval,
            AsinhStretch, LogStretch, SqrtStretch, LinearStretch,
            ImageNormalize
        )
        
        # Confirm the Cube Type
        if isinstance(cube, (str, Path)):
            if cube.endswith('.npy'):
                data = np.load(cube)
                if index not in np.arange(data.shape[0]):
                    raise ValueError("Index is not in both the wavelength and indice range.")
            elif cube.endswith('.fits'):
                with fits.open(cube) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                    wcs = WCS(header).celestial
                    if 'CDELT3' in header or 'CD3_3' in header:
                        delta_wav = header.get('CDELT3', header.get('CD3_3', 1.0)) * 10
                        ref_wav = header.get('CRVAL3', 0.0) * 10
                        wave_array = ref_wav + np.arange(data.shape[0]) * delta_wav
                    
                    if index not in wave_array and index not in np.arange(data.shape[0]):
                        raise ValueError("Index is not in both the wavelength and indice range.")
                    elif index in wave_array:
                        index = np.argwhere(wave_array == index)
                    else:
                        pass
            else:
                raise ValueError("Unsupported file format. Use .npy or .fits files.")
        else:
            raise TypeError("Cube must be a string or Path object pointing to a file.")
        
        # Set Visualization Scale/Interval
        if scale.lower() == 'zscale':
            interval = ZScaleInterval()
            stretch = LinearStretch()
        elif scale.lower() == 'log':
            interval = None
            stretch = LogStretch()
        norm = ImageNormalize(data, interval=interval, stretch=stretch)
        
        # Create a plot of the SED
        if ax is None and output_dir is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150, subplot_kw=dict(projection=wcs))
        else:
            fig = ax.figure
            # Change Projection
            bbox = ax.get_position()
            fig.delaxes(ax)
            ax = fig.add_axes(bbox, projection=wcs)
            
        # Plot slice
        im = ax.imshow(data[index], origin="lower", cmap=cmap, norm=norm)
        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.ax.tick_params(axis='x', which='both', direction='in', labelsize=7,
                                bottom=False, top=False, labelbottom=False, labeltop=False,
                                )
            cbar.ax.tick_params(axis='y', which='both', direction='in', labelsize=7,
                                left=False, right=True, labelleft=False, labelright=True,
                                )
            cbar.ax.set_ylabel('[mJy]', size=7, rotation='horizontal')
        
        ax.set_xlabel('RA [deg]', size=8) if 'ra' in ax.get_xlabel() else ax.set_xlabel('x')
        ax.set_ylabel('Dec [deg]', size=8) if 'dec' in ax.get_ylabel() else ax.set_ylabel('y')
        if wave_array is not None:
            title_label = f"{wave_array[index]:.0f}"+r"$\mathrm{\AA}$"
        else:
            title_label = f"Frame Numer # {index}"
        ax.set_title(title_label, size=9)
        
        
        # Save the plot if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'Slice_num_{index}.png' if output_name is None else output_dir / output_name
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
           return ax
        
    @classmethod
    def allCube(self,
            cube: str | Path,
            colorbar: bool = False,
            scale: str = 'Zscale',
            output_dir: str | Path = None,
            output_name: str = None
            ) -> None:
        
        with fits.open(cube) as hdul:
            header = hdul[0].header
            n_slice = int(header['NAXIS3'])
            
        k, l = self.find_rec(n_slice)
        cmaps = self.makeSepcCmaps(n=20, lightness='dark')
        
        fig, axes = plt.subplots(k, l, figsize=(k*2.5, l*2.), dpi=200)
        for i, ax in enumerate(axes.flatten()):
            col = i % l
            
            ax = self.Slice(cube=cube, index=i, ax=ax, cmap=cmaps[i], colorbar=colorbar, scale=scale)
            ax.tick_params(axis='both', which='both', 
                           labelsize=7, direction='in',
                           grid_color='#BBBBBB', grid_alpha=0.7, grid_linewidth=0.7, grid_linestyle='--')
            ax.tick_params(axis='x', which='both',
                           bottom=True, top=False)
            ax.tick_params(axis='y', which='both',
                           left=True, right=False)
            ax.grid(True)
            if col != 0:
                # ax.coords['Dec'].set_ticks_visible(False)         # tick marks
                ax.coords['Dec'].set_ticklabel_visible(False)     # tick labels
                ax.set_ylabel('')
            
        fig.tight_layout()
        
                # Save the plot if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cube_stem = Path(cube).stem
            output_path = output_dir / f'Cube_2D_{cube_stem}.png' if output_name is None else output_dir / output_name
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
           return fig