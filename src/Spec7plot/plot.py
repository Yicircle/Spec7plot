import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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
    
    def makeSpecColors(self, 
                       n: int = 40, 
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

    def SED(self,
            cube: str | Path,
            output_dir: str | Path = None,
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
            wcs = WCS(header)
            pixel_loc = wcs.world_to_pixel(sky_loc)
            aperture = int(aperture / wcs.pixel_scale_matrix[0, 0])  # Convert aperture to pixel scale
        
        # Extract the SED from the location
        k, l = pixel_loc
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
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
        else:
            fig = ax.figure
                
        # Load colour palette
        mcolors = np.array(self.makeSpecColors(len(piv_lambs)))
        
        for i, (lamb, flux) in enumerate(zip(piv_lambs, sed)):
            label = '7DT' if i == 0 else None
            ax.errorbar(
                [lamb], [flux], yerr=[flux * 0.0] , xerr=[250/2],
                marker='o', lw=0, elinewidth=1, capsize=2,
                c=mcolors[i], label=label
                )
        del_y = sed.max() - sed.min()
        y_max = sed.max() + del_y * 0.1
        y_min = sed.min() - del_y * 0.1
        ax.set_xlim(3600, 9150)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(r'Wavelength [$\mathrm{\AA}$]')
        ax.set_ylabel('Flux [mJy]')
        ax.legend(loc='lower right')
        ax.grid(True, which='both', color='#666666', linestyle='--', alpha=0.5)
        ax.set_title(f'7DT SED\npixel location ({k}, {l})', fontdict={'fontsize':14})
        fig.tight_layout()
        
        # Save the plot if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'sed_pixel_{k}_{l}.png'
            fig.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
            
        return ax