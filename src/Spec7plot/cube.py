import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from glob import glob
import os

class Cube:
    """
    Class to build a 3D data cube from a list of FITS files.

    Parameters
    ----------
    file_list : list of str, optional
        List of FITS file paths to include in the cube.
    directory : str, optional
        Directory containing FITS files. If provided, file_list is ignored.
    pattern : str, optional
        Glob pattern to match FITS files in directory (default: "*.fits").
    ext : int or str, optional
        FITS extension (HDU) to read data from (default: 0).
    sort_key : str, optional
        FITS header keyword by which to sort the files (e.g., wavelength).
    """

    def __init__(
        self,
        file_list=None,
        directory=None,
        pattern="*.fits",
        ext=0,
        sort_key=None,
    ):
        if directory is not None:
            self.file_list = sorted(glob(os.path.join(directory, pattern)))
        else:
            self.file_list = file_list or []

        if sort_key is not None:
            try:
                self.file_list.sort(key=lambda f: fits.getheader(f, ext)[sort_key])
            except KeyError:
                raise KeyError(f"Header keyword '{sort_key}' not found in FITS files.")

        self.ext = ext
        self.data = None
        self.header = None
        self.wcs = None
        self.header_cube = None


    def build(self):
        """
        Load FITS files, stack into a 3D NumPy array, and prepare header/WCS.

        Returns
        -------
        tuple
            (data cube with shape (n_files, ny, nx), cube header with WCS info)
        """
        if not self.file_list:
            raise ValueError("No FITS files provided to build the cube.")

        # 1) Get the spatial header and data shape from the first file
        with fits.open(self.file_list[0]) as hdul:
            data0 = hdul[self.ext].data
            hdr0 = hdul[self.ext].header

        ny, nx = data0.shape
        n_files = len(self.file_list)
        cube_data = np.empty((n_files, ny, nx), dtype=data0.dtype)
        
        # 2) Collect data and wavelength values
        wavelengths = []
        for i, fname in enumerate(self.file_list):
            with fits.open(fname) as hdul:
                data_i = hdul[self.ext].data
                hdr_i = hdul[self.ext].header
            cube_data[i] = data_i

            if 'FILTER' in hdr_i:
                try:
                    wav = float(''.join(filter(str.isdigit, hdr_i['FILTER'])))
                except ValueError:
                    wav = float(i)
            else:
                wav = float(i)
            wavelengths.append(wav)
        
        # 3) Define wavelengths and sort the data 
        sort_idx = np.argsort(wavelengths)

        cube_data   = cube_data[sort_idx, :, :]
        
        wavelengths = [wavelengths[i] for i in sort_idx]
        delta_wav = np.median(np.diff(wavelengths)) if len(wavelengths) > 1 else 1.0
        ref_wav   = wavelengths[0]
        
        print(wavelengths)

        # 4) Create WCS from the first file's header
        wcs0 = WCS(hdr0)
        cube_hdr = wcs0.to_header()

        # 5) Conserve spatial axis information (CDELT/CD)
        if 'CDELT1' in hdr0 and 'CDELT2' in hdr0:
            if hdr0['CDELT1'] == 1.0 or hdr0['CDELT2'] == 1.0:
                print("CDELT1 or CDELT2 is 1.0, using CD matrix instead.")
                del cube_hdr['CDELT1']
                del cube_hdr['CDELT2']
                
                for key in ('CD1_1','CD1_2','CD2_1','CD2_2'):
                    if key in hdr0:
                        print(f"Using {key} from header.")
                        cube_hdr[key] = hdr0[key]
            else:
                cube_hdr['CDELT1'] = hdr0['CDELT1']
                cube_hdr['CDELT2'] = hdr0['CDELT2']
        else:
            pass
        
        # 6) Define the 3D cube axes and add the spectral axis
        cube_hdr['WCSAXES'] = 3
        cube_hdr['NAXIS']   = 3
        cube_hdr['NAXIS1']  = nx
        cube_hdr['NAXIS2']  = ny
        cube_hdr['NAXIS3']  = n_files

        cube_hdr['CTYPE3']  = 'WAVELENGTH'
        cube_hdr['CUNIT3']  = hdr0.get('CUNIT3', hdr0.get('CUNIT1', '')) 
        cube_hdr['CRPIX3']  = 1
        cube_hdr['CRVAL3']  = ref_wav
        cube_hdr['CDELT3']  = delta_wav

        # 7) Add CD matrix for the third axis
        cube_hdr['CD3_3'] = delta_wav
        cube_hdr['CD1_3'] = 0.0
        cube_hdr['CD2_3'] = 0.0

        # 8) Save instance variables
        self.data        = cube_data
        self.header      = hdr0
        self.wcs         = wcs0
        self.header_cube = cube_hdr

        return self.data, self.header_cube

    def save(self, output_path, output_type, overwrite=False):
        if self.data is None or self.header_cube is None:
            raise RuntimeError("Cube data not built. Call build() before save().")
        
        if output_type.lower() == 'fits':
            hdu = fits.PrimaryHDU(data=self.data, header=self.header_cube)
            hdu.writeto(output_path, overwrite=overwrite)
        elif output_type.lower() == 'npy':
            np.save(output_path, self.data)
        else:
            raise ValueError(f"Unsupported output type: {output_type}. Use 'fits' or 'npy'.")

    def run(self, output_path, output_type, overwrite=False):
        data, header = self.build()
        self.save(output_path, output_type=output_type, overwrite=overwrite)
        return output_path