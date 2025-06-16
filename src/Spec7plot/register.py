import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from reproject import reproject_interp


class imRegister:
    def __init__(self,
                 input_images: list | np.ndarray,
                 output_dir: str | Path
                 ) -> None:
        """
        Initialize the image registration class.
        Parameters:
            input_images (list or np.ndarray): List of input file names or a numpy array of file names.
            output_dir (str or Path): Directory where the output files will be saved.
        """
        
        # Check if the input file is a list or a single file.
        if not isinstance(input_images, np.ndarray):
            if isinstance(input_images, str):
                raise ValueError("Input file must be a list of files or a numpy array.")
            else:
                try:
                    self.input_images = np.asarray(input_images)
                except Exception as e:
                    raise ValueError(f"Could not convert input file to numpy array: {e}")
        else:
            self.input_images = input_images
        
        # Get the reference file name.
        self.ref_file = self.input_images[0]
        
        # Set the output directory.
        if isinstance(output_dir, str):
            self.output_dir = Path(output_dir)
        elif isinstance(output_dir, Path):
            self.output_dir = output_dir
        else:
            raise ValueError("Output directory must be a string or a Path object.")
        
        # Ensure the output directory exists.
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise OSError(f"Could not create output directory: {e}")

    
    def run(self,
            position: tuple | list | np.ndarray = (5100, 3400),
            size: tuple | list | np.ndarray = (100, 100)
            ) -> list | np.ndarray:
        """
        Run the image registration process for all input files.
        """
        # Iterate through each input file and perform registration.
        for input_file in self.input_images:
            self.wRegistration(input_file)
            self.cutout(position=position, size=size, filename=input_file)
            
        return self.output_dir
    
    @staticmethod
    def add_prefix(path, prefix: str) -> Path:
        """
        Add a prefix to the file name in the given path.
        Parameters:
            path (str or Path): The original file path.
            prefix (str): The prefix to add to the file name.
        """
        p = Path(path)
        # p.parent: directory path, p.name: file name(with extension)
        new_name = prefix + p.name
        return new_name

    def wRegistration(self,
                      input_file: str
                      ) -> None:
        """
        Perform image registration for various observatories using astropy.reproject.

        Parameters:
            input_file (str): The input file name (image to be registered).
        """

        # Construct output file path
        output_file = os.path.join(
            self.output_dir,
            self.add_prefix(input_file, prefix='wr_')  # Add 'wr_' prefix to the output file
        )
        
        # --- Reprojection using astropy.reproject ---
        # Build full paths to the input and reference files.
        input_path = Path(input_file)
        ref_path = Path(self.ref_file)
        
        
        # Open the input image and retrieve data and header.
        with fits.open(input_path) as hdul_input:
            input_data = hdul_input[0].data
            input_header = hdul_input[0].header

            # Open the reference image and get its header.
            with fits.open(ref_path) as hdul_ref:
                ref_header = hdul_ref[0].header
                
                # Reproject the input image onto the reference header using linear interpolation.
                output_data, footprint = reproject_interp((input_data, input_header), ref_header, order='bilinear')
                
                # Extract WCS information from the reference header.
                ref_wcs = WCS(ref_header)

                # Convert the WCS object to a header with only WCS-related keywords.
                ref_wcs_header = ref_wcs.to_header()

                # Update the input header with WCS information from the reference.
                input_header.update(ref_wcs_header)

            # Write the reprojected image out to the output file using the updated input header.
            fits.writeto(output_file, output_data, input_header, overwrite=True)


    def cutout(
        self,
        position: tuple,
        size: tuple,
        filename: str = None
    ) -> str:
        """
        Cut out a sub-image from a FITS file, preserving its original header
        except for updated WCS and size keywords.

        Parameters
        ----------
        position : tuple
            (x, y) center of the cutout in pixel coordinates.
        size : tuple
            (width, height) in pixels of the cutout.
        filename : str
            Path to the input FITS file.

        Returns
        -------
        str
            Path to the output cutout FITS file.
        """
        if filename is None:
            raise ValueError("Filename must be provided for the cutout operation.")

        # Determine output filename
        base = os.path.basename(filename)
        out_name = self.add_prefix(base, prefix='cut_') if not base.startswith('wr_') else f"cut_{base.replace('wr_', '')}"
        output_file = os.path.join(self.output_dir, out_name)

        # Open input FITS and copy header
        with fits.open(filename) as hdul:
            orig_hdu = hdul[0]
            orig_header = orig_hdu.header.copy()
            data = orig_hdu.data

        # Create WCS from original header
        wcs_orig = WCS(orig_header)

        # Perform cutout, WCS updated internally
        cut = Cutout2D(data, position, size, wcs=wcs_orig)

        # Start new header from original and remove old WCS keywords
        new_header = orig_header.copy()
        for key in list(new_header.keys()):
            if key.startswith(('CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CD')):
                new_header.remove(key, ignore_missing=True)

        # Merge new WCS header
        wcs_header = cut.wcs.to_header(relax=True)
        for k, v in wcs_header.items():
            new_header[k] = v

        # Update size keywords
        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = cut.data.shape[1]
        new_header['NAXIS2'] = cut.data.shape[0]
        
        # Updat CD matrix
        if 'CD1_1' in new_header and 'CD2_2' in new_header:
            pass
        else:
            new_header['CD1_1'] = orig_header.get('CD1_1', 1.0)
            new_header['CD2_2'] = orig_header.get('CD2_2', 1.0)
            new_header['CD1_1'] = orig_header.get('CD1_1', 1.0)
            new_header['CD2_2'] = orig_header.get('CD2_2', 1.0)
            # if 'CDELT1' in orig_header and 'CDELT2' in orig_header:
            #     new_header['CDELT1'] = orig_header.get('CDELT1', 1.0)
            #     new_header['CDELT2'] = orig_header.get('CDELT2', 1.0)

        if 'ZP_AUTO' in orig_header:
            ZP = orig_header['ZP_AUTO']
            flux = 3631 * (cut.data) * 10 ** (-ZP / 2.5)  # Jy
            f_data = 1e3 * flux  # mJy
            new_header['ZP_AUTO'] = ZP
        else:
            f_data = cut.data
        
        # Write output preserving original header entries
        hdu_out = fits.PrimaryHDU(data=f_data, header=new_header)
        hdu_out.writeto(output_file, overwrite=True)

        return output_file