import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.reproject import reproject_interp


class imRegister:
    def __init__(self,
                 input_files: list | np.ndarray
                 output_dir: str | Path
                 ) -> None:
        """
        Initialize the image registration class.
        Parameters:
            input_file (list or np.ndarray): List of input file names or a numpy array of file names.
            output_dir (str or Path): Directory where the output files will be saved.
        """
        
        # Check if the input file is a list or a single file.
        if not isinstance(input_files, np.ndarray):
            if isinstance(input_files, str):
                raise ValueError("Input file must be a list of files or a numpy array.")
            else:
                try:
                    self.input_files = np.asarray(input_files)
                except Exception as e:
                    raise ValueError(f"Could not convert input file to numpy array: {e}")
        else:
            self.input_files = input_files
        
        # Get the reference file name.
        self.ref_file = self.input_files[0]
        
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

    
    def run(self):
        """
        Run the image registration process for all input files.
        """
        # Iterate through each input file and perform registration.
        for input_file in self.input_files:
            self.wRegistration(input_file, self.ref_file, self.output_dir)
        

    def wRegistration(self,
                      input_file: str,
                      ) -> None:
        """
        Perform image registration for various observatories using astropy.reproject.

        Parameters:
            input_file (str): The input file name (image to be registered).
        """

        # Construct output file path
        output_file = os.path.join(
            self.output_dir,
            f"wr_{input_file}"
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


    def cutout(self,
               position: tuple,
               size: tuple,
               filename: str = None
               ) -> None:
        
        if filename is None:
            raise ValueError("Filename must be provided for the cutout operation.")
        
        output_file = os.path.join(
            self.output_dir,
            f"cut_{filename}"
        ) if not 'wr_' in filename else os.path.join(
            self.output_dir,
            f"cut_{filename.replace('wr_', '')}"
        )
        
        # Open fits file
        with fits.open(filename) as hdul:
            hdu = hdul[0]
            data = hdu.data
            header = hdu.header

        # Create a WCS object from the header
        wcs = WCS(header)

        # Create the cutout; this will update the WCS accordingly.
        cutout = Cutout2D(data, position, size, wcs=wcs)

        # Create a new header from the cutout's WCS.
        new_header = cutout.wcs.to_header()
        
        # Update the new header with additional info (e.g., cutout size)
        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = cutout.data.shape[1]
        new_header['NAXIS2'] = cutout.data.shape[0]
        del new_header['CDELT1']
        del new_header['CDELT2']
        new_header['CD1_1'] = -0.505 / 3600  # Default pixel scale of 7DT in arcsec/pixel
        new_header['CD1_2'] = 0
        new_header['CD2_1'] = 0
        new_header['CD2_2'] = 0.505 / 3600

        # Save the cutout as a new FITS file, preserving the updated WCS.

        hdu = fits.PrimaryHDU(data=cutout.data, header=new_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(output_file, overwrite=True)