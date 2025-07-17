import os
import warnings
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy import units as u

from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

class imRegister:
    def __init__(self,
                 input_images: list | np.ndarray,
                 output_dir: str | Path
                 ) -> None:

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

        self.ref_file = self.input_images[0]

        if isinstance(output_dir, str):
            self.output_dir = Path(output_dir)
        elif isinstance(output_dir, Path):
            self.output_dir = output_dir
        else:
            raise ValueError("Output directory must be a string or a Path object.")

        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise OSError(f"Could not create output directory: {e}")

    def run(self,
            position: tuple = (5100, 3400),
            size: tuple = (100, 100),
            use_skycoord: bool = False
            ) -> list:
        for input_file in self.input_images:
            registered = self.wRegistration(input_file)
            if use_skycoord:
                self.cutout_sky(skycoord=position, size=size, filename=registered)
            else:
                self.cutout(position=position, size=size, filename=registered)

        return self.output_dir

    @staticmethod
    def add_prefix(path, prefix: str) -> Path:
        p = Path(path)
        new_name = prefix + p.name
        return new_name

    def wRegistration(self,
                      input_file: str
                      ) -> str:

        output_file = os.path.join(
            self.output_dir,
            self.add_prefix(input_file, prefix='wr_')
        )

        input_path = Path(input_file)
        ref_path = Path(self.ref_file)

        with fits.open(input_path) as hdul_input:
            input_data = hdul_input[0].data
            input_header = hdul_input[0].header

            with fits.open(ref_path) as hdul_ref:
                ref_header = hdul_ref[0].header

                output_data, footprint = reproject_interp((input_data, input_header), ref_header, order='bilinear')

                ref_wcs = WCS(ref_header)
                ref_wcs_header = ref_wcs.to_header()
                input_header.update(ref_wcs_header)

            fits.writeto(output_file, output_data, input_header, overwrite=True)

        return output_file

    def cutout(self,
               position: tuple,
               size: tuple,
               filename: str = None
               ) -> str:

        if filename is None:
            raise ValueError("Filename must be provided for the cutout operation.")

        base = os.path.basename(filename)
        out_name = self.add_prefix(base, prefix='cut_') if not base.startswith('wr_') else f"cut_{base.replace('wr_', '')}"
        output_file = os.path.join(self.output_dir, out_name)

        with fits.open(filename) as hdul:
            orig_hdu = hdul[0]
            orig_header = orig_hdu.header.copy()
            data = orig_hdu.data

        wcs_orig = WCS(orig_header)

        cut = Cutout2D(data, position, size, wcs=wcs_orig)

        new_header = orig_header.copy()
        for key in list(new_header.keys()):
            if key.startswith(('CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CD')):
                new_header.remove(key, ignore_missing=True)

        wcs_header = cut.wcs.to_header(relax=True)
        new_header.update(wcs_header)

        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = cut.data.shape[1]
        new_header['NAXIS2'] = cut.data.shape[0]

        if 'CD1_1' not in new_header or 'CD2_2' not in new_header:
            new_header['CD1_1'] = orig_header.get('CD1_1', 1.0)
            new_header['CD2_2'] = orig_header.get('CD2_2', 1.0)

        if 'ZP_AUTO' in orig_header:
            ZP = orig_header['ZP_AUTO']
            flux = 3631 * (cut.data) * 10 ** (-ZP / 2.5)
            f_data = 1e3 * flux
            new_header['ZP_AUTO'] = ZP
        else:
            f_data = cut.data

        hdu_out = fits.PrimaryHDU(data=f_data, header=new_header)
        hdu_out.writeto(output_file, overwrite=True)

        return output_file

    def cutout_sky(self,
                   skycoord: tuple,
                   size: tuple,
                   filename: str
                   ) -> str:

        base = os.path.basename(filename)
        out_name = self.add_prefix(base, prefix='cut_')
        output_file = os.path.join(self.output_dir, out_name)

        with fits.open(filename) as hdul:
            orig_hdu = hdul[0]
            orig_header = orig_hdu.header.copy()
            data = orig_hdu.data
            wcs = WCS(orig_header)

        skycoord_obj = SkyCoord(ra=skycoord[0]*u.deg, dec=skycoord[1]*u.deg)
        pixel_position = skycoord_obj.to_pixel(wcs=wcs)

        cut = Cutout2D(data, pixel_position, size, wcs=wcs)

        new_header = orig_header.copy()
        for key in list(new_header.keys()):
            if key.startswith(('CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CD')):
                new_header.remove(key, ignore_missing=True)

        wcs_header = cut.wcs.to_header(relax=True)
        new_header.update(wcs_header)

        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = cut.data.shape[1]
        new_header['NAXIS2'] = cut.data.shape[0]

        if 'CD1_1' not in new_header or 'CD2_2' not in new_header:
            new_header['CD1_1'] = orig_header.get('CD1_1', 1.0)
            new_header['CD2_2'] = orig_header.get('CD2_2', 1.0)

        if 'ZP_AUTO' in orig_header:
            ZP = orig_header['ZP_AUTO']
            flux = 3631 * (cut.data) * 10 ** (-ZP / 2.5)
            f_data = 1e3 * flux
            new_header['ZP_AUTO'] = ZP
        else:
            f_data = cut.data

        hdu_out = fits.PrimaryHDU(data=f_data, header=new_header)
        hdu_out.writeto(output_file, overwrite=True)

        return output_file
