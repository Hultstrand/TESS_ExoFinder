# Copyright 2018 Liang Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for reading TESS data.

    TESS File format:
    .../tessyyyydddhhmmss-ssctr-tid-scid-cr_lc.fits

    Example File:
    .../tess2018234235059-s0002-0000000009006668-0121-s_lc.fits
        where the TCE is from sector 2, TIC id 9006668, spacecraft configuration 121, and short cadence light curve

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob

import os.path
from astropy.io import fits
import numpy as np

from tensorflow import io


def tess_filenames(tic, sector,
                     base_dir='C:\\Users\\A_J_F\\Documents\\TESS_ExoFinder\\data\\tess',
                     injected=False,
                     inject_dir='/sector-43',
                     check_existence=True):
    """Returns the light curve filename for a TESS target star.

    Args:
      tic: TIC of the target star. May be an int or a possibly zero-
          padded string.
      base_dir: Base directory containing Kepler data.
      sector: Int, sector number of data.
      cam: Int, camera number of data.
      ccd: Int, CCD number of data.
      injected: Bool, whether target also has a light curve with injected planets.
      injected_dir: Directory containing light curves with injected transits.
      check_existence: If True, only return filenames corresponding to files that
          exist.

    Returns:
      filename for given TIC.
    """

    sector = str(sector).rjust(2, '0')
    tic = str(tic).rjust(16, '0')

    if not injected:
        # Modify as needed
        base_dir='C:\\Users\\A_J_F\\Documents\\TESS_ExoFinder\\data\\tess'
        filesearch = "-s00" + str(sector) + "-" + str(tic)
        filelist = glob.glob(base_dir + "/*" + filesearch + "*.fits")
        if len(filelist) == 0:
            return
        else:
            filename = os.path.join(base_dir, filelist[0])

    else:
        filename = os.path.join(inject_dir, tic + '.lc')

    if check_existence or io.gfile.exists(filename):
        return filename
    return


def read_tess_light_curve(filename, invert, flux_key='KSPMagnitude'):
    """Reads time and flux measurements for a TESS target star.

    Args:
      filename: str name of .fits file containing time and flux measurements.
      invert: Whether to reflect flux values across the y-axis or change code to flip around the median flux value. This is
        performed separately for each .fits file.

    Returns:
      time: The time values of the light curve.
      flux: The flux values of the light curve.
    """
    with fits.open(io.gfile.GFile(filename, mode="rb")) as hdu_list:
        time = hdu_list[1].data['TIME']
        flux = hdu_list[1].data['PDCSAP_FLUX']

        if 'QUALITY' in fits.getdata(filename, ext=1).columns.names:
            quality_flag = np.where(np.array(hdu_list[1].data['QUALITY']) == 0)

            # Remove outliers
            time = time[quality_flag]
            flux = flux[quality_flag]

            # Remove NaN flux values.
            valid_indices = np.where(np.isfinite(flux))
            time = time[valid_indices]
            flux = flux[valid_indices]

    if invert:
        #Flip the order of the flux
        flux = np.flip(flux)
        # Optional flip over the median
        #flux *= -1

    return time, flux