#! /usr/bin/env python
#
#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import netCDF4 as nc
import numpy as np
import os


class SpectralMetadata:
    def __init__(self, wavelengths, fwhm):
        """
        Initializes the SpectralMetadata object.

        Args:
            wavelengths (numpy.ndarray): Array of wavelength values.
            fwhm (numpy.ndarray): Array of full-width half-maximum values.
        """
        self.wavelengths = wavelengths
        self.wl = wavelengths
        self.fwhm = fwhm

class GenericGeoMetadata:
    def __init__(self, band_names, geotransform=None, projection=None, glt=None, pre_orthod=False, nodata_value=None, loc=None):
        """
        Initializes the GenericGeoMetadata object.

        Args:
            band_names (list): List of band names.
            geotransform (tuple, optional): GDAL-style geotransform array. Defaults to None.
            projection (str, optional): Projection string. Defaults to None.
            glt (numpy.ndarray, optional): 3d array of x and y indices. Defaults to None.
            pre_orthod (bool, optional): Whether the data is already orthod. Defaults to False.
            nodata_value (int, optional): The nodata value. Defaults to None.
        """
        self.band_names = band_names
        self.geotransform = geotransform
        self.projection = projection
        self.glt = glt
        self.pre_orthod = False
        self.nodata_value = nodata_value
        self.loc = loc

        if pre_orthod:
            self.orthoable = False
        elif self.glt is None:
            self.orthoable = False
        else:
            self.orthoable = True

def open_emit_rdn(input_file, meta_only=False):
    """
    Opens an EMIT radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        meta_only (bool): Set to True to save time and only load meta data

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The radiance data as a numpy array with shape (lines, bands, samples).
    """
    ds = nc.Dataset(input_file)
    wl = ds['sensor_band_parameters']['wavelengths'][:]
    fwhm = ds['sensor_band_parameters']['fwhm'][:]

    if meta_only:
        rdn = None
    else:
        rdn = np.array(ds['radiance'][:])

    meta = SpectralMetadata(wl, fwhm)

    return meta, rdn

def open_airborne_rdn(input_file, meta_only=False):
    """
    Opens an Airborne radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        meta_only (bool): Set to True to save time and only load meta data

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The radiance data as a numpy array with shape (lines, bands, samples).
    """
    ds = nc.Dataset(input_file)
    wl = ds['radiance']['wavelength'][:]
    fwhm = ds['radiance']['fwhm'][:]

    if meta_only:
        rdn = None
    else:
        rdn = np.transpose(ds['radiance']['radiance'][:], (1, 2, 0))

    meta = SpectralMetadata(wl, fwhm)

    return meta, rdn


def open_netcdf_radiance(input_file, meta_only=False):
    """
    Opens a NetCDF radiance file and extracts the metadata and data.
    Automatically detects whether it's EMIT, AV3, or AV5 format.

    Args:
        input_file (str): Path to the NetCDF file.
        meta_only (bool): Set to True to save time and only load meta data

    Raises:
        ValueError: If the file type cannot be determined.
        FileNotFoundError: If the file does not exist.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The radiance data as a numpy array.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'{input_file} not found.')

    input_filename = os.path.basename(input_file)
    input_filename_lower = input_filename.lower()

    if 'emit' in input_filename_lower and ('rad' in input_filename_lower or 'rdn' in input_filename_lower):
        return open_emit_rdn(input_file, meta_only)

    is_airborne_like = any(tag in input_filename_lower for tag in ['av3', 'av5', 'ang', 'prm', 'prism'])
    if is_airborne_like and 'rdn' in input_filename_lower:
        return open_airborne_rdn(input_file, meta_only)

    raise ValueError(f'Unknown or unsupported NetCDF radiance file type for {input_file}')

def open_netcdf_l2a_mask(input_file, mask_type, lazy=True, load_glt=False, load_loc=False):
    return open_emit_l2a_mask_nc(input_file, mask_type, lazy=True, load_glt=False, load_loc=False)

def open_emit_l2a_mask_nc(input_file, mask_type, lazy=True, load_glt=False, load_loc=False):
    """
    Opens an EMIT L2A_MASK or L1B_BANDMASK NetCDF file and extracts the spectral metadata and mask data.

    Args:
        input_file (str): Path to the NetCDF file.
        mask_type (str): Mask type. Options are
            'mask': L2A_MASK, in this case input_file should be an L2A_MASK file
            'band_mask': L1B_BANDMASK, in this case input_file should be an EMIT L2A_MASK file
                         or an AV3 L1B_RDN file
        lazy (bool, optional): Ignored

    Returns:
        tuple: A tuple containing:
            - GenericGeoMetadata: An object containing the band names
            - numpy.ndarray or netCDF4.Variable: The mask data
    """
    if not mask_type in ['mask', 'band_mask']:
        raise ValueError(f"Invalid mask type {mask_type}. Must use either 'mask' or 'band_mask'")

    ds = nc.Dataset(input_file)
    proj = ds.spatial_ref
    trans = ds.geotransform

    if mask_type == 'mask':
        mask_names = list(ds['sensor_band_parameters']['mask_bands'][...])
    else:
        mask_names = ['']

    nodata_value = float(ds[mask_type]._FillValue)
    glt = None
    if load_glt:
        glt = np.stack([ds['location']['glt_x'][:],ds['location']['glt_y'][:]],axis=-1)
    loc = None
    if load_loc:
        loc = np.stack([ds['location']['lon'][:],ds['location']['lat'][:]],axis=-1)

    mask = np.array(ds[mask_type][...])
    
    meta = GenericGeoMetadata(mask_names, trans, proj, glt=glt, pre_orthod=True, nodata_value=nodata_value, loc=loc)

    return meta, mask