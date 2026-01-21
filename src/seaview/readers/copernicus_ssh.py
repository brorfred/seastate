"""File handler for Copernicus Marine Service SSH NetCDF files.

This module provides a SatPy-compatible file handler for reading Copernicus
Marine Service Sea Surface Height (SSH) data from NetCDF files.
"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr
from satpy.readers.file_handlers import BaseFileHandler

from processor.area_definitions import rectlinear

logger = logging.getLogger(__name__)


class CopernicusSSHFileHandler(BaseFileHandler):
    """File handler for Copernicus SSH NetCDF files.

    This handler reads gridded SSH data with 1D lat/lon coordinates and
    converts them to 2D arrays for SatPy compatibility.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the NetCDF file.
    filename_info : dict
        Dictionary with parsed filename information.
    filetype_info : dict
        Dictionary with file type configuration.

    Attributes
    ----------
    nc : xarray.Dataset
        The opened NetCDF dataset.
    filename_info : dict
        Parsed filename information.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the file handler.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to the NetCDF file.
        filename_info : dict
            Dictionary with parsed filename information.
        filetype_info : dict
            Dictionary with file type configuration.
        """
        super().__init__(filename, filename_info, filetype_info)
        self.nc = xr.open_dataset(
            self.filename,
            decode_cf=True,
            mask_and_scale=True
        )
        self.filename_info = filename_info
        self._extract_time_info()

    def _extract_time_info(self):
        """Extract and set start/end times from the dataset.

        Sets the _start_time and _end_time attributes based on the time
        coordinate in the dataset, or falls back to filename info.
        """
        if 'time' in self.nc.coords:
            time_values = self.nc.time.values
            self._start_time = self._convert_to_datetime(time_values[0])
            self._end_time = self._convert_to_datetime(time_values[-1])
        else:
            # Fallback to filename info
            self._start_time = self.filename_info.get('start_time')
            self._end_time = self._start_time

    @staticmethod
    def _convert_to_datetime(numpy_datetime):
        """Convert numpy datetime64 to Python datetime.

        Parameters
        ----------
        numpy_datetime : numpy.datetime64
            The numpy datetime object to convert.

        Returns
        -------
        datetime.datetime
            Python datetime object.
        """
        if hasattr(numpy_datetime, 'astype'):
            return numpy_datetime.astype('datetime64[us]').astype(datetime)
        return numpy_datetime

    def _create_2d_coordinates(self):
        """Create 2D coordinate arrays from 1D lat/lon.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple of (lon_2d, lat_2d) as 2D numpy arrays.
        """
        lons_1d = self.nc['longitude'].values
        lats_1d = self.nc['latitude'].values
        return np.meshgrid(lons_1d, lats_1d)

    def _load_coordinate(self, var_name):
        """Load a coordinate variable as a 2D array.

        Parameters
        ----------
        var_name : str
            Name of coordinate variable ('longitude' or 'latitude').

        Returns
        -------
        xarray.DataArray
            2D coordinate array with dims ['y', 'x'].
        """
        lon_2d, lat_2d = self._create_2d_coordinates()
        coord_2d = lon_2d if var_name == 'longitude' else lat_2d

        return xr.DataArray(
            coord_2d,
            dims=['y', 'x'],
            attrs=self.nc[var_name].attrs.copy()
        )

    def _load_data_variable(self, var_name):
        """Load a data variable from the NetCDF file.

        Parameters
        ----------
        var_name : str
            Name of the variable to load.

        Returns
        -------
        xarray.DataArray
            Processed data array with renamed dimensions.
        """
        data = self.nc[var_name]

        # Remove singleton time dimension
        if 'time' in data.dims and data.sizes['time'] == 1:
            data = data.squeeze('time', drop=True)

        # Rename spatial dimensions to SatPy convention
        if 'latitude' in data.dims and 'longitude' in data.dims:
            data = data.rename({'latitude': 'y', 'longitude': 'x'})

        return data

    def _add_metadata(self, data, ds_info):
        """Add metadata from dataset configuration to data array.

        Parameters
        ----------
        data : xarray.DataArray
            DataArray to update with metadata.
        ds_info : dict
            Dataset configuration dictionary from YAML.
        """
        # Add standard platform/sensor metadata
        data.attrs.update({
            'platform_name': 'altimeter',
            'sensor': 'altimeter',
        })

        # Add metadata from YAML configuration
        metadata_keys = ['standard_name', 'long_name', 'units']
        for key in metadata_keys:
            if key in ds_info:
                data.attrs[key] = ds_info[key]

    def get_dataset(self, dataset_id, ds_info):
        """Get a dataset from the file.

        Parameters
        ----------
        dataset_id : satpy.dataset.DataID
            DataID for the requested dataset.
        ds_info : dict
            Dataset configuration from YAML.

        Returns
        -------
        xarray.DataArray or None
            The requested dataset, or None if not found.
        """
        var_name = ds_info.get('file_key', dataset_id['name'])

        logger.debug(f"Loading {var_name} from {self.filename}")

        if var_name not in self.nc:
            logger.warning(f"Variable {var_name} not found in {self.filename}")
            return None

        # Load coordinate or data variable
        if var_name in ['longitude', 'latitude']:
            data = self._load_coordinate(var_name)
        else:
            data = self._load_data_variable(var_name)

        # Add metadata
        self._add_metadata(data, ds_info)

        return data

    def get_area_def(self, dsid):
        """Get area definition for the dataset.

        Parameters
        ----------
        dsid : satpy.dataset.DataID
            Dataset ID.

        Returns
        -------
        pyresample.AreaDefinition
            Rectilinear area definition based on data extent.
        """
        kw = dict(shape=self.nc["sla"].shape[-2:],
                  lat1=self.nc.latitude.max().item(),
                  lat2=self.nc.latitude.min().item(),
                  lon1=self.nc.longitude.min().item(),
                  lon2=self.nc.longitude.max().item())
        return  rectlinear(**kw)


    @property
    def start_time(self):
        """datetime.datetime : Start time of the data."""
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        """datetime.datetime : End time of the data."""
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    def __del__(self):
        """Clean up by closing the NetCDF dataset."""
        if hasattr(self, 'nc'):
            self.nc.close()
