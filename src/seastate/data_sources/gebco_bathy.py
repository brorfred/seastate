"""GEBCO Bathymetry data access.

This module provides functions to retrieve and access GEBCO bathymetry
data from CEDA via OPeNDAP.

References
----------
https://www.gebco.net/data_and_products/gridded_bathymetry_data/
"""
import pathlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import satpy
import xarray as xr
import copernicusmarine

from ..area_definitions import rectlinear as rectlin_area
from .. import config
settings = config.settings

DATADIR = pathlib.Path(settings["data_dir"] + "/GEBCO")
DATADIR.mkdir(parents=True, exist_ok=True)


VERBOSE = True


def vprint(text):
    """Print text if verbose mode is enabled.

    Parameters
    ----------
    text : str
        Text to print.
    """
    if VERBOSE:
        print(text)


def filename(dtm=None):
    """Generate filename for GEBCO bathymetry data file.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        Unused parameter for API compatibility.

    Returns
    -------
    str
        Filename for the GEBCO dataset.
    """
    return "gebco_2025_sub_ice.nc"


def open_dataset(dtm=None, step=5, mask_land=True):
    """Open GEBCO bathymetry dataset.

    Downloads the data if not already cached locally.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        Unused parameter for API compatibility.
    step : int, optional
        Subsampling step for latitude and longitude, by default 5.
    mask_land : bool, optional
        Whether to mask land values (elevation >= 0), by default True.

    Returns
    -------
    xarray.Dataset
        Dataset containing the elevation variable.
    """
    fn = DATADIR / filename(dtm=dtm)
    if not fn.is_file():
        retrieve(dtm=dtm)
    lat_slice = slice(None, None, step)
    lon_slice = slice(None, None, step)
    ds = (xr.open_dataset(fn)
            .rename({"lat":"latitude", "lon":"longitude"})
            .sel(latitude=lat_slice, longitude=lon_slice)
            .where(lambda x: x.elevation < 0, drop=False)
    )
    return ds

def open_scene(dtm=None, data_var="elevation"):
    """Open GEBCO bathymetry data as a Satpy Scene.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        Unused parameter for API compatibility.
    data_var : str, optional
        Primary data variable name, by default "elevation".

    Returns
    -------
    satpy.Scene
        Satpy Scene object with loaded bathymetry data.
    """
    fn = DATADIR / filename(dtm=dtm)
    vprint(fn)
    if not fn.is_file():
        retrieve(dtm=dtm)
    scn = satpy.Scene(filenames=[fn], reader='copernicus_ssh')
    scn.load(['adt', 'sla', 'ugos', 'vgos'])
    return scn


def retrieve(dtm=None, force=False, parallel=True):
    """Retrieve GEBCO bathymetry data from CEDA via OPeNDAP.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        Unused parameter for API compatibility.
    force : bool, optional
        Force download even if file exists, by default False.
    parallel : bool, optional
        Unused parameter for API compatibility, by default True.
    """
    fn = DATADIR / filename()
    if fn.is_file() and not force:
        return
    elif force:
        fn.unlink(missing_ok=True)

    dtm = pd.to_datetime(dtm)
    url = ("https://dap.ceda.ac.uk" +
           "/thredds/dodsC/bodc/gebco/global/gebco_2025/" +
           "sub_ice_topography_bathymetry/netcdf/gebco_2025_sub_ice.nc")

    lat_slice = slice(settings["lat1"], settings["lat2"])
    lon_slice = slice(settings["lon1"], settings["lon2"])
    ds = xr.open_dataset(url).sel(lat=lat_slice, lon=lon_slice)
    ds.to_netcdf(fn)
