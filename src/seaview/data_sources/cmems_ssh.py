"""CMEMS Sea Surface Height (SSH) data access.

This module provides functions to retrieve and access Sea Surface Height
data from the Copernicus Marine Service (CMEMS).

References
----------
https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046/description
"""
import pathlib
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
import satpy
import xarray as xr
import copernicusmarine

from ..area_definitions import rectlinear as rectlin_area
from .. import config
settings = config.settings

DATADIR = pathlib.Path(settings["data_dir"] + "/copernicus/SSH")
DATADIR.mkdir(parents=True, exist_ok=True)

def vprint(text):
    """Print text if verbose mode is enabled.

    Parameters
    ----------
    text : str
        Text to print.
    """
    if settings.get("verbose"):
        print(text)


def filename(dtm="2025-06-03"):
    """Generate filename for SSH data file.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date for the filename, by default "2025-06-03".

    Returns
    -------
    str
        Filename in format 'copernicus_SSH_YYYY-MM-DD.nc'.
    """
    dtm = pd.to_datetime(dtm)
    return f"copernicus_SSH_{dtm.date()}.nc"


def open_dataset(dtm="2025-06-03", _pause=0, _retry=0, force=False):
    """Open SSH dataset for a given date.

    Downloads the data if not already cached locally.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to retrieve, by default "2025-06-03".

    Returns
    -------
    xarray.Dataset
        Dataset containing SSH variables (sla, adt, ugos, vgos).
    """
    fn = DATADIR / filename(dtm=dtm)
    vprint(f"SSH Data file: {fn}")
    if force or not fn.is_file():
        retrieve(dtm=dtm, force=force)
    if _retry > 3:
        raise OSError(f"Failed to open {fn} after {_retry} attempts.")
    if _retry > 0:
        vprint("Failed to open the file, will try again")
    time.sleep(_pause)
    try:
        ds = xr.open_dataset(fn, engine="h5netcdf")
    except (OSError,):
        ds = open_dataset(dtm=dtm, _pause=5, _retry=_retry+1)
    return ds

def open_scene(dtm="2025-06-03", data_var="sla"):
    """Open SSH data as a Satpy Scene.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to retrieve, by default "2025-06-03".
    data_var : str, optional
        Primary data variable name, by default "sla".

    Returns
    -------
    satpy.Scene
        Satpy Scene object with loaded SSH variables.
    """
    fn = DATADIR / filename(dtm=dtm)
    vprint(fn)
    if not fn.is_file():
        retrieve(dtm=dtm)
    scn = satpy.Scene(filenames=[fn], reader='copernicus_ssh')
    scn.load(['adt', 'sla', 'ugos', 'vgos'])
    return scn


def retrieve(dtm="2025-06-03", force=False, parallel=True):
    """Retrieve SSH data from Copernicus Marine Service.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to retrieve, by default "2025-06-03".
    force : bool, optional
        Force download even if file exists, by default False.
    parallel : bool, optional
        Use parallel download (unused), by default True.
    """
    if ((DATADIR / filename(dtm)).is_file() and not force):
        return
    elif force:
        (DATADIR / filename(dtm)).unlink(missing_ok=True)
    dtm = pd.to_datetime(dtm)
    vprint(f"Date: {dtm.date()} \nCollection: Sea Surface Height")

    # Define the time and space domains
    dtstart = dtm.normalize().to_pydatetime()
    dtend = (
        dtm.normalize() + pd.Timedelta(1, "d") - pd.Timedelta(1, "s")
    ).to_pydatetime()

    copernicusmarine.subset(
        #dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        #variables=["uo", "vo"],
        username = settings.get("cmems_login"),
        password = settings.get("cmems_password"),
        minimum_longitude=settings["lon1"],
        maximum_longitude=settings["lon2"],
        minimum_latitude=settings["lat1"],
        maximum_latitude=settings["lat2"],
        start_datetime=dtstart,
        end_datetime=dtend,
        #minimum_depth=0,
        #maximum_depth=30,
        output_filename = filename(dtm),
        output_directory = DATADIR
    )
