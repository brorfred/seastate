"""OSTIA Sea Surface Temperature data access.

This module provides functions to retrieve and access OSTIA
(Operational Sea Surface Temperature and Ice Analysis) data
from the Copernicus Marine Service.

References
----------
DOI: https://doi.org/10.48670/moi-00165
Product: https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
"""
import os
import glob
import pathlib
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
import copernicusmarine

import pandas as pd
import satpy
import xarray as xr
import copernicusmarine
from satpy import Scene

from seastate.area_definitions import rectlinear as rectlin_area
from seastate import config
settings = config.settings

DATADIR = pathlib.Path(settings["data_dir"] + "/copernicus/OSTIA")
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


def filename(dtm="2025-06-03"):
    """Generate filename for OSTIA data file.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date for the filename, by default "2025-06-03".

    Returns
    -------
    str
        Filename in format 'copernicus_OSTIA_YYYY-MM-DD.nc'.
    """
    dtm = pd.to_datetime(dtm)
    return f"copernicus_OSTIA_{dtm.date()}.nc"


def open_dataset(dtm="2025-06-03", force=False):
    """Open OSTIA SST dataset for a given date.

    Downloads the data if not already cached locally.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to retrieve, by default "2025-06-03".
    force : bool, optional
        Force download even if file exists, by default False.

    Returns
    -------
    xarray.Dataset
        Dataset containing SST variables.
    """
    fn = DATADIR / filename(dtm=dtm)
    if not fn.is_file():
        retrieve(dtm=dtm, force=force)
    return xr.open_dataset(DATADIR / filename(dtm=dtm))


def open_scene(dtm="2025-06-03", data_var="sla"):
    """Open OSTIA data as a Satpy Scene.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to retrieve, by default "2025-06-03".
    data_var : str, optional
        Primary data variable name, by default "sla".

    Returns
    -------
    satpy.Scene
        Satpy Scene object with loaded variables.
    """
    fn = DATADIR / filename(dtm=dtm)
    vprint(fn)
    if not fn.is_file():
        retrieve(dtm=dtm)
    scn = Scene(filenames=[fn], reader='copernicus_ssh')
    scn.load(['adt', 'sla', 'ugos', 'vgos'])
    return scn


def retrieve(dtm="2025-06-03", force=False, parallel=True):
    """Retrieve OSTIA SST data from Copernicus Marine Service.

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
    dtm = pd.to_datetime(dtm, utc=True)
    vprint(f"Date: {dtm.date()} \nCollection: Sea Surface Temperature")

    # Define the time and space domains
    dtstart = dtm.normalize().to_pydatetime()
    dtend = (
        dtm.normalize() + pd.Timedelta(1, "d") - pd.Timedelta(1, "s")
    ).to_pydatetime()

    copernicusmarine.subset(
        #dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        username = settings.get("cmems_login"),
        password = settings.get("cmems_password"),
        #variables=["uo", "vo"],
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
