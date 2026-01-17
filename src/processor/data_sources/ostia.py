"""

REF
---
https://doi.org/10.48670/moi-00165

URLs
----
https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
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
#from satpy.dataset import DataID, Dataset  #0.59

#os.environ["SATPY_CONFIG_PATH"] =
#from .readers import copernicus_ssh
#satpy.readers.copernicus_ssh = copernicus_ssh
#satpy.config.set(config_path=[str(pathlib.Path(__file__).parent)+"/",])

from processor.area_definitions import rectlinear as rectlin_area
from processor import config
settings = config.settings
#settings = config.settings.from_env("modis_a")

DATADIR = pathlib.Path(settings["data_dir"] + "/copernicus/OSTIA")
DATADIR.mkdir(parents=True, exist_ok=True)


VERBOSE = True
def vprint(text):
    if VERBOSE:
        print(text)

def filename(dtm="2025-06-03"):
    dtm = pd.to_datetime(dtm)
    return f"copernicus_OSTIA_{dtm.date()}.nc"

def open_dataset(dtm="2025-06-03", force=False):
    fn = DATADIR / filename(dtm=dtm)
    if not fn.is_file():
        retrieve(dtm=dtm, force=force)
    return xr.open_dataset(DATADIR / filename(dtm=dtm))

def open_scene(dtm="2025-06-03", data_var="sla"):
    fn = DATADIR / filename(dtm=dtm)
    vprint(fn)
    if not fn.is_file():
        retrieve(dtm=dtm)
    scn = Scene(filenames=[fn], reader='copernicus_ssh')
    scn.load(['adt', 'sla', 'ugos', 'vgos'])
    return scn

def retrieve(dtm="2025-06-03", force=False, parallel=True):
    """
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
