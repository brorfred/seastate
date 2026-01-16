"""

REF
---
https://doi.org/10.48670/moi-00165

URLs
----
https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
"""
import pathlib
import copernicusmarine

import pandas as pd
import satpy
import xarray as xr
import copernicusmarine
from satpy import Scene

from processor.area_definitions import rectlinear as rectlin_area
from processor import config, DateInFutureError
settings = config.settings()
#settings = config.settings.from_env("modis_a")

DATADIR = pathlib.Path(settings["data_dir"] + "/copernicus/GlobColour")
DATADIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_nrt_l3-multi-4km_P1D"
filename_prefix = "GLOBCOLOUR"

VERBOSE = True
def vprint(text):
    if VERBOSE:
        print(text)

def filename(dtm="2025-06-03"):
    dtm = pd.to_datetime(dtm)
    return f"copernicus_{filename_prefix}_{dtm.date()}.nc"

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
    vprint(f"Date: {dtm.date()} \nCollection: GlobColour Chl 4km")

    # Define the time and space domains
    dtstart = dtm.normalize().to_pydatetime()
    dtend = (
        dtm.normalize() + pd.Timedelta(1, "d") - pd.Timedelta(1, "s")
    ).to_pydatetime()

    copernicusmarine.subset(
        #dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        dataset_id=DATASET_ID,
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
