"""OLCI Level-2 ocean colour data access from EUMETSAT.

This module provides functions to retrieve and process Sentinel-3 OLCI
Level-2 ocean colour products from the EUMETSAT Data Store.

References
----------
Product: https://data.eumetsat.int/product/EO:EUM:DAT:0556
User Guide: https://user.eumetsat.int/resources/user-guides/sentinel-3-ocean-colour-level-2-data-guide
API Examples: https://user.eumetsat.int/resources/user-guides/data-store-detailed-guide#ID-Examples
"""

import glob
import pathlib
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor


import eumdac
import pandas as pd
import requests
import satpy
import xarray as xr
from eumdac.errors import Text
import sqlite3

from . import config
settings = config.settings()

DATADIR = pathlib.Path(settings["data_dir"] + "/eumetsat/OLCI_L2")
DATADIR.mkdir(parents=True, exist_ok=True)
SWATHDIR = DATADIR / "swaths"
SWATHDIR.mkdir(parents=True, exist_ok=True)
TMPDIR = DATADIR / "tmp"
TMPDIR.mkdir(parents=True, exist_ok=True)
DBFILE = DATADIR / "swaths.sqlite"

MAX_PARALLEL_CONNS = 10

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


def bbox_polygon(lat1=None, lat2=None, lon1=None, lon2=None):
    """Create a WKT polygon string for bounding box queries.

    Parameters
    ----------
    lat1 : float, optional
        Minimum latitude. If None, uses settings.
    lat2 : float, optional
        Maximum latitude. If None, uses settings.
    lon1 : float, optional
        Minimum longitude. If None, uses settings.
    lon2 : float, optional
        Maximum longitude. If None, uses settings.

    Returns
    -------
    str
        WKT POLYGON string for the bounding box.
    """
    lon_min = lon1 or settings["lon1"]
    lon_max = lon2 or settings["lon2"]
    lat_min = lat1 or settings["lat1"]
    lat_max = lat2 or settings["lat2"]
    box_wkt = (
        f"POLYGON(({lon_min} {lat_min}, "
        f"{lon_max} {lat_min}, "
        f"{lon_max} {lat_max}, "
        f"{lon_min} {lat_max}, "
        f"{lon_min} {lat_min}))"
    )
    return box_wkt


def _get_db():
    """Get SQLite database connection for swath tracking.

    Creates the database and table if they don't exist.

    Returns
    -------
    sqlite3.Connection
        Database connection object.
    """
    conn = sqlite3.connect(DBFILE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS swaths (
            date TEXT NOT NULL,
            swathfile TEXT NOT NULL,
            UNIQUE(date, swathfile)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_swaths_date ON swaths(date)")
    return conn


def add_swath_fn(dtm, fn):
    """Insert a swath filename for a given date (idempotent).

    Parameters
    ----------
    dtm : str or datetime-like
        The date of the swath.
    fn : str
        The swath filename to add.
    """
    print("Adding data")
    dstr = str(pd.to_datetime(dtm).date())
    conn = _get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO swaths (date, swathfile) VALUES (?, ?)",
            (dstr, fn),
        )
        conn.commit()
    finally:
        conn.close()


def swathlist(dtm="2023-06-03"):
    """Return list of swath filenames for a date.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to query, by default "2023-06-03".

    Returns
    -------
    list of str
        List of swath filenames for the specified date.
    """
    dstr = str(pd.to_datetime(dtm).date())
    conn = _get_db()
    try:
        cur = conn.execute(
            "SELECT swathfile FROM swaths WHERE date = ? ORDER BY swathfile",
            (dstr,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_token():
    """Get EUMETSAT API access token from credentials file.

    Returns
    -------
    eumdac.AccessToken
        Access token for EUMETSAT Data Store API.
    """
    with open("/home/bror/.eumdac/credentials", encoding="utf-8") as f:
        credentials = f.read().split(",")
    token = eumdac.AccessToken(credentials)
    return token


def retrieve(dtm="2023-06-03", force=False, parallel=True):
    """Retrieve OLCI L2 swath data from EUMETSAT Data Store.

    Downloads all swaths intersecting the configured bounding box
    for the specified date.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to retrieve, by default "2023-06-03".
    force : bool, optional
        Force re-download of existing files, by default False.
    parallel : bool, optional
        Use parallel downloads, by default True.

    Notes
    -----
    Alternative collection IDs:
    - 'EO:EUM:DAT:0556' - Reprocessed
    - 'EO:EUM:DAT:0027' - OLCI Level2 image
    """
    dtm = pd.to_datetime(dtm)
    token = get_token()
    datastore = eumdac.DataStore(token)

    collectionID = "EO:EUM:DAT:0407"  # Operational
    collection = datastore.get_collection(collectionID)
    vprint(f"Date: {dtm.date()} \nCollection: {collection.title}")

    # Define the time and space domains
    dtstart = dtm.normalize().to_pydatetime()
    dtend = (
        dtm.normalize() + pd.Timedelta(1, "d") - pd.Timedelta(1, "s")
    ).to_pydatetime()
    geo = bbox_polygon()

    def download_product(product):
        with product.open() as fsrc:
            local_fn = SWATHDIR / fsrc.name
            if local_fn.is_file():
                return
            with open(SWATHDIR / fsrc.name, mode="wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
                add_swath_fn(dtm, fsrc.name)
        vprint(product)

    products = collection.search(geo=geo, dtstart=dtstart, dtend=dtend)
    if parallel:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CONNS) as executor:
            executor.map(download_product, products)
    else:
        for product in products:
            download_product(product)


def rmchildren(directory):
    """Recursively remove all contents of a directory.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to clear.
    """
    for item in directory.glob("*"):
        if item.is_dir():
            rmchildren(item)
            item.rmdir()
        else:
            item.unlink()


def extract_swath_scenes(dtm="2023-06-03", tmpdir=None):
    """Extract and load Satpy scenes from downloaded OLCI swath files.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        The date to process, by default "2023-06-03".
    tmpdir : pathlib.Path, optional
        Temporary directory for extraction. If None, uses TMPDIR.

    Returns
    -------
    list of satpy.Scene
        List of Satpy Scene objects with loaded chlorophyll data.
    """
    tmpdir = tmpdir or TMPDIR
    rmchildren(tmpdir)
    scenes = []

    for zip_fn in swathlist(dtm):
        with zipfile.ZipFile(SWATHDIR / zip_fn) as z:
            z.extractall(tmpdir)
        flist = (tmpdir / zip_fn).with_suffix("").glob("*.nc")
        scn = satpy.Scene(reader="olci_l2", filenames=flist)
        scn.load(["chl_oc4me", "chl_nn"])
        scenes.append(scn)
    return scenes
    combined = scenes[0][0]
    for scn, _ in scenes[1:]:
        combined = combined + scn

    combined.load(["chl_oc4me"])

    return combined


"""
from satpy.multiscene import blend

scn_combined = blend(
    mscn_res,
    datasets=["chlor_a"],
    blend_function="mean",  # or "max", "min", "first", custom
)
Now scn_combined is a single Scene.
4. Save to NetCDF (recommended for L3)
scn_combined.save_datasets(
    writer="netcdf",
    datasets=["chlor_a"],
    filename="chlor_a_L3.nc"
)
Or GeoTIFF:
scn_combined.save_datasets(
    writer="geotiff",
    datasets=["chlor_a"],
    filename="chlor_a_L3.tif"
)
Alternative: Save each resampled swath separately
If you want one file per overpass:
for i, scn in enumerate(mscn_res.scenes):
    scn.save_datasets(
        writer="netcdf",
        datasets=["chlor_a"],
        filename=f"chlor_a_{i}.nc"
    )


#Load Scene
from satpy import Scene

scn = Scene(
    reader="generic_cf",
    filenames=["chlor_a_L3.nc"]
)

scn.load(["chlor_a"])



"""
