"""Tile generation module for oceanographic data products.

This module provides functions to generate slippy map tiles from various
oceanographic data sources including SSH (Sea Surface Height), SST (Sea
Surface Temperature), and chlorophyll concentration.
"""
import pathlib

import numpy as np
import pandas as pd
import sysrsync

from .tilers import rectlinear as rectlin_tiler
from .data_sources import cmems_ssh, ostia as ostia_sst
from .data_sources import globcolour as cmems_globcolour
from .data_sources import gebco_bathy
from . import config
settings = config.settings
from copernicusmarine import CoordinatesOutOfDatasetBounds
import cmasher as cmr

def tiles_exists(id, dtm):
    """Check if tiles already exist for a given product and date.

    Parameters
    ----------
    id : str
        Product identifier (e.g., 'ssh', 'ostia', 'globcolour').
    dtm : str or datetime-like
        The date to check.

    Returns
    -------
    bool
        True if the tile directory exists, False otherwise.
    """
    dtm = str(pd.to_datetime(dtm).date())
    tilepath = pathlib.Path(settings["tile_dir"]) / id / dtm
    return tilepath.is_dir()

def bathy(dtm=None, verbose=True, force=True):
    """Generate bathymetry tiles from GEBCO data.

    Parameters
    ----------
    dtm : str or datetime-like, optional
        Unused parameter for API compatibility.
    verbose : bool, optional
        Enable verbose output, by default True.
    force : bool, optional
        Force regeneration even if tiles exist, by default True.
    """
    tile_base = pathlib.Path(settings["tile_dir"]) / "gebco"
    if tile_base.is_dir() and not force:
        return
    ds = gebco_bathy.open_dataset(dtm=dtm)
    tile_base.mkdir(parents=True, exist_ok=True)

    rectlin_tiler.VERBOSE = verbose
    generator = rectlin_tiler.SlippyTileGenerator(
        min_lat=float(ds.latitude.min()),
        max_lat=float(ds.latitude.max()),
        min_lon=float(ds.longitude.min()),
        max_lon=float(ds.longitude.max())
    )
    generator.generate_tiles(np.squeeze(ds["elevation"].data),
                             ds.latitude.data,
                             ds.longitude.data,
                             tile_base,
                             settings["zoom_levels"],
                             cmap=cmr.ocean,
                             levels=np.arange(-6000,100,100),
                             vmin=-6000,
                             vmax=0,
                             add_contour_lines=True,
                             contour_levels=np.arange(-6000,0,500))


def ssh(dtm, verbose=True, force=True):
    """Generate SSH (Sea Surface Height) tiles for a given date.

    Parameters
    ----------
    dtm : str or datetime-like
        The date to generate tiles for.
    verbose : bool, optional
        Enable verbose output, by default True.
    force : bool, optional
        Force regeneration even if tiles exist, by default True.
    """
    if tiles_exists("ssh", dtm) and not force:
        return
    rectlin_tiler.VERBOSE = verbose
    dtm = pd.to_datetime(dtm, utc=True)
    try:
        ds = cmems_ssh.open_dataset(dtm=dtm)
    except CoordinatesOutOfDatasetBounds:
        print(f"  {dtm.date()} failed for SSH")
        return
    tile_base = pathlib.Path(settings["tile_dir"]) / "ssh" / str(dtm.date())
    tile_base.mkdir(parents=True, exist_ok=True)

    generator = rectlin_tiler.SlippyTileGenerator(
        min_lat=float(ds.latitude.min()),
        max_lat=float(ds.latitude.max()),
        min_lon=float(ds.longitude.min()),
        max_lon=float(ds.longitude.max())
    )
    generator.generate_tiles(np.squeeze(ds.sla.data),
                             ds.latitude.data,
                             ds.longitude.data,
                             tile_base,
                             settings["zoom_levels"],
                             cmap="RdBu",
                             vmin=-0.75,
                             vmax=0.75)


def sst(dtm, verbose=True, force=True):
    """Generate SST (Sea Surface Temperature) tiles for a given date.

    Alias for :func:`ostia`.

    Parameters
    ----------
    dtm : str or datetime-like
        The date to generate tiles for.
    verbose : bool, optional
        Enable verbose output, by default True.
    force : bool, optional
        Force regeneration even if tiles exist, by default True.
    """
    ostia(dtm, verbose, force)


def ostia(dtm, verbose=True, force=True):
    """Generate OSTIA SST tiles for a given date.

    Uses the OSTIA (Operational Sea Surface Temperature and Ice Analysis)
    dataset from Copernicus Marine Service.

    Parameters
    ----------
    dtm : str or datetime-like
        The date to generate tiles for.
    verbose : bool, optional
        Enable verbose output, by default True.
    force : bool, optional
        Force regeneration even if tiles exist, by default True.
    """
    if tiles_exists("ostia", dtm) and not force:
        return
    rectlin_tiler.VERBOSE = verbose
    dtm = pd.to_datetime(dtm, utc=True)
    try:
        ds = ostia_sst.open_dataset(dtm=dtm)
    except CoordinatesOutOfDatasetBounds:
        print(f"  {dtm.date()} failed for SST")
        return
    tile_base = pathlib.Path(settings["tile_dir"]) / "ostia" / str(dtm.date())
    tile_base.mkdir(parents=True, exist_ok=True)

    generator = rectlin_tiler.SlippyTileGenerator(
        min_lat=float(ds.latitude.min()),
        max_lat=float(ds.latitude.max()),
        min_lon=float(ds.longitude.min()),
        max_lon=float(ds.longitude.max())
    )
    generator.generate_tiles(np.squeeze(ds.analysed_sst.data)-273.15,
                             ds.latitude.data,
                             ds.longitude.data,
                             tile_base,
                             settings["zoom_levels"],
                             cmap="viridis",
                             vmin=10,
                             vmax=28)


def globcolour(dtm, verbose=True, force=True):
    """Generate GlobColour chlorophyll tiles for a given date.

    Uses the GlobColour merged chlorophyll-a product from
    Copernicus Marine Service. Values are log-transformed for display.

    Parameters
    ----------
    dtm : str or datetime-like
        The date to generate tiles for.
    verbose : bool, optional
        Enable verbose output, by default True.
    force : bool, optional
        Force regeneration even if tiles exist, by default True.
    """
    if tiles_exists("globcolour", dtm) and not force:
        return
    rectlin_tiler.VERBOSE = verbose
    dtm = pd.to_datetime(dtm, utc=True)
    try:
        ds = cmems_globcolour.open_dataset(dtm=dtm)
    except CoordinatesOutOfDatasetBounds:
        print(f"  {dtm.date()} failed for globcolour")
        return
    tile_base = pathlib.Path(settings["tile_dir"]) / "globcolour" / str(dtm.date())
    tile_base.mkdir(parents=True, exist_ok=True)

    generator = rectlin_tiler.SlippyTileGenerator(
        min_lat=float(ds.latitude.min()),
        max_lat=float(ds.latitude.max()),
        min_lon=float(ds.longitude.min()),
        max_lon=float(ds.longitude.max())
    )
    generator.generate_tiles(np.log(np.squeeze(ds.CHL.data)),
                             ds.latitude.data,
                             ds.longitude.data,
                             tile_base,
                             settings["zoom_levels"],
                             cmap="nipy_spectral",
                             vmin=-4.6,
                             vmax=4.6,
                             levels=50)


def all(dtm, force=False, verbose=False):
    """Generate all tile products for a given date.

    Generates SSH, SST, and GlobColour tiles.

    Parameters
    ----------
    dtm : str or datetime-like
        The date to generate tiles for.
    force : bool, optional
        Force regeneration even if tiles exist, by default False.
    verbose : bool, optional
        Enable verbose output, by default False.
    """
    print("Process SSH tiles")
    ssh(dtm, verbose=verbose, force=force)
    print("Process SST tiles")
    sst(dtm, verbose=verbose, force=force)
    print("Process globcolour tiles")
    globcolour(dtm, verbose=verbose, force=force)


def sync():
    """Synchronize local tiles to remote server via rsync.

    Uses sysrsync to transfer tiles from the local tile directory
    to the configured remote server.
    """
    local_tiledir = settings["tile_dir"]
    remote_tile_dir = settings["remote_tile_dir"]
    sysrsync.run(source=local_tiledir,
                 destination=remote_tile_dir,
                 destination_ssh='tvarminne',
                 options=['-az'],
                 sync_source_contents=True,
                 strict=True,
                 )
