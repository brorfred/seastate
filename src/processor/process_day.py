import pathlib

import numpy as np
import pandas as pd
import sysrsync

from .tilers import rectlinear as rectlin_tiler
from .data_sources import cmems_ssh, ostia, globcolour
from . import config
settings = config.settings()
from copernicusmarine import CoordinatesOutOfDatasetBounds

def tile_ssh(dtm, verbose=True):
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

def tile_sst(dtm, verbose=True):
    rectlin_tiler.VERBOSE = verbose
    dtm = pd.to_datetime(dtm, utc=True)
    try:
        ds = ostia.open_dataset(dtm=dtm)
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

def tile_globcolour(dtm, verbose=True):
    rectlin_tiler.VERBOSE = verbose
    dtm = pd.to_datetime(dtm, utc=True)
    try:
        ds = globcolour.open_dataset(dtm=dtm)
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

def all(dtm, verbose=False):
    print("Process SSH tiles")
    tile_ssh(dtm, verbose=verbose)
    print("Process SST tiles")
    tile_sst(dtm, verbose=verbose)
    print("Process globcolour tiles")
    tile_globcolour(dtm, verbose=verbose)

def sync():
    local_tiledir = settings["tile_dir"]
    remote_tile_dir = settings["remote_tile_dir"]
    sysrsync.run(source=local_tiledir,
                 destination=remote_tile_dir,
                 destination_ssh='tvarminne',
                 options=['-az'],
                 sync_source_contents=True,
                 strict=True,
                 )
