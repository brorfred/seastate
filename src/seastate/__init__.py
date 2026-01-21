"""Processor module for generating and syncing map tiles.

This module provides functions to process oceanographic data and generate
slippy map tiles for various data products (SSH, SST, chlorophyll).
"""
import pandas as pd

from . import config, tile, layer_config
settings = config.settings


class DateInFutureError(Exception):
    """Exception raised when a requested date is in the future.

    This error is raised when attempting to process data for a date
    that has not yet occurred.
    """
    pass


def day(dtm, force=False, verbose=False):
    """Process tiles for a specific day.

    Parameters
    ----------
    dtm : str or datetime-like
        The date to process tiles for.

    Notes
    -----
    This function is currently a placeholder and not implemented.
    """
    tile.all(dtm, force=force, verbose=False)


def today(force=False, sync=True):
    """Process tiles for today's date.

    Generates all tile products for the current date and optionally
    syncs to remote server if configured.

    Parameters
    ----------
    force : bool, optional
        Force regeneration even if tiles exist, by default False.
    sync : bool, optional
        Sync tiles to remote server after generation, by default True.
    """
    dtm = pd.Timestamp.now().normalize()
    tile.all(dtm, force=force)
    if config.settings.get("remote_sync") and sync:
        tile.sync()
        layer_config.sync()



def yesterday(force=False, sync=True):
    """Process tiles for yesterday's date.

    Generates all tile products for yesterday and optionally syncs
    tiles and layer configuration to remote server if configured.

    Parameters
    ----------
    force : bool, optional
        Force regeneration even if tiles exist, by default False.
    sync : bool, optional
        Sync tiles to remote server after generation, by default True.
    """
    dtm = pd.Timestamp.now().normalize()-pd.Timedelta(1,"D")
    tile.all(dtm, force=False)
    if config.settings.get("remote_sync") and sync:
        print("Sync tiles")
        tile.sync()
        layer_config.sync()


def last_days(days=7, sync=True):
    """Process tiles for the last N days.

    Processes tiles for each day in the range from (today - days) to today,
    inclusive. Syncs to remote server after processing if configured.

    Parameters
    ----------
    days : int, optional
        Number of days to process, by default 7.
    sync : bool, optional
        Sync tiles to remote server after generation, by default True.
    """
    dtm1 = pd.Timestamp.now().normalize()-pd.Timedelta(days,"D")
    dtm2 = pd.Timestamp.now().normalize()
    for dtm in pd.date_range(dtm1, dtm2):
        tile.all(dtm)
    if config.settings.get("remote_sync") and sync:
        tile.sync()
        layer_config.sync()
