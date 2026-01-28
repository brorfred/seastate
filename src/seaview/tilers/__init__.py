"""Tile generation utilities for map visualization.

This package contains modules for generating slippy map tiles from
oceanographic data using various rendering and processing techniques.
"""

from .rectlinear import SlippyTileGenerator, cruise_tiles
from .olci_swath import OLCISwathTileGenerator, olci_swath_tiles
