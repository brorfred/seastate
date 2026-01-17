"""Slippy Tile Generator for Satpy Scenes.

This module generates filled contour slippy map tiles from satpy scenes
with parallel processing. Uses mercantile for robust tile calculations.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import tri
from PIL import Image
import mercantile
import io

VERBOSE = True


def vprint(string):
    """Print string if verbose mode is enabled.

    Parameters
    ----------
    string : str
        Text to print.
    """
    if VERBOSE:
        print(string)


class SlippyTileGenerator:
    """Generate slippy map tiles from satpy scenes with filled contours.

    Parameters
    ----------
    min_lat : float, optional
        Minimum latitude (southern boundary), by default -15.
    max_lat : float, optional
        Maximum latitude (northern boundary), by default 55.
    min_lon : float, optional
        Minimum longitude (western boundary), by default -75.
    max_lon : float, optional
        Maximum longitude (eastern boundary), by default -5.

    Attributes
    ----------
    TILE_SIZE : int
        Size of output tiles in pixels (256).
    min_lat, max_lat : float
        Latitude bounds.
    min_lon, max_lon : float
        Longitude bounds.
    """

    TILE_SIZE = 256

    def __init__(self, min_lat: float = -15, max_lat: float = 55,
                 min_lon: float = -75, max_lon: float = -5):
        """Initialize tile generator with geographic bounds.

        Parameters
        ----------
        min_lat : float, optional
            Minimum latitude (southern boundary), by default -15.
        max_lat : float, optional
            Maximum latitude (northern boundary), by default 55.
        min_lon : float, optional
            Minimum longitude (western boundary), by default -75.
        max_lon : float, optional
            Maximum longitude (eastern boundary), by default -5.
        """
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

    def get_tiles_for_bounds(self, zoom: int):
        """Get all tiles that intersect with the geographic bounds.

        Parameters
        ----------
        zoom : int
            Zoom level for tile calculation.

        Returns
        -------
        list of mercantile.Tile
            List of tiles covering the bounding box.
        """
        # Get tiles that cover the bounding box
        tiles = list(mercantile.tiles(
            self.min_lon, self.min_lat,
            self.max_lon, self.max_lat,
            zooms=zoom
        ))
        return tiles

    def diagnose_coverage(self, scene_lats: np.ndarray, scene_lons: np.ndarray, zoom: int):
        """Diagnose data coverage across tiles at a given zoom level.

        Parameters
        ----------
        scene_lats : numpy.ndarray
            Latitude coordinates of the scene data.
        scene_lons : numpy.ndarray
            Longitude coordinates of the scene data.
        zoom : int
            Zoom level to diagnose.
        """
        # Get actual data bounds
        if scene_lats.ndim == 1:
            data_min_lat = scene_lats.min()
            data_max_lat = scene_lats.max()
            data_min_lon = scene_lons.min()
            data_max_lon = scene_lons.max()
        else:
            data_min_lat = scene_lats.min()
            data_max_lat = scene_lats.max()
            data_min_lon = scene_lons.min()
            data_max_lon = scene_lons.max()

        vprint(f"\n=== Diagnostic Info for Zoom {zoom} ===")
        vprint(f"Generator bounds: lat=[{self.min_lat}, {self.max_lat}], lon=[{self.min_lon}, {self.max_lon}]")
        vprint(f"Data bounds: lat=[{data_min_lat:.2f}, {data_max_lat:.2f}], lon=[{data_min_lon:.2f}, {data_max_lon:.2f}]")

        tiles = self.get_tiles_for_bounds(zoom)
        vprint(f"Total tiles: {len(tiles)}")

        if tiles:
            sample_tile = tiles[0]
            bounds = mercantile.bounds(sample_tile.x, sample_tile.y, sample_tile.z)
            vprint(f"Sample tile {sample_tile.x}/{sample_tile.y}: lat=[{bounds.south:.2f}, {bounds.north:.2f}], lon=[{bounds.west:.2f}, {bounds.east:.2f}]")
        vprint("="*40)

    def generate_tiles(self, scene_data: np.ndarray, scene_lats: np.ndarray,
                      scene_lons: np.ndarray, output_dir: str, zoom_levels: list,
                      num_workers: int = 10, cmap: str = 'RdBu',
                      levels: int = 20, vmin: Optional[float] = None,
                      vmax: Optional[float] = None):
        """Generate tiles for multiple zoom levels in parallel.

        Parameters
        ----------
        scene_data : numpy.ndarray
            2D array of scene values.
        scene_lats : numpy.ndarray
            1D or 2D array of latitudes (can be ascending or descending).
        scene_lons : numpy.ndarray
            1D or 2D array of longitudes (can be ascending or descending).
        output_dir : str
            Output directory for tiles.
        zoom_levels : list of int
            List of zoom levels to generate.
        num_workers : int, optional
            Number of parallel workers, by default 10.
        cmap : str, optional
            Matplotlib colormap name, by default 'RdBu'.
        levels : int, optional
            Number of contour levels, by default 20.
        vmin : float, optional
            Minimum value for colormap. If None, auto-calculated.
        vmax : float, optional
            Maximum value for colormap. If None, auto-calculated.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert 1D coordinates to 2D grid if needed
        if scene_lats.ndim == 1 and scene_lons.ndim == 1:
            # Check if coordinates are descending and need to be flipped
            lat_descending = scene_lats[0] > scene_lats[-1]
            lon_descending = scene_lons[0] > scene_lons[-1]

            # Work with copies to avoid modifying original data
            lats_1d = scene_lats[::-1] if lat_descending else scene_lats.copy()
            lons_1d = scene_lons[::-1] if lon_descending else scene_lons.copy()
            data_work = scene_data.copy()

            # Flip data arrays to match coordinate ordering
            if lat_descending:
                data_work = np.flip(data_work, axis=0)
            if lon_descending:
                data_work = np.flip(data_work, axis=1)

            # Create 2D grids
            lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
        else:
            # Already 2D
            lon_grid = scene_lons
            lat_grid = scene_lats
            data_work = scene_data.copy()

        # Prepare scene data
        valid_mask = ~np.isnan(data_work)
        if vmin is None:
            vmin = np.nanmin(data_work)
        if vmax is None:
            vmax = np.nanmax(data_work)

        # Flatten arrays for triangulation using the 2D grids
        lats_flat = lat_grid[valid_mask]
        lons_flat = lon_grid[valid_mask]
        data_flat = data_work[valid_mask]

        vprint(f"Processing {len(data_flat)} valid data points")
        vprint(f"Data range: {vmin:.4f} to {vmax:.4f}")

        for zoom in zoom_levels:
            vprint(f"\nGenerating tiles for zoom level {zoom}")

            # Run diagnostics
            self.diagnose_coverage(scene_lats, scene_lons, zoom)

            # Get all tiles using mercantile
            tiles = self.get_tiles_for_bounds(zoom)

            # Create zoom directory
            zoom_dir = output_path / str(zoom)
            zoom_dir.mkdir(exist_ok=True)

            # Create x directories
            unique_x = set(tile.x for tile in tiles)
            for x in unique_x:
                (zoom_dir / str(x)).mkdir(exist_ok=True)

            vprint(f"Total tiles to generate: {len(tiles)}")

            # Process tiles in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        self._generate_single_tile,
                        tile.z, tile.x, tile.y,
                        lats_flat, lons_flat, data_flat,
                        str(output_path), cmap, levels, vmin, vmax
                    ): (tile.x, tile.y) for tile in tiles
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 100 == 0 or completed == len(tiles):
                        vprint(f"Progress: {completed}/{len(tiles)} tiles")

                    try:
                        future.result()
                    except Exception as e:
                        x, y = futures[future]
                        print(f"Error generating tile {zoom}/{x}/{y}: {e}")

            vprint(f"Completed zoom level {zoom}")

    @staticmethod
    def _generate_single_tile(zoom: int, x: int, y: int,
                             lats: np.ndarray, lons: np.ndarray, data: np.ndarray,
                             output_dir: str, cmap: str, levels: int,
                             vmin: float, vmax: float):
        """Generate a single tile (called in parallel).

        Parameters
        ----------
        zoom : int
            Zoom level.
        x : int
            Tile x coordinate.
        y : int
            Tile y coordinate.
        lats : numpy.ndarray
            Flattened latitude values for valid data points.
        lons : numpy.ndarray
            Flattened longitude values for valid data points.
        data : numpy.ndarray
            Flattened data values for valid points.
        output_dir : str
            Output directory path.
        cmap : str
            Matplotlib colormap name.
        levels : int
            Number of contour levels.
        vmin : float
            Minimum value for colormap.
        vmax : float
            Maximum value for colormap.
        """
        # Get tile bounds using mercantile
        bounds = mercantile.bounds(x, y, zoom)
        lon_left = bounds.west
        lon_right = bounds.east
        lat_bottom = bounds.south
        lat_top = bounds.north

        # Calculate tile size in degrees for adaptive buffer
        tile_height = lat_top - lat_bottom
        tile_width = lon_right - lon_left
        # Use larger buffer (50% of tile size) to ensure good interpolation
        buffer_lat = tile_height * 0.5
        buffer_lon = tile_width * 0.5

        # Filter data within tile bounds with buffer for edge interpolation
        mask = (
            (lats >= lat_bottom - buffer_lat) & (lats <= lat_top + buffer_lat) &
            (lons >= lon_left - buffer_lon) & (lons <= lon_right + buffer_lon)
        )

        tile_path = Path(output_dir) / str(zoom) / str(x) / f"{y}.png"

        if not np.any(mask):
            # Generate empty transparent tile
            img = Image.new('RGBA', (SlippyTileGenerator.TILE_SIZE,
                                     SlippyTileGenerator.TILE_SIZE), (0, 0, 0, 0))
            img.save(tile_path)
            return

        # Extract tile data
        tile_lons = lons[mask]
        tile_lats = lats[mask]
        tile_data = data[mask]

        # Need sufficient points for triangulation
        if len(tile_data) < 3:
            img = Image.new('RGBA', (SlippyTileGenerator.TILE_SIZE,
                                     SlippyTileGenerator.TILE_SIZE), (0, 0, 0, 0))
            img.save(tile_path)
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(SlippyTileGenerator.TILE_SIZE / 100,
                                        SlippyTileGenerator.TILE_SIZE / 100),
                              dpi=100)
        ax.set_xlim(lon_left, lon_right)
        ax.set_ylim(lat_bottom, lat_top)
        ax.set_aspect('equal')
        ax.axis('off')

        # Create triangulation and filled contours
        try:
            triang = tri.Triangulation(tile_lons, tile_lats)

            # Mask out triangles that are too large (gaps in data)
            # Calculate the median edge length to identify sparse areas
            x = triang.x[triang.triangles]
            y = triang.y[triang.triangles]

            # Calculate triangle areas using cross product
            edge1_x = x[:, 1] - x[:, 0]
            edge1_y = y[:, 1] - y[:, 0]
            edge2_x = x[:, 2] - x[:, 0]
            edge2_y = y[:, 2] - y[:, 0]
            areas = 0.5 * np.abs(edge1_x * edge2_y - edge1_y * edge2_x)

            # Mask triangles with area > 3x median (likely spanning NaN gaps)
            median_area = np.median(areas)
            mask = areas > 3 * median_area
            triang.set_mask(mask)

            contour_levels = np.linspace(vmin, vmax, levels)
            ax.tricontourf(triang, tile_data, levels=contour_levels,
                          cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
        except Exception:
            # If triangulation fails, create empty tile
            plt.close(fig)
            img = Image.new('RGBA', (SlippyTileGenerator.TILE_SIZE,
                                     SlippyTileGenerator.TILE_SIZE), (0, 0, 0, 0))
            img.save(tile_path)
            return

        # Remove padding
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight',
                   pad_inches=0)
        plt.close(fig)

        # Save as PNG
        buf.seek(0)
        img = Image.open(buf)
        img = img.resize((SlippyTileGenerator.TILE_SIZE,
                         SlippyTileGenerator.TILE_SIZE), Image.LANCZOS)
        img.save(tile_path)
        buf.close()
