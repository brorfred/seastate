"""Slippy Tile Generator for Rectilinear Data.

This module generates filled contour slippy map tiles from rectilinear
(gridded) data with parallel processing. Uses mercantile for robust tile
calculations.

Data is transformed to Web Mercator (EPSG:3857) before triangulation to
ensure proper alignment with slippy map tiles.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import tri
from PIL import Image
import mercantile
import io
from pyproj import Transformer

from .utils import filter_small_contours
from ..utils import vprint
from .. import config

settings = config.settings

# Web Mercator transformer (lon/lat to x/y meters)
_transformer_to_webmerc = Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True
)


def lonlat_to_webmercator(lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform longitude/latitude arrays to Web Mercator coordinates.

    Parameters
    ----------
    lons : numpy.ndarray
        Longitude values in degrees.
    lats : numpy.ndarray
        Latitude values in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        (x, y) coordinates in Web Mercator meters.
    """
    x, y = _transformer_to_webmerc.transform(lons, lats)
    return x.astype(np.float32), y.astype(np.float32)


class SlippyTileGenerator:
    """Generate slippy map tiles from rectilinear data with filled contours.

    Data is transformed to Web Mercator (EPSG:3857) before triangulation
    to ensure proper alignment with slippy map tiles.

    Parameters
    ----------
    min_lat : float, optional
        Minimum latitude (southern boundary). If None, uses settings.
    max_lat : float, optional
        Maximum latitude (northern boundary). If None, uses settings.
    min_lon : float, optional
        Minimum longitude (western boundary). If None, uses settings.
    max_lon : float, optional
        Maximum longitude (eastern boundary). If None, uses settings.

    Attributes
    ----------
    TILE_SIZE : int
        Size of output tiles in pixels (256).
    """

    TILE_SIZE = 256

    def __init__(self, min_lat: float = None, max_lat: float = None,
                 min_lon: float = None, max_lon: float = None):
        """Initialize tile generator with geographic bounds."""
        self.min_lat = min_lat if min_lat is not None else settings.get("lat1", -15)
        self.max_lat = max_lat if max_lat is not None else settings.get("lat2", 55)
        self.min_lon = min_lon if min_lon is not None else settings.get("lon1", -75)
        self.max_lon = max_lon if max_lon is not None else settings.get("lon2", -5)

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
        tiles = list(mercantile.tiles(
            self.min_lon, self.min_lat,
            self.max_lon, self.max_lat,
            zooms=zoom
        ))
        return tiles

    def generate_tiles(self, scene_data: np.ndarray, scene_lats: np.ndarray,
                       scene_lons: np.ndarray, output_dir: str, zoom_levels: List[int],
                       num_workers: int = 10, cmap: str = 'RdBu',
                       levels: int = 20, vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       add_contour_lines: bool = False, contour_levels: int = 5):
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
        add_contour_lines : bool, optional
            Whether to add contour lines on top of filled contours, by default False.
        contour_levels : int, optional
            Number of contour line levels, by default 5.
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
            vmin = float(np.nanmin(data_work))
        if vmax is None:
            vmax = float(np.nanmax(data_work))

        # Flatten arrays for triangulation using the 2D grids
        lats_flat = lat_grid[valid_mask].astype(np.float32)
        lons_flat = lon_grid[valid_mask].astype(np.float32)
        data_flat = data_work[valid_mask].astype(np.float32)

        # Transform to Web Mercator for proper tile alignment
        vprint("Transforming coordinates to Web Mercator...")
        x_wm, y_wm = lonlat_to_webmercator(lons_flat, lats_flat)

        vprint(f"Processing {len(data_flat)} valid data points")
        vprint(f"Data range: {vmin:.4f} to {vmax:.4f}")

        for zoom in zoom_levels:
            vprint(f"\nGenerating tiles for zoom level {zoom}")

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
                        _generate_single_tile,
                        tile.z, tile.x, tile.y,
                        x_wm, y_wm, data_flat,
                        str(output_path), cmap, levels, vmin, vmax,
                        add_contour_lines, contour_levels
                    ): (tile.x, tile.y) for tile in tiles
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    if completed % 100 == 0 or completed == len(tiles):
                        vprint(f"Progress: {completed}/{len(tiles)} tiles", level=1)

                    try:
                        future.result()
                    except Exception as e:
                        x, y = futures[future]
                        print(f"Error generating tile {zoom}/{x}/{y}: {e}")

            vprint(f"Completed zoom level {zoom}")


def _generate_single_tile(
    zoom: int, tile_x: int, tile_y: int,
    x_coords: np.ndarray, y_coords: np.ndarray, data: np.ndarray,
    output_dir: str, cmap: str, levels: int,
    vmin: float, vmax: float,
    add_contour_lines: bool, contour_levels: int
):
    """Generate a single tile using Web Mercator coordinates.

    Parameters
    ----------
    zoom : int
        Zoom level.
    tile_x : int
        Tile x coordinate.
    tile_y : int
        Tile y coordinate.
    x_coords : numpy.ndarray
        Flattened x values in Web Mercator meters.
    y_coords : numpy.ndarray
        Flattened y values in Web Mercator meters.
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
    add_contour_lines : bool
        Whether to add contour lines on top of filled contours.
    contour_levels : int
        Number of contour line levels.
    """
    TILE_SIZE = 256
    tile_path = Path(output_dir) / str(zoom) / str(tile_x) / f"{tile_y}.png"

    # Get tile bounds in Web Mercator meters
    xy_bounds = mercantile.xy_bounds(tile_x, tile_y, zoom)
    x_left, x_right = xy_bounds.left, xy_bounds.right
    y_bottom, y_top = xy_bounds.bottom, xy_bounds.top

    # Buffer for edge interpolation (15% of tile size)
    tile_height = y_top - y_bottom
    tile_width = x_right - x_left
    buffer_y = tile_height * 0.15
    buffer_x = tile_width * 0.15

    buffered_left = x_left - buffer_x
    buffered_right = x_right + buffer_x
    buffered_bottom = y_bottom - buffer_y
    buffered_top = y_top + buffer_y

    # Filter data within tile bounds with buffer
    mask = (
        (x_coords >= buffered_left) & (x_coords <= buffered_right) &
        (y_coords >= buffered_bottom) & (y_coords <= buffered_top)
    )

    if not np.any(mask):
        img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        img.save(tile_path)
        return

    tile_x_coords = x_coords[mask]
    tile_y_coords = y_coords[mask]
    tile_data = data[mask]

    if len(tile_data) < 3:
        img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        img.save(tile_path)
        return

    # Subsample if too many points to prevent OOM in triangulation
    MAX_POINTS = 500_000
    n_points_tile = len(tile_data)
    if n_points_tile > MAX_POINTS:
        step = n_points_tile // MAX_POINTS + 1
        tile_x_coords = tile_x_coords[::step]
        tile_y_coords = tile_y_coords[::step]
        tile_data = tile_data[::step]

    # Create figure
    fig, ax = plt.subplots(figsize=(TILE_SIZE / 100, TILE_SIZE / 100), dpi=100)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.axis('off')

    try:
        triang = tri.Triangulation(tile_x_coords, tile_y_coords)

        # Mask large triangles (gaps in data)
        triangles = triang.triangles
        tri_x = triang.x[triangles]
        tri_y = triang.y[triangles]

        # Vectorized area calculation
        areas = 0.5 * np.abs(
            (tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0]) -
            (tri_x[:, 2] - tri_x[:, 0]) * (tri_y[:, 1] - tri_y[:, 0])
        )
        median_area = np.median(areas)
        triang.set_mask(areas > 3 * median_area)
        del tri_x, tri_y, areas

        if not np.iterable(levels):
            levels = np.linspace(vmin, vmax, levels)
        ax.tricontourf(triang, tile_data, levels=levels,
                       cmap=cmap, vmin=vmin, vmax=vmax, extend='both')

        if add_contour_lines and (zoom > 4):
            cs = ax.tricontour(triang, tile_data, levels=contour_levels,
                               colors="0.7", linewidths=0.5,
                               linestyles="solid", alpha=0.5)
            filter_small_contours(cs, 100)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    except Exception:
        plt.close(fig)
        img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        img.save(tile_path)
        return

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    buf = io.BytesIO()
    try:
        plt.savefig(buf, format='png', transparent=True, pad_inches=0)
    finally:
        plt.close(fig)

    buf.seek(0)
    img = Image.open(buf)
    img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
    img.save(tile_path)
    buf.close()


def cruise_tiles(da, tile_base, cmap="viridis", levels=None, vmin=None, vmax=None, verbose=True):
    """Generate slippy map tiles from an xarray DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with latitude and longitude coordinates.
    tile_base : str
        Base output directory for generated tiles.
    cmap : str, optional
        Matplotlib colormap name, by default "viridis".
    levels : int or list, optional
        Number of contour levels or list of level values. If None, auto-calculated.
    vmin : float, optional
        Minimum value for colormap. If None, auto-calculated from data.
    vmax : float, optional
        Maximum value for colormap. If None, auto-calculated from data.
    verbose : bool, optional
        Whether to print progress messages, by default True.
    """
    settings.set("verbose", verbose)
    generator = SlippyTileGenerator(
        min_lat=float(da.latitude.min()),
        max_lat=float(da.latitude.max()),
        min_lon=float(da.longitude.min()),
        max_lon=float(da.longitude.max())
    )
    generator.generate_tiles(
        np.squeeze(da.data),
        da.latitude.data,
        da.longitude.data,
        tile_base,
        settings.get("zoom_levels", [3, 4, 5, 6, 7]),
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax
    )
