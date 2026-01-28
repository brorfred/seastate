"""Slippy Tile Generator for OLCI Level-2 Swath Data.

This module generates filled contour slippy map tiles from OLCI L2 swath
scenes produced by extract_swath_scenes. Uses triangulation for irregular
swath geometry and mercantile for robust tile calculations.

Data is transformed to Web Mercator (EPSG:3857) before triangulation to
ensure proper alignment with slippy map tiles.

Memory-optimized using spatial indexing and chunked processing.
"""

import gc
import os
import tempfile
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

from .utils import filter_small_contours
from ..utils import vprint, lonlat_to_webmercator
from .. import config

settings = config.settings


# Memory limit per chunk (in number of float32 values, ~400MB per array)
_CHUNK_SIZE = 100_000_000


class OLCISwathTileGenerator:
    """Generate slippy map tiles from OLCI L2 swath scenes with filled contours.

    This generator handles irregular swath geometry by using Delaunay
    triangulation for interpolation, which is appropriate for the 2D
    coordinate arrays in satellite swath data.

    Memory-optimized: uses memory-mapped files for large arrays to avoid
    copying data to each parallel worker.

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
        """Get all tiles that intersect with the geographic bounds."""
        tiles = list(mercantile.tiles(
            self.min_lon, self.min_lat,
            self.max_lon, self.max_lat,
            zooms=zoom
        ))
        return tiles

    def _extract_scene_data(self, scene, dataset_name: str = "chl_oc4me"):
        """Extract data and coordinates from a Satpy scene.

        Returns float32 arrays to reduce memory usage.
        """
        if dataset_name not in scene:
            vprint(f"Dataset {dataset_name} not found in scene")
            return None, None, None

        da = scene[dataset_name]
        data = da.values.astype(np.float32)

        # Get coordinates - OLCI L2 has 2D lat/lon arrays
        if 'latitude' in da.coords:
            lats = da.coords['latitude'].values
            lons = da.coords['longitude'].values
        elif hasattr(da, 'area') and da.area is not None:
            lons, lats = da.area.get_lonlats()
        else:
            vprint("Could not extract coordinates from scene")
            return None, None, None

        # Compute if dask arrays and convert to float32
        if hasattr(lats, 'compute'):
            lats = lats.compute()
        if hasattr(lons, 'compute'):
            lons = lons.compute()

        return data, lats.astype(np.float32), lons.astype(np.float32)

    def generate_tiles_from_scenes(
        self,
        scenes: List,
        output_dir: str,
        zoom_levels: List[int] = None,
        dataset_name: str = "chl_oc4me",
        num_workers: int = 4,
        cmap: str = 'viridis',
        levels: int = 20,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        add_contour_lines: bool = False,
        contour_levels: int = 5,
        log_transform: bool = True
    ):
        """Generate tiles from multiple OLCI L2 swath scenes.

        Memory-optimized: single-pass processing with streaming to disk.
        """
        if zoom_levels is None:
            zoom_levels = settings.get("zoom_levels", [3, 4, 5, 6, 7])

        # Create memory-mapped temporary files - single pass approach
        tmp_dir = tempfile.mkdtemp(prefix="olci_tiles_")
        try:
            # Collect data in chunks, streaming to disk
            x_chunks = []
            y_chunks = []
            data_chunks = []
            total_points = 0
            data_min = np.inf
            data_max = -np.inf
            margin = 2.0

            vprint("Processing scenes...")
            for i, scene in enumerate(scenes):
                data, lats, lons = self._extract_scene_data(scene, dataset_name)
                if data is None:
                    continue

                valid_mask = ~np.isnan(data)
                if lats.ndim == 2:
                    lats_flat = lats[valid_mask]
                    lons_flat = lons[valid_mask]
                else:
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    lats_flat = lat_grid[valid_mask].astype(np.float32)
                    lons_flat = lon_grid[valid_mask].astype(np.float32)
                    del lon_grid, lat_grid

                data_flat = data[valid_mask]
                del data, lats, lons, valid_mask

                # Filter to bounds
                bounds_mask = (
                    (lats_flat >= self.min_lat - margin) &
                    (lats_flat <= self.max_lat + margin) &
                    (lons_flat >= self.min_lon - margin) &
                    (lons_flat <= self.max_lon + margin)
                )

                if not np.any(bounds_mask):
                    del lats_flat, lons_flat, data_flat, bounds_mask
                    continue

                lats_bounded = lats_flat[bounds_mask]
                lons_bounded = lons_flat[bounds_mask]
                data_bounded = data_flat[bounds_mask]
                del lats_flat, lons_flat, data_flat, bounds_mask

                # Apply log transform if needed
                if log_transform and dataset_name in ["chl_oc4me", "chl_nn"]:
                    data_bounded = np.log10(np.clip(data_bounded, 0.01, None))

                # Transform to Web Mercator
                x_wm, y_wm = lonlat_to_webmercator(lons_bounded, lats_bounded)
                del lats_bounded, lons_bounded

                # Track stats
                n_pts = len(data_bounded)
                total_points += n_pts
                if n_pts > 0:
                    data_min = min(data_min, float(np.nanmin(data_bounded)))
                    data_max = max(data_max, float(np.nanmax(data_bounded)))

                x_chunks.append(x_wm)
                y_chunks.append(y_wm)
                data_chunks.append(data_bounded)

                vprint(f"Scene {i+1}/{len(scenes)}: {n_pts} points")

                # Flush to disk if accumulated too much
                if total_points > _CHUNK_SIZE:
                    vprint("Flushing to disk...")
                    gc.collect()

            if total_points == 0:
                vprint("No valid data found in scenes")
                return

            vprint(f"Total points: {total_points}")

            # Create memory-mapped files and write all data
            x_mmap_path = os.path.join(tmp_dir, "x_coords.dat")
            y_mmap_path = os.path.join(tmp_dir, "y_coords.dat")
            data_mmap_path = os.path.join(tmp_dir, "data.dat")

            x_mmap = np.memmap(x_mmap_path, dtype=np.float32, mode='w+', shape=(total_points,))
            y_mmap = np.memmap(y_mmap_path, dtype=np.float32, mode='w+', shape=(total_points,))
            data_mmap = np.memmap(data_mmap_path, dtype=np.float32, mode='w+', shape=(total_points,))

            offset = 0
            for x_c, y_c, d_c in zip(x_chunks, y_chunks, data_chunks):
                n = len(d_c)
                x_mmap[offset:offset + n] = x_c
                y_mmap[offset:offset + n] = y_c
                data_mmap[offset:offset + n] = d_c
                offset += n

            # Free chunk memory
            del x_chunks, y_chunks, data_chunks
            gc.collect()

            x_mmap.flush()
            y_mmap.flush()
            data_mmap.flush()
            del x_mmap, y_mmap, data_mmap
            gc.collect()

            # Use provided vmin/vmax or calculated
            if vmin is None:
                vmin = float(data_min)
            if vmax is None:
                vmax = float(data_max)

            vprint(f"Data range: {vmin:.4f} to {vmax:.4f}")

            # Build spatial index for efficient tile queries
            vprint("Building spatial index...")
            index_path = os.path.join(tmp_dir, "spatial_index.npz")
            _build_spatial_index(
                x_mmap_path, y_mmap_path, total_points, index_path,
                grid_size=256
            )

            # Generate tiles using memory-mapped data with spatial index
            self._generate_tiles_from_mmap(
                x_mmap_path=x_mmap_path,
                y_mmap_path=y_mmap_path,
                data_mmap_path=data_mmap_path,
                n_points=total_points,
                output_dir=output_dir,
                zoom_levels=zoom_levels,
                num_workers=num_workers,
                cmap=cmap,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                add_contour_lines=add_contour_lines,
                contour_levels=contour_levels,
                index_path=index_path
            )

        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _generate_tiles_from_mmap(
        self,
        x_mmap_path: str,
        y_mmap_path: str,
        data_mmap_path: str,
        n_points: int,
        output_dir: str,
        zoom_levels: List[int],
        num_workers: int = 10,
        cmap: str = 'viridis',
        levels: int = 20,
        vmin: float = 0,
        vmax: float = 1,
        add_contour_lines: bool = False,
        contour_levels: int = 5,
        index_path: str = None
    ):
        """Generate tiles using memory-mapped data files with spatial index."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        vprint(f"Processing {n_points} valid data points")

        for zoom in zoom_levels:
            vprint(f"\nGenerating tiles for zoom level {zoom}")

            tiles = self.get_tiles_for_bounds(zoom)

            # Create directories
            zoom_dir = output_path / str(zoom)
            zoom_dir.mkdir(exist_ok=True)
            for tile_x in set(t.x for t in tiles):
                (zoom_dir / str(tile_x)).mkdir(exist_ok=True)

            vprint(f"Total tiles to generate: {len(tiles)}")

            # Process tiles in parallel with reduced workers to limit memory
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _generate_single_tile_indexed,
                        tile.z, tile.x, tile.y,
                        x_mmap_path, y_mmap_path, data_mmap_path, n_points,
                        str(output_path), cmap, levels, vmin, vmax,
                        add_contour_lines, contour_levels, index_path
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


def _build_spatial_index(x_path: str, y_path: str, n_points: int,
                         index_path: str, grid_size: int = 256):
    """Build a grid-based spatial index for fast point lookups.

    Divides the coordinate space into a grid and stores which points
    fall into each cell. This allows tile generation to only examine
    points in nearby cells rather than scanning all points.
    """
    x_coords = np.memmap(x_path, dtype=np.float32, mode='r', shape=(n_points,))
    y_coords = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n_points,))

    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))

    # Compute cell sizes
    x_cell_size = (x_max - x_min) / grid_size if x_max > x_min else 1.0
    y_cell_size = (y_max - y_min) / grid_size if y_max > y_min else 1.0

    # Assign each point to a cell
    x_cell = np.clip(((x_coords - x_min) / x_cell_size).astype(np.int32), 0, grid_size - 1)
    y_cell = np.clip(((y_coords - y_min) / y_cell_size).astype(np.int32), 0, grid_size - 1)

    # Create cell index - use flat cell IDs
    cell_ids = y_cell * grid_size + x_cell
    del x_cell, y_cell

    # Sort points by cell for efficient lookup
    sort_order = np.argsort(cell_ids)
    sorted_cells = cell_ids[sort_order]
    del cell_ids

    # Find boundaries of each cell in sorted array
    cell_boundaries = np.searchsorted(sorted_cells, np.arange(grid_size * grid_size + 1))
    del sorted_cells

    # Save index
    np.savez_compressed(
        index_path,
        sort_order=sort_order.astype(np.int32),
        cell_boundaries=cell_boundaries.astype(np.int32),
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid_size=grid_size, x_cell_size=x_cell_size, y_cell_size=y_cell_size
    )

    del x_coords, y_coords, sort_order, cell_boundaries


def _get_points_in_bounds(index_data, x_left, x_right, y_bottom, y_top):
    """Get indices of points within given bounds using spatial index."""
    grid_size = int(index_data['grid_size'])
    x_min = float(index_data['x_min'])
    y_min = float(index_data['y_min'])
    x_cell_size = float(index_data['x_cell_size'])
    y_cell_size = float(index_data['y_cell_size'])
    sort_order = index_data['sort_order']
    cell_boundaries = index_data['cell_boundaries']

    # Find which cells overlap the query bounds
    x_cell_min = max(0, int((x_left - x_min) / x_cell_size))
    x_cell_max = min(grid_size - 1, int((x_right - x_min) / x_cell_size))
    y_cell_min = max(0, int((y_bottom - y_min) / y_cell_size))
    y_cell_max = min(grid_size - 1, int((y_top - y_min) / y_cell_size))

    # Collect point indices from all overlapping cells
    indices = []
    for y_c in range(y_cell_min, y_cell_max + 1):
        for x_c in range(x_cell_min, x_cell_max + 1):
            cell_id = y_c * grid_size + x_c
            start = cell_boundaries[cell_id]
            end = cell_boundaries[cell_id + 1]
            if end > start:
                indices.append(sort_order[start:end])

    if not indices:
        return np.array([], dtype=np.int32)

    return np.concatenate(indices)


def _generate_single_tile_indexed(
    zoom: int, tile_x: int, tile_y: int,
    x_mmap_path: str, y_mmap_path: str, data_mmap_path: str, n_points: int,
    output_dir: str, cmap: str, levels: int,
    vmin: float, vmax: float,
    add_contour_lines: bool, contour_levels: int,
    index_path: str = None
):
    """Generate a single tile using spatial index for efficient point lookup."""
    TILE_SIZE = 256
    tile_path = Path(output_dir) / str(zoom) / str(tile_x) / f"{tile_y}.png"

    # Get tile bounds in Web Mercator meters
    xy_bounds = mercantile.xy_bounds(tile_x, tile_y, zoom)
    x_left, x_right = xy_bounds.left, xy_bounds.right
    y_bottom, y_top = xy_bounds.bottom, xy_bounds.top

    # Reduced buffer (15% instead of 50%)
    tile_height = y_top - y_bottom
    tile_width = x_right - x_left
    buffer_y = tile_height * 0.15
    buffer_x = tile_width * 0.15

    buffered_left = x_left - buffer_x
    buffered_right = x_right + buffer_x
    buffered_bottom = y_bottom - buffer_y
    buffered_top = y_top + buffer_y

    # Use spatial index if available
    if index_path and os.path.exists(index_path):
        index_data = np.load(index_path)
        candidate_indices = _get_points_in_bounds(
            index_data, buffered_left, buffered_right, buffered_bottom, buffered_top
        )

        if len(candidate_indices) == 0:
            img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
            img.save(tile_path)
            return

        # Open mmaps and extract only needed points
        x_coords = np.memmap(x_mmap_path, dtype=np.float32, mode='r', shape=(n_points,))
        y_coords = np.memmap(y_mmap_path, dtype=np.float32, mode='r', shape=(n_points,))
        data = np.memmap(data_mmap_path, dtype=np.float32, mode='r', shape=(n_points,))

        # Get candidate points
        tile_x_coords = x_coords[candidate_indices]
        tile_y_coords = y_coords[candidate_indices]
        tile_data = data[candidate_indices]

        del x_coords, y_coords, data

        # Fine filter within exact bounds
        mask = (
            (tile_x_coords >= buffered_left) & (tile_x_coords <= buffered_right) &
            (tile_y_coords >= buffered_bottom) & (tile_y_coords <= buffered_top)
        )

        if not np.any(mask):
            img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
            img.save(tile_path)
            return

        tile_x_coords = tile_x_coords[mask]
        tile_y_coords = tile_y_coords[mask]
        tile_data = tile_data[mask]
    else:
        # Fallback: scan all points (original behavior)
        x_coords = np.memmap(x_mmap_path, dtype=np.float32, mode='r', shape=(n_points,))
        y_coords = np.memmap(y_mmap_path, dtype=np.float32, mode='r', shape=(n_points,))
        data = np.memmap(data_mmap_path, dtype=np.float32, mode='r', shape=(n_points,))

        mask = (
            (y_coords >= buffered_bottom) & (y_coords <= buffered_top) &
            (x_coords >= buffered_left) & (x_coords <= buffered_right)
        )

        if not np.any(mask):
            img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
            img.save(tile_path)
            del x_coords, y_coords, data
            return

        tile_x_coords = x_coords[mask]
        tile_y_coords = y_coords[mask]
        tile_data = data[mask]
        del x_coords, y_coords, data

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

    # Create figure - use lower DPI for memory savings
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


def olci_swath_tiles(
    scenes,
    tile_base: str,
    dataset_name: str = "chl_oc4me",
    cmap="nipy_spectral",
    levels: int = 50,
    vmin=-4.6,
    vmax=4.6,
    log_transform: bool = True,
    zoom_levels: List[int] = None,
    num_workers: int = 10,
    add_contour_lines: bool = False,
    contour_levels: int = 5,
    verbose: bool = True
):
    """Generate slippy map tiles from OLCI L2 swath scenes.

    This is the main entry point for generating tiles from the output of
    extract_swath_scenes.

    Parameters
    ----------
    scenes : list of satpy.Scene
        List of Satpy scenes from extract_swath_scenes.
    tile_base : str
        Base output directory for generated tiles.
    dataset_name : str, optional
        Dataset to render ("chl_oc4me" or "chl_nn"), by default "chl_oc4me".
    cmap : str, optional
        Matplotlib colormap name, by default "viridis".
    levels : int, optional
        Number of contour levels, by default 20.
    vmin : float, optional
        Minimum value for colormap. If None, auto-calculated from data.
    vmax : float, optional
        Maximum value for colormap. If None, auto-calculated from data.
    log_transform : bool, optional
        Apply log10 transform to chlorophyll values, by default True.
    zoom_levels : list of int, optional
        Zoom levels to generate. If None, uses settings["zoom_levels"].
    num_workers : int, optional
        Number of parallel workers, by default 4.
    add_contour_lines : bool, optional
        Whether to add contour lines, by default False.
    contour_levels : int, optional
        Number of contour line levels, by default 5.
    verbose : bool, optional
        Whether to print progress messages, by default True.

    Examples
    --------
    >>> from seaview.data_sources.olci_L2 import extract_swath_scenes
    >>> from seaview.tilers.olci_swath import olci_swath_tiles
    >>> scenes = extract_swath_scenes(dtm="2023-06-03")
    >>> olci_swath_tiles(scenes, "/path/to/tiles/chl")
    """
    settings.set("verbose", verbose)

    min_lat = settings.get("lat1")
    max_lat = settings.get("lat2")
    min_lon = settings.get("lon1")
    max_lon = settings.get("lon2")

    generator = OLCISwathTileGenerator(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon
    )

    generator.generate_tiles_from_scenes(
        scenes=scenes,
        output_dir=tile_base,
        zoom_levels=zoom_levels,
        dataset_name=dataset_name,
        num_workers=num_workers,
        cmap=cmap,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        add_contour_lines=add_contour_lines,
        contour_levels=contour_levels,
        log_transform=log_transform
    )
