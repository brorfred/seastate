"""Fast SSH tile generation utilities.

This module provides optimized functions for generating SSH slippy map
tiles with per-tile resampling for accurate alignment.
"""
import os
import pathlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw
import pandas as pd
import warnings
import mercantile
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from satpy import MultiScene, Scene

from .area_definitions import webmercator as webmercator_area
from . import config
settings = config.settings()


def reproject(scene_obj, zoom):
    """Reproject a Satpy scene to Web Mercator at given zoom level.

    Parameters
    ----------
    scene_obj : satpy.Scene or list of satpy.Scene
        Single scene or list of scenes to reproject.
    zoom : int
        Web Mercator zoom level.

    Returns
    -------
    satpy.Scene
        Reprojected scene (blended if multiple input scenes).
    """
    area = webmercator_area(settings["lat1"],
                            settings["lat2"],
                            settings["lon1"],
                            settings["lon2"],
                            zoom=zoom)
    with np.errstate(invalid="ignore", divide="ignore"):
        if isinstance(scene_obj, Scene):
            scn = scene_obj.resample(area, resampler="bilinear")
        else:
            wscenes = [scn.resample(area, resampler="bilinear") for scn in scene_obj]
            scn = MultiScene(wscenes).blend()
    return scn


def tile_overlaps_scene(tile, scene_bounds):
    """Check if a tile overlaps the scene bounds.

    Parameters
    ----------
    tile : mercantile.Tile
        Tile to check.
    scene_bounds : tuple
        Scene bounds as (lon_min, lat_min, lon_max, lat_max).

    Returns
    -------
    bool
        True if tile overlaps scene bounds.
    """
    w, s, e, n = mercantile.bounds(tile)
    lon_min, lat_min, lon_max, lat_max = scene_bounds
    return not ((e <= lon_min) or (w >= lon_max) or (n <= lat_min) or (s >= lat_max))


def _render_tile(tile, tile_data, norm, cmap, out_dir, fill_empty=(0,0,0,0),
                 contour_levels=None, add_contours=True):
    """Render a 256x256 tile using PIL with optional contours.

    Parameters
    ----------
    tile : mercantile.Tile
        Tile to render.
    tile_data : numpy.ndarray
        Data array for this tile (from per-tile resample).
    norm : matplotlib.colors.Normalize
        Color normalization object.
    cmap : matplotlib.colors.Colormap
        Colormap to use.
    out_dir : pathlib.Path
        Output directory.
    fill_empty : tuple of int, optional
        RGBA color for empty/NaN pixels, by default (0,0,0,0).
    contour_levels : numpy.ndarray, optional
        Contour levels to draw.
    add_contours : bool, optional
        Whether to add contour lines, by default True.

    Returns
    -------
    bool
        True if tile was rendered.
    """
    path = out_dir / str(tile.z) / str(tile.x)
    path.mkdir(parents=True, exist_ok=True)
    outfile = path / f"{tile.y}.png"

    # Ensure exactly 256x256
    h, w_ = tile_data.shape
    final_tile = np.full((256,256), np.nan, dtype=tile_data.dtype)
    final_tile[:min(h,256), :min(w_,256)] = tile_data[:256, :256]

    # Mask NaNs
    mask = np.isnan(final_tile)
    final_tile[mask] = 0.0

    # Normalize and colormap
    normed = norm(final_tile)
    rgba = (cmap(normed) * 255).astype(np.uint8)
    rgba[mask] = np.array(fill_empty, dtype=np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")

    # Optional contours (slow for high-res; can be vectorized later)
    if add_contours and contour_levels is not None:
        draw = ImageDraw.Draw(img)
        for level in contour_levels:
            for y in range(255):
                for x in range(255):
                    neighbors = final_tile[y:y+2, x:x+2]
                    if np.any(neighbors >= level) and np.any(neighbors < level):
                        draw.point((x, y), fill=(0,0,0,160))

    img.save(outfile)
    return True

def satpy_ssh_to_tiles_fixed(
    scene,
    dtm=None,
    min_zoom=0,
    max_zoom=8,
    contour_levels=None,
    cmap=None,
    workers=None,
    fill_empty=(0,0,0,0),
    add_contours=True,
    log_qc_path=None
):
    """Generate SSH tiles with per-tile resampling for accurate alignment.

    This version resamples each tile individually to ensure proper
    alignment with slippy map coordinates.

    Parameters
    ----------
    scene : satpy.Scene
        Input scene with SSH/SLA data.
    dtm : str or datetime-like, optional
        Date for tile output directory naming.
    min_zoom : int, optional
        Minimum zoom level, by default 0.
    max_zoom : int, optional
        Maximum zoom level, by default 8.
    contour_levels : numpy.ndarray, optional
        Contour levels (None for auto).
    cmap : matplotlib.colors.Colormap, optional
        Colormap. If None, uses 'viridis'.
    workers : int, optional
        Number of worker threads. If None, uses CPU count - 1.
    fill_empty : tuple of int, optional
        RGBA color for empty tiles, by default (0,0,0,0).
    add_contours : bool, optional
        Whether to add contour lines, by default True.
    log_qc_path : str or pathlib.Path, optional
        Path for QC log output. If None, no log is written.
    """
    dtm = "" if dtm is None else str(pd.to_datetime(dtm).date())
    tile_base = pathlib.Path(settings["tile_dir"]) / "ssh" / dtm
    tile_base.mkdir(parents=True, exist_ok=True)
    workers = workers or max(os.cpu_count() - 1, 1)
    cmap = cmap or cm.get_cmap("viridis")

    # Scene bounds for QA
    area = scene['sla'].area
    lon_min, lat_min, lon_max, lat_max = area.area_extent
    scene_bounds = (lon_min, lat_min, lon_max, lat_max)

    qc_log = []

    # Precompute tiles
    total_tiles = 0
    tiles_by_zoom = {}
    for z in range(min_zoom, max_zoom + 1):
        tiles = list(mercantile.tiles(-180, -85, 180, 85, z))
        tiles_by_zoom[z] = tiles
        total_tiles += len(tiles)

    with tqdm(total=total_tiles, desc="Rendering tiles", unit="tile") as pbar:
        for z in range(min_zoom, max_zoom + 1):
            tiles = tiles_by_zoom[z]

            def process_tile(tile):
                #try:
                w, s, e, n = mercantile.bounds(tile)
                # Area for this single tile
                tile_area = webmercator_area(s, n, w, e, zoom=z)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyresample.bilinear")
                    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj.crs")
                    scn_tile = scene.resample(tile_area, resampler="bilinear")
                    scn_tile["sla"] = scn_tile["sla"].load()  # force numpy
                    data = scn_tile["sla"].values
                # Skip tiles with no data
                if np.isnan(data).all():
                    tile_data = np.full((256,256), np.nan)
                    _render_tile(tile, tile_data, Normalize(-1,1), cmap, tile_base,
                                    fill_empty, contour_levels, add_contours)
                    qc_log.append((tile.z, tile.x, tile.y, "empty"))
                    return
                # Color normalization
                if contour_levels is None:
                    vmin = float(np.nanmin(data))
                    vmax = float(np.nanmax(data))
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
                else:
                    vmin, vmax = contour_levels[0], contour_levels[-1]
                norm = Normalize(vmin=vmin, vmax=vmax)
                _render_tile(tile, data, norm, cmap, tile_base,
                                fill_empty, contour_levels, add_contours)
                if not tile_overlaps_scene(tile, scene_bounds):
                    qc_log.append((tile.z, tile.x, tile.y, "out_of_bounds"))
                #except Exception as e:
                #    qc_log.append((tile.z, tile.x, tile.y, f"error:{e}"))
                pbar.update(1)

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as exe:
                exe.map(process_tile, tiles)

    # Save QA log
    if log_qc_path:
        import csv
        log_qc_path = pathlib.Path(log_qc_path)
        log_qc_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_qc_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zoom","x","y","status"])
            writer.writerows(qc_log)
