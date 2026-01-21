"""Satpy-based tile generation utilities.

This module provides functions for generating slippy map tiles from
Satpy scenes, including support for SSH and chlorophyll visualization.
"""
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm


from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from PIL import Image
from pyresample.geometry import AreaDefinition
from satpy import MultiScene, Scene

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


from .area_definitions import webmercator as webmercator_area
from . import config
settings = config.settings()


def enhance_chl(da, vmin=0.01, vmax=50):
    """Enhance chlorophyll data for visualization.

    Applies log-normalization and colormap to chlorophyll data.

    Parameters
    ----------
    da : xarray.DataArray
        Chlorophyll data array.
    vmin : float, optional
        Minimum value for log normalization, by default 0.01.
    vmax : float, optional
        Maximum value for log normalization, by default 50.

    Returns
    -------
    numpy.ndarray
        RGBA image array with uint8 values.
    """
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    data = da.data
    data = np.clip(data, vmin, vmax)
    rgba = cmap(norm(data))

    # Convert to uint8 RGBA
    rgba = (rgba * 255).astype(np.uint8)
    return rgba


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


def satpy_chl_to_tiles(
    scenes,
    dtm=None,
    min_zoom=0,
    max_zoom=6,
):
    """Generate chlorophyll tiles from Satpy scenes.

    Parameters
    ----------
    scenes : satpy.Scene or list of satpy.Scene
        Input scene(s) with chlorophyll data.
    dtm : str or datetime-like, optional
        Date for tile output directory naming.
    min_zoom : int, optional
        Minimum zoom level, by default 0.
    max_zoom : int, optional
        Maximum zoom level, by default 6.
    """
    dtm = "" if dtm is None else str(pd.to_datetime(dtm).date())
    tile_base = pathlib.Path(settings["tile_dir"]) / "chl" / dtm
    tile_base.mkdir(parents=True, exist_ok=True)

    for z in range(min_zoom, max_zoom + 1):
        print(f"Zoom level: {z}")
        da = reproject(scenes, z)["chl_nn"]

        #area = webmercator_area(45, 55, -10, 10, zoom=z)
        #wscenes = [scn.resample(area, resampler="bilinear") for scn in scenes]
        #mscn = MultiScene(wscenes).blend()
        #da = mscn["chl_nn"]
        rgba = enhance_chl(da)

        for tile in mercantile.tiles(-180, -85, 180, 85, z):
            print("tile")
            x0 = tile.x * 256
            y0 = tile.y * 256
            tile_img = rgba[y0 : y0 + 256, x0 : x0 + 256]
            if tile_img.shape[0] != 256 or tile_img.shape[1] != 256:
                continue

            path = tile_base / str(z) / str(tile.x)
            os.makedirs(path, exist_ok=True)
            Image.fromarray(tile_img).save(f"{path}/{tile.y}.png")



def satpy_ssh_to_tiles(
    scene,
    dtm=None,
    min_zoom=0,
    max_zoom=8,
    contour_levels=None,
    cmap='viridis',
    add_contour_lines=True,
):
    """Generate SSH contour tiles from a Satpy scene.

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
        Contour levels to use. If None, auto-calculated.
    cmap : str, optional
        Matplotlib colormap name, by default 'viridis'.
    add_contour_lines : bool, optional
        Whether to add contour lines, by default True.
    """
    dtm = "" if dtm is None else str(pd.to_datetime(dtm).date())
    tile_base = pathlib.Path(settings["tile_dir"]) / "ssh" / dtm
    tile_base.mkdir(parents=True, exist_ok=True)

    if contour_levels is None:
        cmin = np.nanmin(scene["sla"].values)
        cmax = np.nanmax(scene["sla"].values)
        cmax = np.max((np.abs(cmin), np.abs(cmax)))
        contour_levels = np.linspace(-cmax, cmax, 15)

    for z in range(min_zoom, max_zoom + 1):
        da = reproject(scene, z)["sla"]
        data = da.values

        # Get extent in pixel coordinates
        height, width = data.shape
        print(data.shape)

        for tile in mercantile.tiles(-180, -85, 180, 85, z):
            x0 = tile.x * 256
            y0 = tile.y * 256
            x1 = x0 + 256
            y1 = y0 + 256

            # Skip tiles outside the data bounds
            if x0 >= width or y0 >= height:
                continue

            # Extract tile data
            tile_data = data[y0:y1, x0:x1]

            # Skip if tile is empty or wrong size
            if tile_data.size == 0:
                continue

            # Create coordinate grids for the tile
            tile_height, tile_width = tile_data.shape
            x = np.arange(tile_width)
            y = np.arange(tile_height)
            X, Y = np.meshgrid(x, y)

            # Create figure with exact tile dimensions
            dpi = 100
            fig = Figure(figsize=(256/dpi, 256/dpi), dpi=dpi)
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
            ax.set_xlim(0, 255)
            ax.set_ylim(255, 0)  # Inverted y-axis for image coordinates
            ax.axis('off')

            # Create filled contours
            if tile_height == 256 and tile_width == 256:
                cf = ax.contourf(X, Y, tile_data, levels=contour_levels,
                                cmap=cmap, extend='both')

                # Add contour lines if requested
                if add_contour_lines:
                    ax.contour(X, Y, tile_data, levels=contour_levels,
                              colors='k', linewidths=0.5, alpha=0.3)
            else:
                # Handle edge tiles - pad or skip
                # For simplicity, you might want to skip incomplete tiles
                continue

            # Save tile
            path = tile_base / str(z) / str(tile.x)
            os.makedirs(path, exist_ok=True)

            # Render to PNG
            canvas.print_png(f"{path}/{tile.y}.png")
            plt.close(fig)











#####################
#
#
#
#
#
#
#
#
# ###################


def _render_tile(
    tile,
    data,
    norm,
    cmap,
    out_dir,
):
    """Render a single map tile using matplotlib.

    Parameters
    ----------
    tile : mercantile.Tile
        Tile to render.
    data : numpy.ndarray
        2D data array for the entire zoom level.
    norm : matplotlib.colors.Normalize
        Color normalization object.
    cmap : str or matplotlib.colors.Colormap
        Colormap to use.
    out_dir : pathlib.Path
        Output directory.

    Returns
    -------
    bool or None
        True if tile was rendered, None otherwise.
    """
    x0 = tile.x * 256
    y0 = tile.y * 256
    x1 = x0 + 256
    y1 = y0 + 256

    h, w = data.shape
    if x1 > w or y1 > h:
        return

    tile_data = data[y0:y1, x0:x1]
    if tile_data.shape != (256, 256):
        return

    fig, ax = plt.subplots(
        figsize=(256 / 96, 256 / 96),
        dpi=96,
        frameon=False,
    )
    ax.imshow(tile_data, cmap=cmap, norm=norm, origin="upper")
    ax.axis("off")

    path = out_dir / str(tile.z) / str(tile.x)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / f"{tile.y}.png", dpi=96, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True

def _render_tile_pil(tile, data, norm, cmap, out_dir, fill_empty=(0,0,0,0),
                     contour_levels=None, add_contours=True):
    """Render a 256x256 tile with optional contours using PIL.

    Parameters
    ----------
    tile : mercantile.Tile
        Tile to render.
    data : numpy.ndarray
        2D data array for the entire zoom level.
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
    x0 = tile.x * 256
    y0 = tile.y * 256
    x1 = x0 + 256
    y1 = y0 + 256

    h, w = data.shape
    path = out_dir / str(tile.z) / str(tile.x)
    path.mkdir(parents=True, exist_ok=True)
    outfile = path / f"{tile.y}.png"

    if outfile.exists():
        return True

    # Prepare empty tile array
    tile_data = np.full((256, 256), np.nan, dtype=data.dtype)

    # Compute intersection with data
    x_start = max(0, x0)
    y_start = max(0, y0)
    x_end = min(x1, w)
    y_end = min(y1, h)

    if x_end > x_start and y_end > y_start:
        # Flip vertically to match Mercator bottom-left origin
        tile_data[:y_end - y_start, :x_end - x_start] = np.flipud(
            data[y_start:y_end, x_start:x_end]
        )

    # Mask NaNs
    mask = np.isnan(tile_data)
    tile_data[mask] = 0.0

    # Base colormap
    normed = norm(tile_data)
    rgba = (cmap(normed) * 255).astype(np.uint8)
    rgba[mask] = np.array(fill_empty, dtype=np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")

    # Optional contours
    if add_contours and contour_levels is not None:
        fig = plt.Figure(figsize=(256/100, 256/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        X, Y = np.meshgrid(np.arange(256), np.arange(256))
        ax.contour(X, Y, tile_data, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.6)
        canvas.draw()
        contour_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        contour_img = contour_img.reshape((256, 256, 4))
        # Convert ARGB → RGBA
        contour_img = contour_img[:, :, [1,2,3,0]]
        contour_pil = Image.fromarray(contour_img, mode='RGBA')
        img = Image.alpha_composite(img, contour_pil)
        plt.close(fig)

    img.save(outfile)
    return True




def _process_zoom(
    scene,
    z,
    tile_base,
    cmap,
    contour_levels,
):
    """Process all tiles for a single zoom level.

    Parameters
    ----------
    scene : satpy.Scene
        Input scene with data.
    z : int
        Zoom level.
    tile_base : pathlib.Path
        Base output directory.
    cmap : str or matplotlib.colors.Colormap
        Colormap to use.
    contour_levels : numpy.ndarray or None
        Contour levels (None for auto).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="pyproj.crs"
        )
        da = reproject(scene, z)["sla"]

    # Compute stats lazily
    if contour_levels is None:
        vmin = float(da.min())
        vmax = float(da.max())
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        vmax = vmax
    else:
        vmin, vmax = contour_levels[0], contour_levels[-1]

    norm = Normalize(vmin=vmin, vmax=vmax)

    # Materialize ONCE per zoom (still big, but unavoidable)
    data = da.data.compute() if hasattr(da.data, "compute") else da.values

    tiles = list(mercantile.tiles(-180, -85, 180, 85, z))

    out_dir = tile_base

    for tile in tiles:
        _render_tile(tile, data, norm, cmap, out_dir)


def satpy_ssh_to_tiles_2(
    scene,
    dtm=None,
    min_zoom=0,
    max_zoom=8,
    contour_levels=None,
    cmap="viridis",
    workers=None,
):
    """Generate SSH tiles with threaded rendering (version 2).

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
    cmap : str, optional
        Matplotlib colormap name, by default "viridis".
    workers : int, optional
        Number of worker threads. If None, uses CPU count - 1.
    """
    dtm = "" if dtm is None else str(pd.to_datetime(dtm).date())
    tile_base = pathlib.Path(settings["tile_dir"]) / "ssh" / dtm
    tile_base.mkdir(parents=True, exist_ok=True)

    workers = workers or max(os.cpu_count() - 1, 1)
    tiles_by_zoom = {}
    total_tiles = 0
    for z in range(min_zoom, max_zoom + 1):
        tiles = list(mercantile.tiles(-180, -85, 180, 85, z))
        tiles_by_zoom[z] = tiles
        total_tiles += len(tiles)

    with tqdm(total=total_tiles, desc="Rendering tiles", unit="tile") as pbar:

        for z in range(min_zoom, max_zoom + 1):

            # --- Resample (main thread, Satpy-safe) ---
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module="pyresample.bilinear"
                )
                da = reproject(scene, z)["sla"]

            # --- Color scaling ---
            if contour_levels is None:
                vmin = float(da.min())
                vmax = float(da.max())
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
            else:
                vmin, vmax = contour_levels[0], contour_levels[-1]

            norm = Normalize(vmin=vmin, vmax=vmax)

            # --- Materialize once per zoom ---
            data = da.data.compute() if hasattr(da.data, "compute") else da.values

            tiles = tiles_by_zoom[z]

            # --- Threaded tile rendering ---
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = [
                    exe.submit(
                        _render_tile,
                        tile,
                        data,
                        norm,
                        cmap,
                        tile_base,
                    )
                    for tile in tiles
                ]

                for f in as_completed(futures):
                    if f.result():
                        pbar.update(1)
                    else:
                        pbar.update(1)


def satpy_ssh_to_tiles_3(
    scene,
    dtm=None,
    min_zoom=0,
    max_zoom=8,
    contour_levels=None,
    cmap=None,
    workers=None,
    fill_empty=(0,0,0,0)
):
    """Generate SSH tiles with PIL rendering (version 3).

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
    """
    import matplotlib.cm as cm

    dtm = "" if dtm is None else str(pd.to_datetime(dtm).date())
    tile_base = pathlib.Path(settings["tile_dir"]) / "ssh" / dtm
    tile_base.mkdir(parents=True, exist_ok=True)

    workers = workers or max(os.cpu_count() - 1, 1)
    cmap = cmap or cm.get_cmap("viridis")

    # Precompute tiles for progress bar
    tiles_by_zoom = {}
    total_tiles = 0
    for z in range(min_zoom, max_zoom + 1):
        tiles = list(mercantile.tiles(-180, -85, 180, 85, z))
        tiles_by_zoom[z] = tiles
        total_tiles += len(tiles)

    with tqdm(total=total_tiles, desc="Rendering tiles", unit="tile") as pbar:
        for z in range(min_zoom, max_zoom + 1):
            # --- Safe Satpy resample ---
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module="pyresample.bilinear"
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module="pyproj.crs"
                )
                da = reproject(scene, z)["sla"]

            # --- Color normalization ---
            if contour_levels is None:
                vmin = float(da.min())
                vmax = float(da.max())
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
            else:
                vmin, vmax = contour_levels[0], contour_levels[-1]

            norm = Normalize(vmin=vmin, vmax=vmax)
            data = da.data.compute() if hasattr(da.data, "compute") else da.values

            # --- Threaded tile rendering ---
            tiles = tiles_by_zoom[z]
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = [
                    exe.submit(
                        _render_tile_pil,
                        tile,
                        data,
                        norm,
                        cmap,
                        tile_base,
                        fill_empty
                    )
                    for tile in tiles
                ]
                for f in as_completed(futures):
                    pbar.update(1)


def satpy_ssh_to_tiles_4(
    scene,
    dtm=None,
    min_zoom=0,
    max_zoom=8,
    contour_levels=None,
    cmap=None,
    workers=None,
    fill_empty=(0,0,0,0),
    add_contours=True
):
    """Generate SSH tiles with contours (version 4).

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
    """
    dtm = "" if dtm is None else str(pd.to_datetime(dtm).date())
    tile_base = pathlib.Path(settings["tile_dir"]) / "ssh" / dtm
    tile_base.mkdir(parents=True, exist_ok=True)

    workers = workers or max(os.cpu_count() - 1, 1)
    cmap = cmap or cm.get_cmap("viridis")

    # Precompute tiles for progress bar
    tiles_by_zoom = {}
    total_tiles = 0
    for z in range(min_zoom, max_zoom + 1):
        tiles = list(mercantile.tiles(-180, -85, 180, 85, z))
        tiles_by_zoom[z] = tiles
        total_tiles += len(tiles)

    with tqdm(total=total_tiles, desc="Rendering tiles", unit="tile") as pbar:
        for z in range(min_zoom, max_zoom + 1):
            # --- Safe Satpy resample ---
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module="pyresample.bilinear"
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module="pyproj.crs"
                )
                da = reproject(scene, z)["sla"]

            # --- Color normalization ---
            if contour_levels is None:
                vmin = float(da.min())
                vmax = float(da.max())
                vmax = max(abs(vmin), abs(vmax))
                vmin = -vmax
            else:
                vmin, vmax = contour_levels[0], contour_levels[-1]

            norm = Normalize(vmin=vmin, vmax=vmax)
            data = da.data.compute() if hasattr(da.data, "compute") else da.values

            # --- Threaded tile rendering ---
            tiles = tiles_by_zoom[z]
            with ThreadPoolExecutor(max_workers=workers) as exe:
                futures = [
                    exe.submit(
                        _render_tile_pil,
                        tile,
                        data,
                        norm,
                        cmap,
                        tile_base,
                        fill_empty,
                        contour_levels,
                        add_contours
                    )
                    for tile in tiles
                ]
                for f in as_completed(futures):
                    pbar.update(1)
