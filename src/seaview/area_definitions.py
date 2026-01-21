"""Area definition utilities for satellite data processing.

This module provides functions to create pyresample AreaDefinition objects
for various coordinate systems and grid types commonly used in oceanographic
and satellite data processing.
"""
import math

import numpy as np
from pyresample.geometry import AreaDefinition
from pyproj import CRS, Transformer

WEBMERCATOR_RADIUS = 6378137.0
TILE_SIZE = 256


def zoom_to_resolution_m(zoom: int) -> float:
    """Convert Web Mercator zoom level to resolution in meters per pixel.

    Parameters
    ----------
    zoom : int
        Web Mercator zoom level (slippy-map convention).

    Returns
    -------
    float
        Resolution in meters per pixel at the equator.
    """
    return (2 * math.pi * WEBMERCATOR_RADIUS) / (TILE_SIZE * 2**zoom)


def webmercator(
    lat1: float,
    lat2: float,
    lon1: float,
    lon2: float,
    *,
    zoom: int | None = None,
    resolution_m: float | None = None,
    description: str | None = None,
    area_id="webmercator_region",
):
    """Create a Satpy Web Mercator area definition from a lat/lon bbox.

    Specify either `zoom` or `resolution_m`.

    Parameters
    ----------
    lat1 : float
        Minimum latitude (southern boundary) in degrees.
    lat2 : float
        Maximum latitude (northern boundary) in degrees.
    lon1 : float
        Minimum longitude (western boundary) in degrees.
    lon2 : float
        Maximum longitude (eastern boundary) in degrees.
    zoom : int, optional
        Web Mercator zoom level (slippy-map convention).
    resolution_m : float, optional
        Pixel size in meters (overrides zoom if provided).
    description : str, optional
        Human-readable description.
    area_id : str, optional
        Area identifier for Satpy, by default 'webmercator_region'.

    Returns
    -------
    pyresample.AreaDefinition
        Web Mercator area definition for the specified bounding box.
    """
    lon_min, lat_min, lon_max, lat_max = lon1, lat1, lon2, lat2

    if resolution_m is None:
        if zoom is None:
            raise ValueError("Either 'zoom' or 'resolution_m' must be provided")
        resolution_m = zoom_to_resolution_m(zoom)

    proj_dict = {
        "proj": "merc",
        "a": WEBMERCATOR_RADIUS,
        "b": WEBMERCATOR_RADIUS,
        "lat_ts": 0.0,
        "lon_0": 0.0,
        "x_0": 0.0,
        "y_0": 0.0,
        "units": "m",
        "no_defs": None,
    }

    #transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    crs_ll = CRS.from_epsg(4326)
    crs_wm = CRS.from_epsg(3857)  # or CRS.from_wkt(WEBMERCATOR_WKT)
    transformer = Transformer.from_crs(crs_ll, crs_wm, always_xy=True)



    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    width = math.ceil((x_max - x_min) / resolution_m)
    height = math.ceil((y_max - y_min) / resolution_m)
    area_extent = (x_min, y_min, x_max, y_max)

    return AreaDefinition(
        area_id=area_id,
        description=description or area_id,
        #proj_id="epsg3857",
        #proj_id="EPSG:3857",
        proj_id="webmercator_wkt",
        projection=crs_wm.to_wkt(),
        #projection=proj_dict,
        width=width,
        height=height,
        area_extent=area_extent,
    )


def nasa(resolution="4km", lat1=-90, lat2=90, lon1=-180, lon2=180, **kw):
    """Create an AreaDefinition for NASA standard L3 grids.

    Parameters
    ----------
    resolution : str, optional
        Grid resolution. One of '9km', '4km', '1km', or '500m',
        by default '4km'.
    lat1 : float, optional
        Minimum latitude (southern boundary), by default -90.
    lat2 : float, optional
        Maximum latitude (northern boundary), by default 90.
    lon1 : float, optional
        Minimum longitude (western boundary), by default -180.
    lon2 : float, optional
        Maximum longitude (eastern boundary), by default 180.
    **kw
        Additional keyword arguments (unused).

    Returns
    -------
    pyresample.AreaDefinition
        Area definition for the NASA grid.

    Raises
    ------
    ValueError
        If an unsupported resolution is specified.
    """
    if resolution == "9km":
        i0t, imt, j0t, jmt = (0000, 4320, 0, 2160)
    elif resolution == "4km":
        i0t, imt, j0t, jmt = (0000, 8640, 0, 4320)
    elif resolution == "1km":
        i0t, imt, j0t, jmt = (0000, 34560, 0, 17280)
    elif resolution == "500m":
        i0t, imt, j0t, jmt = (0000, 69120, 0, 34560)
    else:
        raise ValueError("Wrong resolution")
    incr = 360.0 / imt
    jR = np.arange(j0t, jmt)
    iR = np.arange(i0t, imt)
    latvec = (90 - jR * incr - incr / 2)[::-1]
    latvec = latvec[(latvec >= lat1) & (latvec <= lat2)]
    lonvec = -180 + iR * incr + incr / 2
    lonvec = lonvec[(lonvec >= lon1) & (lonvec <= lon2)]

    area = AreaDefinition(
        area_id="nasa_grid",
        description=f"NASA rectilinear {resolution} L3 grid.",
        proj_id="latlon",
        projection={"proj": "latlong"},
        width=len(lonvec),
        height=len(latvec),
        area_extent=[lonvec.min(), latvec.min(), lonvec.max(), latvec.max()],
    )
    return area

    """
    lons, lats = np.meshgrid(lonvec, latvec)
    grid = pr.geometry.GridDefinition(lons=lons, lats=lats)
    grid.ivec = np.arange(grid.shape[1])
    grid.jvec = np.arange(grid.shape[0])
    grid.iarr, grid.jarr = np.meshgrid(grid.ivec, grid.jvec)
    """

    return grid


def rectlinear(shape, lat1=-90, lat2=90, lon1=-180, lon2=180, **kw):
    """Create a generic rectilinear lat/lon AreaDefinition.

    Parameters
    ----------
    shape : tuple of int
        Shape of the grid as (height, width).
    lat1 : float, optional
        Minimum latitude (southern boundary), by default -90.
    lat2 : float, optional
        Maximum latitude (northern boundary), by default 90.
    lon1 : float, optional
        Minimum longitude (western boundary), by default -180.
    lon2 : float, optional
        Maximum longitude (eastern boundary), by default 180.
    **kw
        Additional keyword arguments (unused).

    Returns
    -------
    pyresample.AreaDefinition
        Area definition for the rectilinear grid.
    """
    area = AreaDefinition(
        area_id="rectlinear_grid",
        description=f"Generic rectlilinear L3 grid.",
        proj_id="latlon",
        projection={"proj": "latlong"},
        width=shape[1],
        height=shape[0],
        area_extent=[lon1, lat1, lon2, lat2],
    )
    return area
