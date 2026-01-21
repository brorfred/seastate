"""Shared pytest fixtures for seastate tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_grid_data():
    """Provide sample 2D grid data for tile generation tests."""
    lats = np.linspace(-10, 10, 100)
    lons = np.linspace(-20, 20, 100)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    data = np.sin(lat_grid * 0.1) * np.cos(lon_grid * 0.1)
    return {
        "lats_1d": lats,
        "lons_1d": lons,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
        "data": data
    }


@pytest.fixture
def sample_flat_data():
    """Provide sample flattened data for single tile tests."""
    lats = np.linspace(-5, 5, 200)
    lons = np.linspace(-5, 5, 200)
    data = np.random.randn(200)
    return {
        "lats": lats,
        "lons": lons,
        "data": data
    }


@pytest.fixture
def mock_dataset():
    """Provide a mock xarray Dataset for testing data source functions."""
    mock_ds = MagicMock()
    mock_ds.latitude.min.return_value = -45
    mock_ds.latitude.max.return_value = -10
    mock_ds.longitude.min.return_value = -70
    mock_ds.longitude.max.return_value = -10
    mock_ds.latitude.data = np.linspace(-45, -10, 100)
    mock_ds.longitude.data = np.linspace(-70, -10, 100)
    return mock_ds


@pytest.fixture
def sample_layer_config():
    """Provide sample layer configuration for JSON tests."""
    return {
        "base_url": "https://example.com/tiles",
        "layers": [
            {
                "id": "ssh",
                "name": "Sea Surface Height",
                "url_template": "{base_url}/ssh/{date}/{z}/{x}/{y}.png",
                "attribution": "Copernicus Marine",
                "date_range": {
                    "start": "2025-01-01",
                    "end": "2025-01-15"
                },
                "exclusive": False,
                "collapsed": False
            },
            {
                "id": "ostia",
                "name": "Sea Surface Temperature",
                "url_template": "{base_url}/ostia/{date}/{z}/{x}/{y}.png",
                "attribution": "OSTIA",
                "date_range": {
                    "start": "2025-01-01",
                    "end": "2025-01-15"
                },
                "exclusive": False,
                "collapsed": False
            }
        ]
    }
