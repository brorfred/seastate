"""Tests for the seaview.tile module."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from seaview import tile


class TestTilesExists:
    """Tests for the tiles_exists function."""

    @patch.object(tile, 'settings')
    def test_returns_true_when_dir_exists(self, mock_settings):
        """tiles_exists should return True when tile directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the tile directory
            tile_path = Path(tmpdir) / "ssh" / "2025-01-15"
            tile_path.mkdir(parents=True)

            mock_settings.__getitem__ = MagicMock(return_value=tmpdir)

            result = tile.tiles_exists("ssh", "2025-01-15")
            assert result is True

    @patch.object(tile, 'settings')
    def test_returns_false_when_dir_missing(self, mock_settings):
        """tiles_exists should return False when tile directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.__getitem__ = MagicMock(return_value=tmpdir)

            result = tile.tiles_exists("ssh", "2025-01-15")
            assert result is False

    @patch.object(tile, 'settings')
    def test_handles_various_date_formats(self, mock_settings):
        """tiles_exists should handle various date input formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tile_path = Path(tmpdir) / "ssh" / "2025-01-15"
            tile_path.mkdir(parents=True)

            mock_settings.__getitem__ = MagicMock(return_value=tmpdir)

            # String format
            assert tile.tiles_exists("ssh", "2025-01-15") is True

            # Timestamp format
            assert tile.tiles_exists("ssh", pd.Timestamp("2025-01-15")) is True

            # Different date format
            assert tile.tiles_exists("ssh", "2025/01/15") is True


class TestSSH:
    """Tests for the ssh function."""

    @patch.object(tile, 'cmems_ssh')
    @patch.object(tile, 'rectlin_tiler')
    @patch.object(tile, 'settings')
    @patch.object(tile, 'tiles_exists')
    def test_skips_when_tiles_exist_and_no_force(
        self, mock_exists, mock_settings, mock_tiler, mock_cmems
    ):
        """ssh should skip processing when tiles exist and force=False."""
        mock_exists.return_value = True

        tile.ssh("2025-01-15", force=False)

        # Should not call data retrieval
        mock_cmems.open_dataset.assert_not_called()

    @patch.object(tile, 'cmems_ssh')
    @patch.object(tile, 'rectlin_tiler')
    @patch.object(tile, 'settings')
    @patch.object(tile, 'tiles_exists')
    def test_processes_when_force_true(
        self, mock_exists, mock_settings, mock_tiler, mock_cmems
    ):
        """ssh should process even when tiles exist if force=True."""
        mock_exists.return_value = True

        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.latitude.min.return_value = -10
        mock_ds.latitude.max.return_value = 10
        mock_ds.longitude.min.return_value = -20
        mock_ds.longitude.max.return_value = 20
        mock_ds.latitude.data = np.linspace(-10, 10, 100)
        mock_ds.longitude.data = np.linspace(-20, 20, 100)
        mock_ds.sla.data = np.random.randn(100, 100)
        mock_cmems.open_dataset.return_value = mock_ds

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.__getitem__ = MagicMock(side_effect=lambda k: {
                "tile_dir": tmpdir,
                "zoom_levels": [0, 1]
            }.get(k))

            tile.ssh("2025-01-15", force=True)

            mock_cmems.open_dataset.assert_called_once()

    @patch.object(tile, 'cmems_ssh')
    @patch.object(tile, 'settings')
    @patch.object(tile, 'tiles_exists')
    def test_handles_coordinates_out_of_bounds(
        self, mock_exists, mock_settings, mock_cmems, capsys
    ):
        """ssh should handle CoordinatesOutOfDatasetBounds gracefully."""
        from copernicusmarine import CoordinatesOutOfDatasetBounds

        mock_exists.return_value = False
        mock_cmems.open_dataset.side_effect = CoordinatesOutOfDatasetBounds("test")

        # Should not raise
        tile.ssh("2025-01-15", force=True)

        captured = capsys.readouterr()
        assert "failed" in captured.out.lower()


class TestSST:
    """Tests for the sst function."""

    @patch.object(tile, 'ostia')
    def test_calls_ostia(self, mock_ostia):
        """sst should be an alias for ostia."""
        tile.sst("2025-01-15", verbose=False, force=True)
        mock_ostia.assert_called_once_with("2025-01-15", False, True)


class TestOstia:
    """Tests for the ostia function."""

    @patch.object(tile, 'ostia_sst')
    @patch.object(tile, 'rectlin_tiler')
    @patch.object(tile, 'settings')
    @patch.object(tile, 'tiles_exists')
    def test_skips_when_tiles_exist_and_no_force(
        self, mock_exists, mock_settings, mock_tiler, mock_ostia
    ):
        """ostia should skip processing when tiles exist and force=False."""
        mock_exists.return_value = True

        tile.ostia("2025-01-15", force=False)

        mock_ostia.open_dataset.assert_not_called()


class TestGlobcolour:
    """Tests for the globcolour function."""

    @patch.object(tile, 'cmems_globcolour')
    @patch.object(tile, 'rectlin_tiler')
    @patch.object(tile, 'settings')
    @patch.object(tile, 'tiles_exists')
    def test_skips_when_tiles_exist_and_no_force(
        self, mock_exists, mock_settings, mock_tiler, mock_globcolour
    ):
        """globcolour should skip processing when tiles exist and force=False."""
        mock_exists.return_value = True

        tile.globcolour("2025-01-15", force=False)

        mock_globcolour.open_dataset.assert_not_called()


class TestBathy:
    """Tests for the bathy function."""

    @patch.object(tile, 'gebco_bathy')
    @patch.object(tile, 'rectlin_tiler')
    @patch.object(tile, 'settings')
    def test_skips_when_tiles_exist_and_no_force(
        self, mock_settings, mock_tiler, mock_gebco
    ):
        """bathy should skip processing when tiles exist and force=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the tile directory
            tile_path = Path(tmpdir) / "gebco"
            tile_path.mkdir(parents=True)

            mock_settings.__getitem__ = MagicMock(return_value=tmpdir)

            tile.bathy(force=False)

            mock_gebco.open_dataset.assert_not_called()


class TestAll:
    """Tests for the all function."""

    @patch.object(tile, 'globcolour')
    @patch.object(tile, 'sst')
    @patch.object(tile, 'ssh')
    def test_calls_all_tile_generators(self, mock_ssh, mock_sst, mock_globcolour):
        """all should call ssh, sst, and globcolour functions."""
        tile.all("2025-01-15", force=True, verbose=False)

        mock_ssh.assert_called_once_with("2025-01-15", verbose=False, force=True)
        mock_sst.assert_called_once_with("2025-01-15", verbose=False, force=True)
        mock_globcolour.assert_called_once_with("2025-01-15", verbose=False, force=True)


class TestSync:
    """Tests for the sync function."""

    @patch('sysrsync.run')
    @patch.object(tile, 'settings')
    def test_calls_sysrsync(self, mock_settings, mock_rsync):
        """sync should call sysrsync.run with correct parameters."""
        mock_settings.__getitem__ = MagicMock(side_effect=lambda k: {
            "tile_dir": "/local/tiles",
            "remote_tile_dir": "/remote/tiles"
        }.get(k))

        tile.sync()

        mock_rsync.assert_called_once()
        call_kwargs = mock_rsync.call_args[1]
        assert call_kwargs["source"] == "/local/tiles"
        assert call_kwargs["destination"] == "/remote/tiles"
        assert call_kwargs["destination_ssh"] == "tvarminne"
        assert "-az" in call_kwargs["options"]
