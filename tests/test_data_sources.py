"""Tests for the seastate.data_sources modules."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import xarray as xr
import numpy as np

from seastate.data_sources import cmems_ssh, ostia, globcolour, gebco_bathy


class TestCmemsSSH:
    """Tests for the cmems_ssh module."""

    def test_filename_format(self):
        """filename should return correct format."""
        result = cmems_ssh.filename("2025-06-15")
        assert result == "copernicus_SSH_2025-06-15.nc"

    def test_filename_handles_timestamp(self):
        """filename should handle pandas Timestamp."""
        result = cmems_ssh.filename(pd.Timestamp("2025-06-15"))
        assert result == "copernicus_SSH_2025-06-15.nc"

    def test_filename_handles_datetime(self):
        """filename should handle datetime objects."""
        from datetime import datetime
        result = cmems_ssh.filename(datetime(2025, 6, 15))
        assert result == "copernicus_SSH_2025-06-15.nc"

    @patch.object(cmems_ssh, 'DATADIR', new_callable=lambda: Path('/tmp/test'))
    @patch.object(cmems_ssh, 'retrieve')
    @patch('xarray.open_dataset')
    def test_open_dataset_downloads_if_missing(self, mock_open, mock_retrieve, mock_datadir):
        """open_dataset should call retrieve if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmems_ssh.DATADIR = Path(tmpdir)

            mock_ds = MagicMock()
            mock_open.return_value = mock_ds

            cmems_ssh.open_dataset("2025-06-15")

            mock_retrieve.assert_called_once()

    @patch.object(cmems_ssh, 'DATADIR')
    @patch.object(cmems_ssh, 'retrieve')
    @patch('xarray.open_dataset')
    def test_open_dataset_skips_download_if_exists(self, mock_open, mock_retrieve, mock_datadir):
        """open_dataset should not download if file already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cmems_ssh.DATADIR = tmppath

            # Create the file
            (tmppath / "copernicus_SSH_2025-06-15.nc").touch()

            mock_ds = MagicMock()
            mock_open.return_value = mock_ds

            cmems_ssh.open_dataset("2025-06-15")

            mock_retrieve.assert_not_called()

    @patch.object(cmems_ssh, 'DATADIR')
    @patch('copernicusmarine.subset')
    @patch.object(cmems_ssh, 'settings')
    def test_retrieve_calls_copernicusmarine(self, mock_settings, mock_subset, mock_datadir):
        """retrieve should call copernicusmarine.subset with correct parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cmems_ssh.DATADIR = tmppath

            mock_settings.__getitem__ = MagicMock(side_effect=lambda k: {
                "lon1": -70, "lon2": -10, "lat1": -45, "lat2": -10
            }.get(k))
            mock_settings.get = MagicMock(side_effect=lambda k: {
                "cmems_login": "user",
                "cmems_password": "pass"
            }.get(k))

            cmems_ssh.retrieve("2025-06-15")

            mock_subset.assert_called_once()
            call_kwargs = mock_subset.call_args[1]
            assert call_kwargs["minimum_longitude"] == -70
            assert call_kwargs["maximum_longitude"] == -10
            assert call_kwargs["minimum_latitude"] == -45
            assert call_kwargs["maximum_latitude"] == -10

    @patch.object(cmems_ssh, 'DATADIR')
    @patch('copernicusmarine.subset')
    @patch.object(cmems_ssh, 'settings')
    def test_retrieve_skips_if_file_exists_no_force(self, mock_settings, mock_subset, mock_datadir):
        """retrieve should skip if file exists and force=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cmems_ssh.DATADIR = tmppath

            # Create the file
            (tmppath / "copernicus_SSH_2025-06-15.nc").touch()

            cmems_ssh.retrieve("2025-06-15", force=False)

            mock_subset.assert_not_called()


class TestOstia:
    """Tests for the ostia module."""

    def test_filename_format(self):
        """filename should return correct format."""
        result = ostia.filename("2025-06-15")
        assert result == "copernicus_OSTIA_2025-06-15.nc"

    def test_filename_handles_timestamp(self):
        """filename should handle pandas Timestamp."""
        result = ostia.filename(pd.Timestamp("2025-06-15"))
        assert result == "copernicus_OSTIA_2025-06-15.nc"

    @patch.object(ostia, 'DATADIR')
    @patch.object(ostia, 'retrieve')
    @patch('xarray.open_dataset')
    def test_open_dataset_downloads_if_missing(self, mock_open, mock_retrieve, mock_datadir):
        """open_dataset should call retrieve if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ostia.DATADIR = Path(tmpdir)

            mock_ds = MagicMock()
            mock_open.return_value = mock_ds

            ostia.open_dataset("2025-06-15")

            mock_retrieve.assert_called_once()


class TestGlobcolour:
    """Tests for the globcolour module."""

    def test_filename_format(self):
        """filename should return correct format."""
        result = globcolour.filename("2025-06-15")
        assert result == "copernicus_GLOBCOLOUR_2025-06-15.nc"

    def test_filename_handles_timestamp(self):
        """filename should handle pandas Timestamp."""
        result = globcolour.filename(pd.Timestamp("2025-06-15"))
        assert result == "copernicus_GLOBCOLOUR_2025-06-15.nc"

    @patch.object(globcolour, 'DATADIR')
    @patch.object(globcolour, 'retrieve')
    @patch('xarray.open_dataset')
    def test_open_dataset_downloads_if_missing(self, mock_open, mock_retrieve, mock_datadir):
        """open_dataset should call retrieve if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            globcolour.DATADIR = Path(tmpdir)

            mock_ds = MagicMock()
            mock_open.return_value = mock_ds

            globcolour.open_dataset("2025-06-15")

            mock_retrieve.assert_called_once()

    def test_dataset_id_is_defined(self):
        """DATASET_ID should be defined for Copernicus API."""
        assert hasattr(globcolour, 'DATASET_ID')
        assert isinstance(globcolour.DATASET_ID, str)
        assert len(globcolour.DATASET_ID) > 0


class TestGebcoBathy:
    """Tests for the gebco_bathy module."""

    def test_filename_returns_static_name(self):
        """filename should return static gebco filename."""
        # dtm parameter is ignored for bathymetry
        result = gebco_bathy.filename()
        assert result == "gebco_2025_sub_ice.nc"

        result = gebco_bathy.filename("2025-01-01")
        assert result == "gebco_2025_sub_ice.nc"

    @patch.object(gebco_bathy, 'DATADIR')
    @patch.object(gebco_bathy, 'retrieve')
    @patch('xarray.open_dataset')
    def test_open_dataset_downloads_if_missing(self, mock_open, mock_retrieve, mock_datadir):
        """open_dataset should call retrieve if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gebco_bathy.DATADIR = Path(tmpdir)

            mock_ds = MagicMock()
            # Mock xarray dataset behavior
            mock_ds.rename.return_value = mock_ds
            mock_ds.sel.return_value = mock_ds
            mock_ds.where.return_value = mock_ds
            mock_open.return_value = mock_ds

            gebco_bathy.open_dataset()

            mock_retrieve.assert_called_once()

    @patch.object(gebco_bathy, 'DATADIR')
    @patch('xarray.open_dataset')
    def test_open_dataset_applies_subsampling(self, mock_open, mock_datadir):
        """open_dataset should apply lat/lon subsampling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            gebco_bathy.DATADIR = tmppath
            (tmppath / "gebco_2025_sub_ice.nc").touch()

            mock_ds = MagicMock()
            mock_ds.rename.return_value = mock_ds
            mock_ds.sel.return_value = mock_ds
            mock_ds.where.return_value = mock_ds
            mock_open.return_value = mock_ds

            gebco_bathy.open_dataset(step=10)

            # Should call sel with slice
            mock_ds.sel.assert_called()


class TestVerbosePrint:
    """Tests for the vprint functions in data source modules."""

    def test_cmems_ssh_vprint(self, capsys):
        """cmems_ssh.vprint should print when VERBOSE is True."""
        original = cmems_ssh.VERBOSE
        try:
            cmems_ssh.VERBOSE = True
            cmems_ssh.vprint("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out
        finally:
            cmems_ssh.VERBOSE = original

    def test_cmems_ssh_vprint_silent(self, capsys):
        """cmems_ssh.vprint should be silent when VERBOSE is False."""
        original = cmems_ssh.VERBOSE
        try:
            cmems_ssh.VERBOSE = False
            cmems_ssh.vprint("test message")
            captured = capsys.readouterr()
            assert captured.out == ""
        finally:
            cmems_ssh.VERBOSE = original

    def test_ostia_vprint(self, capsys):
        """ostia.vprint should print when VERBOSE is True."""
        original = ostia.VERBOSE
        try:
            ostia.VERBOSE = True
            ostia.vprint("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out
        finally:
            ostia.VERBOSE = original

    def test_globcolour_vprint(self, capsys):
        """globcolour.vprint should print when VERBOSE is True."""
        original = globcolour.VERBOSE
        try:
            globcolour.VERBOSE = True
            globcolour.vprint("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out
        finally:
            globcolour.VERBOSE = original

    def test_gebco_vprint(self, capsys):
        """gebco_bathy.vprint should print when VERBOSE is True."""
        original = gebco_bathy.VERBOSE
        try:
            gebco_bathy.VERBOSE = True
            gebco_bathy.vprint("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out
        finally:
            gebco_bathy.VERBOSE = original
