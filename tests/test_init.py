"""Tests for the seastate package __init__ module."""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import seastate
from seastate import DateInFutureError


class TestDateInFutureError:
    """Tests for the DateInFutureError exception."""

    def test_is_exception(self):
        """DateInFutureError should be an Exception subclass."""
        assert issubclass(DateInFutureError, Exception)

    def test_can_be_raised(self):
        """DateInFutureError should be raisable with a message."""
        with pytest.raises(DateInFutureError):
            raise DateInFutureError("Date is in the future")


class TestDay:
    """Tests for the day function."""

    @patch.object(seastate.tile, 'all')
    def test_calls_tile_all(self, mock_all):
        """day should call tile.all with the provided date."""
        seastate.day("2025-01-15", force=True, verbose=False)
        mock_all.assert_called_once_with("2025-01-15", force=True, verbose=False)

    @patch.object(seastate.tile, 'all')
    def test_default_parameters(self, mock_all):
        """day should use default force=False and verbose=False."""
        seastate.day("2025-01-15")
        mock_all.assert_called_once_with("2025-01-15", force=False, verbose=False)


class TestToday:
    """Tests for the today function."""

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_processes_todays_date(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """today should process the current date."""
        mock_settings.get = MagicMock(return_value=False)

        seastate.today(force=False, sync=False)

        # Should call tile.all with today's date
        assert mock_all.called
        call_args = mock_all.call_args[0][0]
        assert isinstance(call_args, pd.Timestamp)
        assert call_args.date() == pd.Timestamp.now().normalize().date()

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_syncs_when_remote_sync_enabled(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """today should sync when remote_sync is enabled and sync=True."""
        mock_settings.get = MagicMock(return_value=True)

        seastate.today(force=False, sync=True)

        mock_tile_sync.assert_called_once()
        mock_layer_sync.assert_called_once()

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_skips_sync_when_disabled(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """today should skip sync when sync=False."""
        mock_settings.get = MagicMock(return_value=True)

        seastate.today(force=False, sync=False)

        mock_tile_sync.assert_not_called()
        mock_layer_sync.assert_not_called()


class TestYesterday:
    """Tests for the yesterday function."""

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_processes_yesterdays_date(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """yesterday should process yesterday's date."""
        mock_settings.get = MagicMock(return_value=False)

        seastate.yesterday(force=False, sync=False)

        # Should call tile.all with yesterday's date
        assert mock_all.called
        call_args = mock_all.call_args[0][0]
        assert isinstance(call_args, pd.Timestamp)
        expected_date = (pd.Timestamp.now().normalize() - pd.Timedelta(1, "D")).date()
        assert call_args.date() == expected_date

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_syncs_when_enabled(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """yesterday should sync when remote_sync is enabled."""
        mock_settings.get = MagicMock(return_value=True)

        seastate.yesterday(force=False, sync=True)

        mock_tile_sync.assert_called_once()
        mock_layer_sync.assert_called_once()


class TestLastDays:
    """Tests for the last_days function."""

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_processes_multiple_days(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """last_days should process the specified number of days."""
        mock_settings.get = MagicMock(return_value=False)

        seastate.last_days(days=3, sync=False)

        # Should call tile.all for each day in range
        # days=3 means 4 calls: today, yesterday, day before yesterday, 3 days ago
        assert mock_all.call_count >= 3

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_default_days_is_seven(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """last_days should default to 7 days."""
        mock_settings.get = MagicMock(return_value=False)

        seastate.last_days(sync=False)

        # Should call tile.all for 8 days (7 days back + today)
        assert mock_all.call_count >= 7

    @patch.object(seastate.layer_config, 'sync')
    @patch.object(seastate.tile, 'sync')
    @patch.object(seastate.tile, 'all')
    @patch.object(seastate.config, 'settings')
    def test_syncs_when_enabled(self, mock_settings, mock_all, mock_tile_sync, mock_layer_sync):
        """last_days should sync when remote_sync is enabled."""
        mock_settings.get = MagicMock(return_value=True)

        seastate.last_days(days=2, sync=True)

        mock_tile_sync.assert_called_once()
        mock_layer_sync.assert_called_once()


class TestPackageExports:
    """Tests for package-level exports."""

    def test_config_is_accessible(self):
        """seastate.config should be accessible."""
        assert hasattr(seastate, 'config')

    def test_tile_is_accessible(self):
        """seastate.tile should be accessible."""
        assert hasattr(seastate, 'tile')

    def test_layer_config_is_accessible(self):
        """seastate.layer_config should be accessible."""
        assert hasattr(seastate, 'layer_config')

    def test_day_function_is_accessible(self):
        """seastate.day should be accessible."""
        assert hasattr(seastate, 'day')
        assert callable(seastate.day)

    def test_today_function_is_accessible(self):
        """seastate.today should be accessible."""
        assert hasattr(seastate, 'today')
        assert callable(seastate.today)

    def test_yesterday_function_is_accessible(self):
        """seastate.yesterday should be accessible."""
        assert hasattr(seastate, 'yesterday')
        assert callable(seastate.yesterday)

    def test_last_days_function_is_accessible(self):
        """seastate.last_days should be accessible."""
        assert hasattr(seastate, 'last_days')
        assert callable(seastate.last_days)
