"""Tests for the seaview.layer_config module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from seaview import layer_config


class TestGenerateFile:
    """Tests for the generate_file function."""

    def test_creates_json_file(self):
        """generate_file should create a layer_config.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_config.generate_file(
                remote_tile_url="https://example.com/tiles",
                config_file_path=tmpdir
            )

            output_file = Path(tmpdir) / "layer_config.json"
            assert output_file.exists()

    def test_json_has_required_structure(self):
        """Generated JSON should have base_url and layers keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_config.generate_file(
                remote_tile_url="https://example.com/tiles",
                config_file_path=tmpdir
            )

            output_file = Path(tmpdir) / "layer_config.json"
            with open(output_file) as f:
                data = json.load(f)

            assert "base_url" in data
            assert "layers" in data
            assert isinstance(data["layers"], list)

    def test_layers_have_required_fields(self):
        """Each layer should have id, name, url_template, attribution, date_range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_config.generate_file(
                remote_tile_url="https://example.com/tiles",
                config_file_path=tmpdir
            )

            output_file = Path(tmpdir) / "layer_config.json"
            with open(output_file) as f:
                data = json.load(f)

            for layer in data["layers"]:
                assert "id" in layer
                assert "name" in layer
                assert "url_template" in layer
                assert "attribution" in layer
                assert "date_range" in layer
                assert "start" in layer["date_range"]
                assert "end" in layer["date_range"]

    def test_includes_expected_layers(self):
        """Generated config should include ssh, ostia, and globcolour layers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layer_config.generate_file(
                remote_tile_url="https://example.com/tiles",
                config_file_path=tmpdir
            )

            output_file = Path(tmpdir) / "layer_config.json"
            with open(output_file) as f:
                data = json.load(f)

            layer_ids = [layer["id"] for layer in data["layers"]]
            assert "ssh" in layer_ids
            assert "ostia" in layer_ids
            assert "globcolour" in layer_ids

    def test_base_url_matches_parameter(self):
        """base_url in JSON should match the provided remote_tile_url."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_url = "https://my-server.com/custom/tiles"
            layer_config.generate_file(
                remote_tile_url=test_url,
                config_file_path=tmpdir
            )

            output_file = Path(tmpdir) / "layer_config.json"
            with open(output_file) as f:
                data = json.load(f)

            assert data["base_url"] == test_url


class TestUpdateDateRanges:
    """Tests for the update_date_ranges function."""

    def _create_test_json(self, tmpdir):
        """Helper to create a test JSON file."""
        test_data = {
            "base_url": "https://example.com",
            "layers": [
                {
                    "id": "ssh",
                    "name": "SSH",
                    "date_range": {"start": "2025-01-01", "end": "2025-01-10"}
                },
                {
                    "id": "ostia",
                    "name": "SST",
                    "date_range": {"start": "2025-01-01", "end": "2025-01-10"}
                }
            ]
        }
        json_path = Path(tmpdir) / "test_config.json"
        with open(json_path, "w") as f:
            json.dump(test_data, f)
        return json_path

    def test_updates_start_date(self):
        """update_date_ranges should update start date for specified layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = self._create_test_json(tmpdir)

            layer_dates = {
                "ssh": {"start": "2025-02-01", "end": "2025-02-15"}
            }
            layer_config.update_date_ranges(json_path, layer_dates=layer_dates)

            with open(json_path) as f:
                data = json.load(f)

            ssh_layer = next(l for l in data["layers"] if l["id"] == "ssh")
            assert ssh_layer["date_range"]["start"] == "2025-02-01"
            assert ssh_layer["date_range"]["end"] == "2025-02-15"

    def test_preserves_unmodified_layers(self):
        """update_date_ranges should not modify layers not in layer_dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = self._create_test_json(tmpdir)

            layer_dates = {
                "ssh": {"start": "2025-02-01", "end": "2025-02-15"}
            }
            layer_config.update_date_ranges(json_path, layer_dates=layer_dates)

            with open(json_path) as f:
                data = json.load(f)

            ostia_layer = next(l for l in data["layers"] if l["id"] == "ostia")
            # Should be unchanged
            assert ostia_layer["date_range"]["start"] == "2025-01-01"
            assert ostia_layer["date_range"]["end"] == "2025-01-10"

    def test_writes_to_output_file(self):
        """update_date_ranges should write to output_file_path if specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = self._create_test_json(tmpdir)
            output_path = Path(tmpdir) / "output_config.json"

            layer_dates = {"ssh": {"start": "2025-03-01", "end": "2025-03-10"}}
            layer_config.update_date_ranges(
                json_path,
                layer_dates=layer_dates,
                output_file_path=output_path
            )

            assert output_path.exists()
            # Original should be unchanged
            with open(json_path) as f:
                original = json.load(f)
            ssh_original = next(l for l in original["layers"] if l["id"] == "ssh")
            assert ssh_original["date_range"]["start"] == "2025-01-01"

    def test_accepts_pandas_timestamps(self):
        """update_date_ranges should accept pandas Timestamp objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = self._create_test_json(tmpdir)

            layer_dates = {
                "ssh": {
                    "start": pd.Timestamp("2025-04-01"),
                    "end": pd.Timestamp("2025-04-15")
                }
            }
            layer_config.update_date_ranges(json_path, layer_dates=layer_dates)

            with open(json_path) as f:
                data = json.load(f)

            ssh_layer = next(l for l in data["layers"] if l["id"] == "ssh")
            assert ssh_layer["date_range"]["start"] == "2025-04-01"
            assert ssh_layer["date_range"]["end"] == "2025-04-15"

    def test_accepts_date_objects(self):
        """update_date_ranges should accept date objects."""
        from datetime import date
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = self._create_test_json(tmpdir)

            layer_dates = {
                "ssh": {
                    "start": date(2025, 5, 1),
                    "end": date(2025, 5, 15)
                }
            }
            layer_config.update_date_ranges(json_path, layer_dates=layer_dates)

            with open(json_path) as f:
                data = json.load(f)

            ssh_layer = next(l for l in data["layers"] if l["id"] == "ssh")
            assert ssh_layer["date_range"]["start"] == "2025-05-01"
            assert ssh_layer["date_range"]["end"] == "2025-05-15"

    def test_handles_empty_layer_dates(self):
        """update_date_ranges should handle None or empty layer_dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = self._create_test_json(tmpdir)

            # Should not raise with None
            layer_config.update_date_ranges(json_path, layer_dates=None)

            # Should not raise with empty dict
            layer_config.update_date_ranges(json_path, layer_dates={})


class TestFindFirstLastTileDates:
    """Tests for the find_first_last_tile_dates function."""

    @patch.object(layer_config, 'settings')
    def test_returns_dict_of_date_ranges(self, mock_settings):
        """find_first_last_tile_dates should return dict mapping layer names to date ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test tile directories
            ssh_dir = Path(tmpdir) / "ssh"
            ssh_dir.mkdir()
            (ssh_dir / "2025-01-01").mkdir()
            (ssh_dir / "2025-01-05").mkdir()
            (ssh_dir / "2025-01-10").mkdir()

            mock_settings.__getitem__ = MagicMock(return_value=tmpdir)
            mock_settings.get = MagicMock(side_effect=lambda k: {
                "tile_dir": tmpdir,
                "updated_tiles": ["ssh"],
                "max_tile_days": 30
            }.get(k))

            result = layer_config.find_first_last_tile_dates()

            assert isinstance(result, dict)
            assert "ssh" in result
            assert "start" in result["ssh"]
            assert "end" in result["ssh"]


class TestVprint:
    """Tests for the vprint helper function."""

    def test_prints_when_verbose_true(self, capsys):
        """vprint should print when VERBOSE is True."""
        original_verbose = layer_config.VERBOSE
        try:
            layer_config.VERBOSE = True
            layer_config.vprint("test message")
            captured = capsys.readouterr()
            assert "test message" in captured.out
        finally:
            layer_config.VERBOSE = original_verbose

    def test_silent_when_verbose_false(self, capsys):
        """vprint should not print when VERBOSE is False."""
        original_verbose = layer_config.VERBOSE
        try:
            layer_config.VERBOSE = False
            layer_config.vprint("test message")
            captured = capsys.readouterr()
            assert captured.out == ""
        finally:
            layer_config.VERBOSE = original_verbose
