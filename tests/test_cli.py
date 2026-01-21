"""Tests for the seastate.cli module."""

from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from seastate.cli import app


runner = CliRunner()


class TestUpdateCommand:
    """Tests for the update CLI command."""

    @patch('seastate.config.change_env')
    def test_changes_env_when_not_default(self, mock_change_env):
        """update command should change environment when env is not DEFAULT."""
        result = runner.invoke(app, ["update", "--env", "production", "--no-sync"])

        # Should call change_env with the specified environment
        mock_change_env.assert_called_once_with("production")
        assert result.exit_code == 0

    @patch('seastate.config.change_env')
    def test_skips_env_change_for_default(self, mock_change_env):
        """update command should not change environment for DEFAULT."""
        result = runner.invoke(app, ["update", "--env", "DEFAULT", "--no-sync"])

        mock_change_env.assert_not_called()
        assert result.exit_code == 0

    def test_prints_environment_name(self):
        """update command should print the environment name."""
        result = runner.invoke(app, ["update", "--env", "test_env", "--no-sync"])

        assert "test_env" in result.output

    def test_accepts_sync_flag(self):
        """update command should accept --sync/--no-sync flag."""
        # Both variations should work without error
        result_sync = runner.invoke(app, ["update", "--sync"])
        assert result_sync.exit_code == 0

        result_no_sync = runner.invoke(app, ["update", "--no-sync"])
        assert result_no_sync.exit_code == 0


class TestShootCommand:
    """Tests for the shoot CLI command."""

    def test_shoot_outputs_message(self):
        """shoot command should output shooting message."""
        result = runner.invoke(app, ["shoot"])

        assert result.exit_code == 0
        assert "Shooting portal gun" in result.output


class TestLoadCommand:
    """Tests for the load CLI command."""

    def test_load_outputs_message(self):
        """load command should output loading message."""
        result = runner.invoke(app, ["load"])

        assert result.exit_code == 0
        assert "Loading portal gun" in result.output


class TestCallback:
    """Tests for the CLI callback (help text)."""

    def test_help_shows_description(self):
        """--help should show the app description."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "cruise support" in result.output.lower()
        assert "slippy tiles" in result.output.lower()

    def test_help_lists_commands(self):
        """--help should list available commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "update" in result.output
        assert "shoot" in result.output
        assert "load" in result.output


class TestCommandHelp:
    """Tests for individual command help."""

    def test_update_help(self):
        """update --help should show command options."""
        result = runner.invoke(app, ["update", "--help"])

        assert result.exit_code == 0
        assert "--env" in result.output
        assert "--sync" in result.output

    def test_shoot_help(self):
        """shoot --help should show command description."""
        result = runner.invoke(app, ["shoot", "--help"])

        assert result.exit_code == 0
        assert "portal gun" in result.output.lower()

    def test_load_help(self):
        """load --help should show command description."""
        result = runner.invoke(app, ["load", "--help"])

        assert result.exit_code == 0
        assert "portal gun" in result.output.lower()


class TestAppStructure:
    """Tests for the CLI app structure."""

    def test_app_is_typer_instance(self):
        """app should be a Typer instance."""
        from typer import Typer
        assert isinstance(app, Typer)

    def test_invalid_command_shows_error(self):
        """Invalid command should show error message."""
        result = runner.invoke(app, ["nonexistent_command"])

        assert result.exit_code != 0
