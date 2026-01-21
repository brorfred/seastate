"""Command-line interface for the seaview processor.

This module provides CLI commands for generating and managing oceanographic
map tiles using the Typer framework.
"""

import seaview

def update():
    """Update tiles with yesterday's and today's fields."""
    seaview.today(force=False, sync=False)
    seaview.yesterday(force=True, sync=False)
