"""Command-line interface for the seaview processor.

This module provides CLI commands for generating and managing oceanographic
map tiles using the Typer framework.
"""
import typer
from typing import Annotated


import seaview

app = typer.Typer()

@app.command()
def update(
    env: Annotated[str, typer.Option(help="Environment used in settings file.")] = "DEFAULT",
    sync: Annotated[bool, typer.Option(help="Sync tiles to remote.")] = True,

):
    """Update tiles with yesterday's and today's fields."""

    if env.lower() != "default":
        seaview.config.change_env(env)
        typer.echo(f"Using environment: {env}")
    seaview.today(force=False, sync=False)
    seaview.yesterday(force=True, sync=sync)


@app.callback()
def callback():
    """A cruise support system converting geophysical fields to slippy tiles.

    Currently SST, SSH, and Chl from Copernicus are available.
    """




@app.command()
def shoot():
    """Shoot the portal gun."""
    typer.echo("Shooting portal gun")


@app.command()
def load():
    """Load the portal gun."""
    typer.echo("Loading portal gun")
