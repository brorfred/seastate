"""Layer configuration management for map tile server.

This module provides functions to generate, update, and synchronize
JSON configuration files that define map layer settings for the
tile server web interface.
"""
import json
import pathlib

import pandas as pd
import sysrsync
from jinja2 import Template

from . import config
settings = config.settings

VERBOSE = False


def vprint(text):
    """Print text if verbose mode is enabled.

    Parameters
    ----------
    text : str
        Text to print.
    """
    if VERBOSE:
        print(text)


def sync():
    """Generate and sync layer configuration to remote server.

    Generates a new layer configuration file, updates it with current
    tile date ranges, and syncs to the remote server via rsync.
    """
    tmp_dir = pathlib.Path("/tmp")
    generate_file(config_file_path=tmp_dir)
    update(json_file_path=tmp_dir / "layer_config.json")
    sysrsync.run(source=str(tmp_dir / "layer_config.json"),
                 destination=settings.get("remote_html_dir"),
                 destination_ssh=settings.get("remote_server"),
                 options=['-azv'],
                 strict=True,
                 )


def find_first_last_tile_dates():
    """Find the date range of available tiles for each layer.

    Scans the tile directory to determine the earliest and latest
    available tile dates for each layer, respecting the maximum
    tile days setting.

    Returns
    -------
    dict
        Dictionary mapping layer names to date range dictionaries
        with 'start' and 'end' keys containing date objects.
    """
    tilepath = pathlib.Path(settings["tile_dir"])
    vprint(tilepath)
    layer_dates = dict()
    for layer_name in settings.get("updated_tiles"):
        layer = tilepath / layer_name
        vprint(layer)
        date_range = pd.to_datetime([d.name for d in layer.glob("*")])
        vprint(date_range)
        dtm1 = date_range.min()
        dtm2 = date_range.max()
        ddtm = pd.Timedelta(min((dtm2-dtm1).days, settings.get("max_tile_days")-1), "D")
        layer_dates[layer.name] = dict(start=(dtm2 - ddtm).date(), end=dtm2.date())
    return layer_dates


def update(json_file_path="layer_config.json"):
    """Update layer configuration file with current tile date ranges.

    Parameters
    ----------
    json_file_path : str or pathlib.Path, optional
        Path to the JSON configuration file, by default "layer_config.json".
    """
    layer_dates = find_first_last_tile_dates()
    vprint(layer_dates)
    update_date_ranges(json_file_path=json_file_path, layer_dates=layer_dates)


def update_date_ranges(json_file_path, layer_dates=None, output_file_path=None):
    """Update start and end dates for each layer in a JSON configuration file.

    Parameters
    ----------
    json_file_path : str or pathlib.Path
        Path to the input JSON file.
    layer_dates : dict, optional
        Dictionary mapping layer IDs to their new date ranges.
        Format: {'layer_id': {'start': date, 'end': date}}.
        Dates can be pd.Timestamp, datetime, or string.
    output_file_path : str or pathlib.Path, optional
        Path to save the updated JSON. If None, overwrites the input file.

    Examples
    --------
    Update different dates for different layers using pandas Timestamps:

    >>> layer_dates = {
    ...     'ssh': {'start': pd.Timestamp('2026-01-10'), 'end': pd.Timestamp('2026-01-15')},
    ...     'sst': {'start': pd.Timestamp('2026-01-11'), 'end': pd.Timestamp('2026-01-16')},
    ... }
    >>> update_date_ranges('data.json', layer_dates=layer_dates)

    Use string dates (pandas will parse them):

    >>> layer_dates = {'ssh': {'start': '2026-01-10', 'end': '2026-01-15'}}
    >>> update_date_ranges('data.json', layer_dates=layer_dates)
    """

    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Update each layer's date_range based on layer_dates
    if layer_dates:
        for layer in data.get('layers', []):
            layer_id = layer.get('id')

            # Check if this layer has updates
            if layer_id in layer_dates and 'date_range' in layer:
                vprint(layer_id)
                updates = layer_dates[layer_id]

                # Update start date if provided
                if 'start' in updates:
                    start_date = pd.Timestamp(updates['start'])
                    vprint(start_date)
                    layer['date_range']['start'] = start_date.strftime('%Y-%m-%d')

                # Update end date if provided
                if 'end' in updates:
                    end_date = pd.Timestamp(updates['end'])
                    vprint(end_date)
                    layer['date_range']['end'] = end_date.strftime('%Y-%m-%d')

    # Save the updated JSON
    output_path = output_file_path or json_file_path
    vprint(output_path)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_file(remote_tile_url=None, config_file_path="./"):
    """Generate a JSON configuration file for map layers.

    Creates a layer_config.json file with definitions for all available
    map layers including SSH, SST, and chlorophyll products.

    Parameters
    ----------
    remote_tile_url : str, optional
        The base URL for the tile server. If None, uses settings.
    config_file_path : str or pathlib.Path, optional
        Directory path where the file will be saved, by default "./".
    """
    base_url = settings.get("remote_url")+"/tiles/"+settings.get("cruise_name")
    base_url = remote_tile_url or base_url
    # Define layer configurations
    layers_config = [
        {
            "id": "ssh",
            "name": "SSH CMEMS 0.125°  ",
            "url_path": "ssh",
            "attribution": "Copernicus 1/125° SSH -0.75–0.75 m",
        },
        {
            "id": "ostia",
            "name": "SST OSTIA 5km     ",
            "url_path": "ostia",
            "attribution": "OSTIA 2km SST 10–28°C",
        },
        {
            "id": "globcolour",
            "name": "Chl GlobColour 4km",
            "url_path": "globcolour",
            "attribution": "Globcolour 4km Chl 0.01-100 mg/m3",
        },
    ]

    # Common settings for all layers
    common_settings = {
        "date_range": {
            "start": "2026-01-08",
            "end": "2026-01-14"
        },
        "exclusive": False,
        "collapsed": False
    }

    # Build complete layer objects
    layers = []
    for layer in layers_config:
        layer_obj = {
            "id": layer["id"],
            "name": layer["name"],
            "url_template": f"{{base_url}}/{layer['url_path']}/{{date}}/{{z}}/{{x}}/{{y}}.png",
            "attribution": layer["attribution"],
            **common_settings
        }
        layers.append(layer_obj)

    # Jinja2 template for the JSON structure
    template_str = """
{
  "base_url": "{{ base_url }}",
  "layers": {{ layers | tojson(indent=4) }}
}"""

    template = Template(template_str)

    # Render the template
    rendered = template.render(
        base_url=base_url,
        layers=layers
    )

    # Write to file
    output_path = pathlib.Path(config_file_path) / "layer_config.json"
    with open(output_path, "w") as fp:
        fp.write(rendered)
