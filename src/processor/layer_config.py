
import json
import pathlib

import pandas as pd
import sysrsync

from . import config
settings = config.settings()

VERBOSE = False
def vprint(text):
    if VERBOSE:
        print(text)

def sync():
    update()
    sysrsync.run(source="layer_config.json",
                 destination="/var/www/html/cruise/",
                 destination_ssh='tvarminne',
                 options=['-az'],
                 sync_source_contents=True,
                 strict=True,
                 )


def find_first_last_tile_dates():
    tilepath = pathlib.Path(settings["tile_dir"])
    vprint(tilepath)
    layer_dates = dict()
    for layer in tilepath.glob("*"):
        vprint(layer)
        layer_name = layer.name
        date_range = pd.to_datetime([d.name for d in layer.glob("*")])
        vprint(date_range)
        layer_dates[layer.name] = dict(start=date_range.min().date(),
                                       end=date_range.max().date())
    return layer_dates

def update(json_file_path="layer_config.json"):

    layer_dates = find_first_last_tile_dates()
    vprint(layer_dates)
    update_date_ranges(json_file_path=json_file_path, layer_dates=layer_dates)

def update_date_ranges(json_file_path, layer_dates=None, output_file_path=None):
    """
    Update start and end dates in date_ranges for each layer individually in a JSON file.

    Parameters:
    -----------
    json_file_path : str
        Path to the input JSON file
    layer_dates : dict, optional
        Dictionary mapping layer IDs to their new date ranges.
        Format: {
            'layer_id': {
                'start': pd.Timestamp or str,  # optional
                'end': pd.Timestamp or str      # optional
            }
        }
    output_file_path : str, optional
        Path to save the updated JSON. If None, overwrites the input file.

    Returns:
    --------
    dict : Updated JSON data

    Examples:
    ---------
    # Update different dates for different layers using pandas Timestamps
    layer_dates = {
        'ssh': {'start': pd.Timestamp('2026-01-10'), 'end': pd.Timestamp('2026-01-15')},
        'sst': {'start': pd.Timestamp('2026-01-11'), 'end': pd.Timestamp('2026-01-16')},
        'globcolour': {'start': pd.Timestamp('2026-01-09'), 'end': pd.Timestamp('2026-01-14')}
    }
    update_date_ranges('data.json', layer_dates=layer_dates)

    # Can also use string dates (pandas will parse them)
    layer_dates = {
        'ssh': {'start': '2026-01-10', 'end': '2026-01-15'}
    }
    update_date_ranges('data.json', layer_dates=layer_dates)

    # Use date arithmetic with pandas
    base_date = pd.Timestamp('2026-01-10')
    layer_dates = {
        'ssh': {'start': base_date, 'end': base_date + pd.Timedelta(days=5)},
        'sst': {'start': base_date + pd.Timedelta(days=1), 'end': base_date + pd.Timedelta(days=6)}
    }
    update_date_ranges('data.json', layer_dates=layer_dates)
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


def generate_file(path="./"):
    payload = """
    {
      "base_url": "https://monhegan.unh.edu/cruise/tiles/bioreactors1",
      "layers": [
        {
          "id": "ssh",
          "name": "SSH CMEMS 0.125\u00b0  ",
          "url_template": "{base_url}/ssh/{date}/{z}/{x}/{y}.png",
          "attribution": "Copernicus 1/125\u00b0 SSH -0.75\u20130.75 m",
          "date_range": {
            "start": "2026-01-08",
            "end": "2026-01-14"
          },
          "exclusive": false,
          "collapsed": false
        },
        {
          "id": "ostia",
          "name": "SST OSTIA 5km     ",
          "url_template": "{base_url}/ostia/{date}/{z}/{x}/{y}.png",
          "attribution": "OSTIA 2km SST 10\u201328\u00b0C",
          "date_range": {
            "start": "2026-01-08",
            "end": "2026-01-14"
          },
          "exclusive": false,
          "collapsed": false
        },
        {
          "id": "globcolour",
          "name": "Chl GlobColour 4km",
          "url_template": "{base_url}/globcolour/{date}/{z}/{x}/{y}.png",
          "attribution": "Globcolour 4km Chl 0.01-100 mg/m3",
          "date_range": {
            "start": "2026-01-08",
            "end": "2026-01-14"
          },
          "exclusive": false,
          "collapsed": false
        }
      ]
    }"""
    with open(pathlib.Path(path) / "layer_config.json", "w") as fp:
        fp.write(payload)
