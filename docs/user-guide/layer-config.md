# Layer Configuration

The layer configuration system manages the JSON file that tells web clients about available map layers.

## Overview

The `layer_config.json` file defines:

- Available map layers (SSH, SST, Chlorophyll)
- Date ranges for each layer
- Tile URL templates
- Layer metadata and attribution

## Generating Configuration

Generate a new configuration file:

```python
from seastate.layer_config import generate_file

# Generate in current directory
generate_file()

# Generate in specific directory
generate_file(config_file_path="/path/to/output")

# Override tile URL
generate_file(remote_tile_url="https://custom.tiles.com/tiles")
```

## Updating Date Ranges

After generating new tiles, update the date ranges:

```python
from seastate.layer_config import update, find_first_last_tile_dates

# Auto-detect date ranges from tile directories
update("layer_config.json")

# Or manually specify dates
from seastate.layer_config import update_date_ranges

layer_dates = {
    'ssh': {'start': '2026-01-01', 'end': '2026-01-15'},
    'ostia': {'start': '2026-01-01', 'end': '2026-01-15'},
    'globcolour': {'start': '2026-01-01', 'end': '2026-01-15'}
}
update_date_ranges('layer_config.json', layer_dates=layer_dates)
```

## Syncing to Remote

Sync the configuration to your remote server:

```python
from seastate.layer_config import sync

# Generate, update, and sync in one call
sync()
```

This will:
1. Generate a fresh configuration file
2. Update date ranges from tile directories
3. Upload to the configured remote server

## Configuration File Format

```json
{
  "base_url": "https://tiles.example.com/tiles/cruise_2026",
  "layers": [
    {
      "id": "ssh",
      "name": "SSH CMEMS 0.125°",
      "url_template": "{base_url}/ssh/{date}/{z}/{x}/{y}.png",
      "attribution": "Copernicus 1/125° SSH -0.75–0.75 m",
      "date_range": {
        "start": "2026-01-01",
        "end": "2026-01-15"
      },
      "exclusive": false,
      "collapsed": false
    },
    {
      "id": "ostia",
      "name": "SST OSTIA 5km",
      "url_template": "{base_url}/ostia/{date}/{z}/{x}/{y}.png",
      "attribution": "OSTIA 2km SST 10–28°C",
      "date_range": {
        "start": "2026-01-01",
        "end": "2026-01-15"
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
        "start": "2026-01-01",
        "end": "2026-01-15"
      },
      "exclusive": false,
      "collapsed": false
    }
  ]
}
```

## Configuration Options

### Layer Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | string | Unique layer identifier |
| `name` | string | Display name |
| `url_template` | string | Tile URL template with placeholders |
| `attribution` | string | Data source attribution |
| `date_range.start` | string | First available date (YYYY-MM-DD) |
| `date_range.end` | string | Last available date (YYYY-MM-DD) |
| `exclusive` | boolean | If true, only one layer visible at a time |
| `collapsed` | boolean | Layer group collapsed by default |

### URL Template Placeholders

| Placeholder | Description |
|-------------|-------------|
| `{base_url}` | Base URL from configuration |
| `{date}` | Selected date (YYYY-MM-DD) |
| `{z}` | Zoom level |
| `{x}` | Tile column |
| `{y}` | Tile row |

## Web Client Integration

### Leaflet Example

```javascript
// Load layer configuration
fetch('layer_config.json')
  .then(response => response.json())
  .then(config => {
    config.layers.forEach(layer => {
      const tileLayer = L.tileLayer(
        layer.url_template
          .replace('{base_url}', config.base_url)
          .replace('{date}', selectedDate),
        {
          attribution: layer.attribution,
          opacity: 0.7
        }
      );
      layerControl.addOverlay(tileLayer, layer.name);
    });
  });
```

### Date Slider Integration

```javascript
// Update tiles when date changes
function onDateChange(newDate) {
  activeLayers.forEach(layer => {
    layer.setUrl(
      layer.options.urlTemplate.replace('{date}', newDate)
    );
  });
}
```

## Automatic Updates

Set up a cron job to automatically process and sync:

```bash
# Run daily at 6 AM
0 6 * * * cd /path/to/cruise && python -c "import seastate; seastate.yesterday; yesterday()"
```

This will:
1. Download latest data
2. Generate tiles
3. Sync tiles and configuration to remote server
