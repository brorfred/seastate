# Quick Start

This guide will help you generate your first oceanographic map tiles.

## Prerequisites

1. Install the Cruise Processor ([Installation Guide](installation.md))
2. Configure your settings ([Configuration Guide](configuration.md))
3. Ensure you have valid CMEMS credentials

## Generate Tiles for Yesterday

The simplest way to generate tiles is to process yesterday's data:

```python
import seastate

# Process all products (SSH, SST, Chlorophyll) for yesterday
seastate.yesterday()
```

This will:

1. Download SSH data from CMEMS
2. Download OSTIA SST data from CMEMS
3. Download GlobColour chlorophyll data from CMEMS
4. Generate slippy map tiles for each product
5. Optionally sync to remote server (if configured)

## Generate Tiles for a Specific Date

```python
from seastate import tile

# Generate SSH tiles for a specific date
tile.ssh("2026-01-15", verbose=True)

# Generate SST tiles
tile.sst("2026-01-15", verbose=True)

# Generate chlorophyll tiles
tile.globcolour("2026-01-15", verbose=True)

# Or generate all products at once
tile.all("2026-01-15", verbose=True)
```

## Generate Tiles for Multiple Days

```python
import seastate

# Process the last 7 days
seastate.last_days(days=7)

# Process the last 30 days
seastate.last_days(days=30)
```

## Manual Data Access

You can also access the data directly without generating tiles:

```python
from seastate.data_sources import cmems_ssh, ostia, globcolour

# Open SSH dataset
ds_ssh = cmems_ssh.open_dataset(dtm="2026-01-15")
print(ds_ssh)

# Open SST dataset
ds_sst = ostia.open_dataset(dtm="2026-01-15")
print(ds_sst)

# Open chlorophyll dataset
ds_chl = globcolour.open_dataset(dtm="2026-01-15")
print(ds_chl)
```

## Visualize Ship Track

Create an interactive map of your cruise track:

```python
from shiptrack import ShipTrackMapper

# Create mapper from CSV with position data
mapper = ShipTrackMapper(
    "cruise_positions.csv",
    lat_col="latitude",
    lon_col="longitude"
)

# Save interactive map
mapper.save_map(
    "cruise_track.html",
    zoom_start=8,
    line_color="darkblue",
    show_markers=True,
    marker_interval=10  # Show marker every 10 points
)
```

## Sync to Remote Server

If you've configured remote sync, you can manually trigger synchronization:

```python
from seastate import tile, layer_config

# Sync tiles to remote server
tile.sync()

# Sync layer configuration
layer_config.sync()
```

## Output Structure

Generated tiles follow the slippy map tile structure:

```
tiles/
├── ssh/
│   └── 2026-01-15/
│       ├── 0/
│       │   └── 0/
│       │       └── 0.png
│       ├── 1/
│       │   ├── 0/
│       │   └── 1/
│       └── ...
├── ostia/
│   └── 2026-01-15/
│       └── ...
└── globcolour/
    └── 2026-01-15/
        └── ...
```

## Next Steps

- [Data Sources](../user-guide/data-sources.md) - Learn about available data sources
- [Tile Generation](../user-guide/tile-generation.md) - Advanced tile generation options
- [API Reference](../api/index.md) - Complete API documentation
