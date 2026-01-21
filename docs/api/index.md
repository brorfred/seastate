# API Reference

This section contains the complete API reference for Seastate, auto-generated from source code docstrings.

## Package Structure

```
seastate/
├── __init__.py          # Main entry points
├── config.py            # Configuration management
├── tile.py              # Tile generation interface
├── layer_config.py      # Layer configuration
├── area_definitions.py  # Coordinate system utilities
├── cli.py               # Command-line interface
├── data_sources/        # Data retrieval modules
│   ├── cmems_ssh.py
│   ├── ostia.py
│   ├── globcolour.py
│   ├── gebco_bathy.py
│   └── olci_L2.py
├── readers/             # Satpy file handlers
│   └── copernicus_ssh.py
└── tilers/              # Tile rendering engines
    ├── rectlinear.py
    └── utils.py

shiptrack.py             # Ship track visualization (standalone)
```

## Quick Links

### Core Modules

- [seastate](processor.md) - Main entry points (`today`, `yesterday`, `last_days`)
- [seastate.config](config.md) - Configuration management
- [seastate.tile](tile.md) - Tile generation interface
- [seastate.layer_config](layer_config.md) - Layer configuration management
- [seastate.area_definitions](area_definitions.md) - Coordinate system utilities
- [seastate.cli](cli.md) - Command-line interface

### Data Sources

- [seastate.data_sources.cmems_ssh](data_sources/cmems_ssh.md) - SSH data from CMEMS
- [seastate.data_sources.ostia](data_sources/ostia.md) - SST data from OSTIA
- [seastate.data_sources.globcolour](data_sources/globcolour.md) - Chlorophyll from GlobColour
- [seastate.data_sources.gebco_bathy](data_sources/gebco_bathy.md) - Bathymetry from GEBCO
- [seastate.data_sources.olci_L2](data_sources/olci_L2.md) - OLCI L2 from EUMETSAT

### Readers

- [seastate.readers.copernicus_ssh](readers/copernicus_ssh.md) - Satpy file handler for SSH

### Tilers

- [seastate.tilers.rectlinear](tilers/rectlinear.md) - Main tile generator
- [seastate.tilers.utils](tilers/utils.md) - Tile utility functions

### Utilities

- [shiptrack](shiptrack.md) - Ship track visualization

## Common Patterns

### Processing Daily Data

```python
import seastate

# Process yesterday's data
seastate.yesterday()

# Process last week
seastate.last_days(days=7)
```

### Direct Data Access

```python
from seastate.data_sources import cmems_ssh

ds = cmems_ssh.open_dataset(dtm="2026-01-15")
print(ds.sla)
```

### Custom Tile Generation

```python
from seastate.tilers.rectlinear import SlippyTileGenerator

generator = SlippyTileGenerator(
    min_lat=45, max_lat=65,
    min_lon=-10, max_lon=30
)
generator.generate_tiles(data, lats, lons, output_dir, [0, 1, 2, 3])
```

### Configuration Access

```python
from seastate import config

print(config.settings.tile_dir)
print(config.settings.zoom_levels)
```
