# Seastate Documentation

Welcome to the Seastate documentation. This system provides tools for processing oceanographic satellite data and generating slippy map tiles for web visualization.

## Features

- **Multi-source Data Access**: Retrieve data from Copernicus Marine Service (CMEMS) and EUMETSAT
- **Automated Tile Generation**: Generate slippy map tiles for SSH, SST, and chlorophyll products
- **Flexible Configuration**: Dynaconf-based configuration with environment support
- **Remote Synchronization**: Automatic sync to remote tile servers via rsync
- **Ship Track Visualization**: Interactive Folium maps for cruise track display

## Supported Data Products

| Product | Source | Resolution | Description |
|---------|--------|------------|-------------|
| SSH (Sea Surface Height) | CMEMS | 0.125° | Sea level anomaly from altimetry |
| SST (Sea Surface Temperature) | OSTIA/CMEMS | ~5km | Analyzed sea surface temperature |
| Chlorophyll-a | GlobColour/CMEMS | 4km | Merged ocean color product |
| OLCI L2 | EUMETSAT | 300m | Sentinel-3 ocean colour swaths |

## Quick Example

```python
import seastate

# Process tiles for yesterday
seastate.yesterday()

# Process tiles for the last 7 days
seastate.last_days(days=7)
```

## Architecture Overview

```
seastate/
├── config.py           # Configuration management
├── tile.py             # Main tile generation interface
├── layer_config.py     # Layer configuration management
├── area_definitions.py # Coordinate system utilities
├── cli.py              # Command-line interface
├── data_sources/       # Data retrieval modules
│   ├── cmems_ssh.py
│   ├── ostia.py
│   ├── globcolour.py
│   ├── gebco_bathy.py
│   └── olci_L2.py
├── readers/            # Satpy file handlers
│   └── copernicus_ssh.py
└── tilers/             # Tile rendering engines
    ├── rectlinear.py
    └── utils.py
```

## Getting Started

Check out the [Installation Guide](getting-started/installation.md) to get started, or jump to the [Quick Start](getting-started/quickstart.md) for a hands-on introduction.

## API Reference

For detailed API documentation, see the [API Reference](api/index.md).
