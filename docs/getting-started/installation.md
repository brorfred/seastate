# Installation

## Requirements

- Python 3.10+
- GDAL/PROJ libraries
- Access to Copernicus Marine Service (for data downloads)
- Access to EUMETSAT Data Store (for OLCI data)

## Installation with Pixi

The recommended way to install Seastate is using [Pixi](https://pixi.sh):

```bash
# Clone the repository
git clone https://github.com/your-org/seastate.git
cd seastate

# Install with pixi
pixi install
```

## Installation with pip

Alternatively, you can install using pip:

```bash
# Clone the repository
git clone https://github.com/your-org/seastate.git
cd seastate

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Dependencies

The main dependencies include:

| Package | Purpose |
|---------|---------|
| `satpy` | Satellite data processing |
| `pyresample` | Resampling and reprojection |
| `xarray` | NetCDF data handling |
| `copernicusmarine` | CMEMS data access |
| `eumdac` | EUMETSAT data access |
| `mercantile` | Slippy tile calculations |
| `matplotlib` | Visualization and tile rendering |
| `Pillow` | Image processing |
| `folium` | Interactive map visualization |
| `dynaconf` | Configuration management |
| `sysrsync` | Remote synchronization |

## Verifying Installation

After installation, verify everything works:

```python
from seastate import config

# Check loaded configuration files
print("Loaded files:", config.settings.loaded_files)
print("Settings:", dict(config.settings))
```

## Next Steps

- [Configuration](configuration.md) - Set up your configuration files
- [Quick Start](quickstart.md) - Generate your first tiles
