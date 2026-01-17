# _seastate_ a cruise Support Processor

A Python package for generating slippy map tiles from oceanographic satellite data for research cruise support. The processor retrieves near-real-time (NRT) data from Copernicus Marine Service and generates web-ready map tiles for visualization.

## Features

- **Automated Data Retrieval**: Downloads oceanographic data from Copernicus Marine Service
  - Sea Surface Height (SSH) from DUACS L4 altimetry
  - Sea Surface Temperature (SST) from OSTIA
  - Chlorophyll concentration from GlobColour
- **Slippy Map Tile Generation**: Creates standard XYZ tiles compatible with Leaflet, OpenLayers, and other web mapping libraries
- **Parallel Processing**: Multi-core tile generation for fast processing
- **Configurable Regions**: Define custom geographic bounds for your cruise area
- **Remote Sync**: Sync tiles to a remote web server via SSH

## Installation

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

```bash
# Clone the repository
git clone https://github.com/brorfred/seastate.git
cd seastate

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Using pip

```bash
git clone https://github.com/brorfred/seastate.git
cd seastate
pip install -e ".[dev]"
```

### Using pixi

This project also supports [pixi](https://pixi.sh) for conda-based dependency management (recommended for complex geospatial dependencies).

```bash
git clone https://github.com/brorfred/seastate.git
cd seastate
pixi install
```

### Prerequisites

- Python 3.11+
- [Copernicus Marine Service account](https://data.marine.copernicus.eu/) (free registration required)
- Configure your Copernicus credentials:
  ```bash
  copernicusmarine login
  ```

## Quick Start

```python
from processor import tile

# Generate tiles for a specific date
tile.all("2026-01-15")

# Or generate individual products
tile.ssh("2026-01-15")         # Sea Surface Height
tile.sst("2026-01-15")         # Sea Surface Temperature
tile.globcolour("2026-01-15")  # Chlorophyll
```

## Configuration

Configuration is managed using [Dynaconf](https://www.dynaconf.com/). Create a `settings.toml` file in your project directory:

```toml
[DEFAULT]
base_tile_dir = "/path/to/tile/output/"
base_data_dir = "/path/to/data/cache/"
remote_html_dir = "/var/www/html/cruise/"
remote_server = "yourserver"

# Cruise Specific Information
cruise_name = "your_cruise_name"
lat1 = -55        # Southern boundary
lat2 = -10        # Northern boundary
lon1 = -75        # Western boundary
lon2 = -5         # Eastern boundary
zoom_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Directories (use @format for interpolation)
data_dir = "@format {this.base_data_dir}/{this.cruise_name}"
tile_dir = "@format {this.base_tile_dir}/{this.cruise_name}"
remote_tile_dir = "@format {this.remote_html_dir}/tiles/{this.cruise_name}"
```

Settings are loaded from multiple locations (in order of precedence):
1. `/etc/seastate/settings.toml` (system-wide)
2. `~/.config/seastate/settings.toml` (user)
3. `./settings.toml` (project directory)

Environment variables prefixed with `SEASTATE_` will override settings.

## Project Structure

```
seastate/
├── src/processor/
│   ├── __init__.py
│   ├── config.py              # Dynaconf settings loader
│   ├── tile.py                # Main tile generation orchestration
│   ├── layer_config.py        # Web layer configuration management
│   ├── area_definitions.py    # Geographic projection utilities
│   ├── data_sources/          # Data retrieval modules
│   │   ├── cmems_ssh.py       # Sea Surface Height (Copernicus)
│   │   ├── ostia.py           # Sea Surface Temperature (OSTIA)
│   │   └── globcolour.py      # Chlorophyll (GlobColour)
│   ├── tilers/
│   │   └── rectlinear.py      # Slippy tile generator
│   └── readers/
│       └── copernicus_ssh.py  # SatPy file handler for SSH
├── tests/
│   └── test_processor.py      # Test suite
├── settings.toml              # Configuration file
├── pixi.toml                  # Pixi dependencies
└── pyproject.toml             # Python package configuration
```

## API Reference

### tile

Main module for generating tiles.

```python
from processor import tile

# Generate all products for a date
tile.all(dtm, verbose=False)

# Generate specific products
tile.ssh(dtm, verbose=True)
tile.sst(dtm, verbose=True)
tile.globcolour(dtm, verbose=True)

# Sync tiles to remote server
tile.sync()
```

**Parameters:**
- `dtm`: Date string (e.g., "2026-01-15") or datetime object
- `verbose`: Enable verbose output (default: True for individual functions)
- `force`: Re-generate tiles even if they already exist (default: True)

### layer_config

Manages the `layer_config.json` file for web map layer configuration.

```python
from processor import layer_config

# Update date ranges from existing tiles
layer_config.update()

# Generate a sample layer_config.json
layer_config.generate_file("./")

# Sync configuration to remote server
layer_config.sync()
```

### area_definitions

Geographic projection and grid utilities.

```python
from processor import area_definitions

# Get Web Mercator resolution for a zoom level
resolution = area_definitions.zoom_to_resolution_m(zoom=5)

# Create a Web Mercator area definition
area = area_definitions.webmercator(
    lat1=-45, lat2=-10, lon1=-70, lon2=-10,
    zoom=6
)

# Create a NASA rectilinear grid
area = area_definitions.nasa(resolution="4km")
```

### SlippyTileGenerator

Low-level tile generation class.

```python
from processor.tilers.rectlinear import SlippyTileGenerator

generator = SlippyTileGenerator(
    min_lat=-45,
    max_lat=-10,
    min_lon=-70,
    max_lon=-10
)

generator.generate_tiles(
    scene_data=data_array,        # 2D numpy array
    scene_lats=latitudes,         # 1D or 2D array
    scene_lons=longitudes,        # 1D or 2D array
    output_dir="/path/to/tiles",
    zoom_levels=[0, 1, 2, 3, 4, 5],
    cmap="viridis",               # Matplotlib colormap
    levels=20,                    # Number of contour levels
    vmin=10,                      # Minimum value
    vmax=28,                      # Maximum value
    num_workers=10                # Parallel workers
)
```

## Data Sources

### Sea Surface Height (SSH)

- **Source**: Copernicus Marine Service
- **Dataset**: `cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D`
- **Resolution**: 0.125° (~14 km)
- **Variables**: Sea Level Anomaly (SLA), Absolute Dynamic Topography (ADT), Geostrophic currents
- **Reference**: [SEALEVEL_GLO_PHY_L4_NRT_008_046](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_NRT_008_046)

### Sea Surface Temperature (SST)

- **Source**: UK Met Office OSTIA via Copernicus Marine Service
- **Dataset**: `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2`
- **Resolution**: ~5 km
- **Reference**: [SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001](https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001)

### Chlorophyll (GlobColour)

- **Source**: Copernicus Marine Service GlobColour
- **Dataset**: `cmems_obs-oc_glo_bgc-plankton_nrt_l3-multi-4km_P1D`
- **Resolution**: 4 km
- **Reference**: [OCEANCOLOUR_GLO_BGC_L3_NRT_009_101](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L3_NRT_009_101)

## Output Format

Generated tiles follow the standard slippy map tile naming convention:

```
{tile_dir}/{product}/{date}/{z}/{x}/{y}.png
```

Example:
```
/data/tiles/bioreactors1/ssh/2026-01-15/5/12/18.png
```

### Layer Configuration

The `layer_config.json` file configures the web map layer panel:

```json
{
  "base_url": "https://yourserver.com/cruise/tiles/cruisename",
  "layers": [
    {
      "id": "ssh",
      "name": "SSH CMEMS 0.125°",
      "url_template": "{base_url}/ssh/{date}/{z}/{x}/{y}.png",
      "attribution": "Copernicus SSH -0.75–0.75 m",
      "date_range": {
        "start": "2026-01-08",
        "end": "2026-01-14"
      }
    }
  ]
}
```

## Running Tests

```bash
# Using uv/pip
pytest tests/ -v

# Using pixi
pixi run python -m pytest tests/ -v
```

## Batch Processing

Process multiple days:

```python
import pandas as pd
from processor import tile

# Generate tiles for a date range
for date in pd.date_range("2026-01-01", "2026-01-15"):
    print(f"Processing {date.date()}")
    tile.all(date)
```

## Dependencies

Key dependencies managed via pixi:

- **Data handling**: numpy, xarray, pandas, netCDF4
- **Geospatial**: pyresample, pyproj, satpy, mercantile
- **Visualization**: matplotlib, pillow
- **Data access**: copernicusmarine
- **Remote sync**: sysrsync

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Run tests (`pixi run python -m pytest tests/ -v`)
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Open a Pull Request

## License

[Add your license here]

## Acknowledgments

- [Copernicus Marine Service](https://marine.copernicus.eu/) for providing oceanographic data
- [SatPy](https://satpy.readthedocs.io/) for satellite data processing capabilities
- [Pyresample](https://pyresample.readthedocs.io/) for geographic resampling
