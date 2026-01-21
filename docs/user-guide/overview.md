# Overview

Seastate is a Python-based system for processing oceanographic satellite data and generating web-ready map tiles. It integrates with major oceanographic data providers and produces slippy map tiles compatible with Leaflet, OpenLayers, and other web mapping libraries.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                              │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   CMEMS SSH     │   CMEMS OSTIA   │   CMEMS GlobColour          │
│   (Altimetry)   │   (SST)         │   (Ocean Colour)            │
└────────┬────────┴────────┬────────┴────────┬────────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Retrieval Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ cmems_ssh   │  │   ostia     │  │     globcolour          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tile Generation Layer                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   SlippyTileGenerator                        ││
│  │  - Triangulation-based rendering                             ││
│  │  - Parallel processing                                       ││
│  │  - Contour generation                                        ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output / Distribution                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Local Tiles │  │ Remote Sync │  │  Layer Configuration    │  │
│  │   (PNG)     │  │   (rsync)   │  │      (JSON)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Data Sources (`seastate.data_sources`)

Modules for retrieving oceanographic data from remote services:

- **cmems_ssh**: Sea Surface Height from CMEMS altimetry products
- **ostia**: Sea Surface Temperature from OSTIA analysis
- **globcolour**: Chlorophyll-a from GlobColour merged products
- **gebco_bathy**: GEBCO bathymetry data from CEDA
- **olci_L2**: Sentinel-3 OLCI Level-2 swath data from EUMETSAT

### Tile Generation (`seastate.tile`, `seastate.tilers`)

The tile generation system converts gridded data into slippy map tiles:

- **tile.py**: High-level interface for generating tiles by product type
- **rectlinear.py**: Main tile generator using triangulation for smooth rendering
- **utils.py**: Utility functions for contour filtering and processing

### Configuration (`seastate.config`)

Dynaconf-based configuration with support for:

- Multiple configuration file locations
- Environment-based settings
- Secrets management
- Environment variable overrides

### Layer Configuration (`seastate.layer_config`)

Manages the JSON configuration file that defines:

- Available map layers
- Date ranges for each layer
- Tile URL templates
- Layer metadata and attribution

## Data Flow

1. **Retrieval**: Data is downloaded from remote services and cached locally
2. **Processing**: Raw data is processed and prepared for visualization
3. **Tiling**: Data is rendered into 256x256 PNG tiles at multiple zoom levels
4. **Distribution**: Tiles are optionally synced to a remote tile server
5. **Configuration**: Layer metadata is generated and synced

## Coordinate Systems

The system handles two main coordinate systems:

| System | EPSG | Use |
|--------|------|-----|
| WGS84 | 4326 | Input data, geographic coordinates |
| Web Mercator | 3857 | Output tiles, slippy map standard |

The `area_definitions` module provides utilities for creating area definitions in both systems.

## Tile Naming Convention

Tiles follow the standard XYZ/TMS convention:

```
{layer}/{date}/{z}/{x}/{y}.png
```

Where:
- `layer`: Product name (ssh, ostia, globcolour)
- `date`: ISO date (YYYY-MM-DD)
- `z`: Zoom level (0-8 typically)
- `x`: Tile column
- `y`: Tile row
