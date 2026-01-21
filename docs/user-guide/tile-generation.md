# Tile Generation

Seastate generates slippy map tiles compatible with web mapping libraries like Leaflet and OpenLayers.

## High-Level Interface

The `seastate.tile` module provides a simple interface for tile generation:

```python
from seastate import tile

# Generate SSH tiles
tile.ssh("2026-01-15", verbose=True)

# Generate SST tiles
tile.sst("2026-01-15", verbose=True)

# Generate chlorophyll tiles
tile.globcolour("2026-01-15", verbose=True)

# Generate all products
tile.all("2026-01-15", verbose=True)
```

## Tile Generator Options

### SSH Tiles

```python
tile.ssh(
    dtm="2026-01-15",  # Date to process
    verbose=True,       # Print progress
    force=True          # Regenerate existing tiles
)
```

Default visualization:
- Colormap: `RdBu` (diverging red-blue)
- Range: -0.75 to 0.75 m
- Good for showing positive/negative anomalies

### SST Tiles (OSTIA)

```python
tile.ostia(
    dtm="2026-01-15",
    verbose=True,
    force=True
)
```

Default visualization:
- Colormap: `viridis`
- Range: 10 to 28 °C
- Temperature converted from Kelvin to Celsius

### Chlorophyll Tiles

```python
tile.globcolour(
    dtm="2026-01-15",
    verbose=True,
    force=True
)
```

Default visualization:
- Colormap: `nipy_spectral`
- Range: log(-4.6) to log(4.6) (~0.01 to 100 mg/m³)
- Log-transformed for wide dynamic range
- 50 contour levels for smooth gradients

## Low-Level Tile Generation

For more control, use the `SlippyTileGenerator` class directly:

```python
from seastate.tilers.rectlinear import SlippyTileGenerator
import numpy as np

# Create generator with geographic bounds
generator = SlippyTileGenerator(
    min_lat=45.0,
    max_lat=65.0,
    min_lon=-10.0,
    max_lon=30.0
)

# Generate tiles from data arrays
generator.generate_tiles(
    scene_data=data_array,      # 2D numpy array
    scene_lats=latitude_array,   # 1D or 2D latitude coordinates
    scene_lons=longitude_array,  # 1D or 2D longitude coordinates
    output_dir="/path/to/tiles",
    zoom_levels=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    num_workers=10,              # Parallel workers
    cmap="viridis",              # Matplotlib colormap
    levels=20,                   # Number of contour levels
    vmin=0.0,                    # Minimum value (None for auto)
    vmax=1.0                     # Maximum value (None for auto)
)
```

## Tile Rendering Methods

### Triangulation-Based (Default)

Uses Delaunay triangulation for smooth filled contours:

```python
from seastate.tilers.rectlinear import SlippyTileGenerator

generator = SlippyTileGenerator(min_lat=45, max_lat=65, min_lon=-10, max_lon=30)
generator.generate_tiles(data, lats, lons, output_dir, zoom_levels)
```

**Advantages**:
- Smooth contours
- Handles irregular data well
- Automatically masks gaps in data

### Satpy-Based Resampling

Uses Satpy's resampling for web Mercator projection:

```python
from seastate.tilers.chatgpt_satpy import satpy_ssh_to_tiles

satpy_ssh_to_tiles(
    scene,                    # Satpy Scene object
    dtm="2026-01-15",
    min_zoom=0,
    max_zoom=8,
    contour_levels=None,      # Auto-calculate
    cmap='viridis',
    add_contour_lines=True
)
```

### Per-Tile Resampling (Most Accurate)

Resamples data for each tile individually for perfect alignment:

```python
from seastate.tilers.ssh_tiles_fast import satpy_ssh_to_tiles_fixed

satpy_ssh_to_tiles_fixed(
    scene,
    dtm="2026-01-15",
    min_zoom=0,
    max_zoom=8,
    workers=8,
    add_contours=True,
    log_qc_path="tile_qc.csv"  # Optional QC log
)
```

## Zoom Levels

Configure zoom levels in your settings:

```toml
zoom_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

| Zoom | Tile Size | Tiles (Global) | Typical Use |
|------|-----------|----------------|-------------|
| 0 | 360° | 1 | World overview |
| 1 | 180° | 4 | Hemisphere |
| 2 | 90° | 16 | Ocean basin |
| 3 | 45° | 64 | Regional |
| 4 | 22.5° | 256 | Sub-regional |
| 5 | 11.25° | 1,024 | Large area |
| 6 | 5.6° | 4,096 | Medium area |
| 7 | 2.8° | 16,384 | Local |
| 8 | 1.4° | 65,536 | Detailed |

## Skip Existing Tiles

By default, existing tiles are not regenerated:

```python
# Skip if tiles exist (default)
tile.ssh("2026-01-15", force=False)

# Force regeneration
tile.ssh("2026-01-15", force=True)
```

Check if tiles exist:

```python
from seastate.tile import tiles_exists

if tiles_exists("ssh", "2026-01-15"):
    print("SSH tiles already exist")
```

## Output Format

Tiles are saved as PNG files with transparency:

- Format: PNG with alpha channel
- Size: 256x256 pixels
- Color: RGBA
- Compression: Default PNG compression

```
{tile_dir}/{product}/{date}/{z}/{x}/{y}.png
```

## Parallel Processing

Tile generation uses parallel processing:

```python
# Set number of workers (rectlinear)
generator.generate_tiles(..., num_workers=10)

# Set number of workers (satpy-based)
satpy_ssh_to_tiles_fixed(..., workers=8)
```

!!! tip "Worker Count"
    A good default is `cpu_count() - 1` to leave one core for system tasks.

## Troubleshooting

### Empty Tiles

Empty (transparent) tiles are generated for areas with no data. This is normal for:
- Tiles outside the data bounds
- Areas with missing/NaN data
- Cloud-covered regions (for optical data)

### Memory Issues

For large datasets or high zoom levels:

- Reduce geographic bounds
- Process one zoom level at a time
- Reduce number of parallel workers
