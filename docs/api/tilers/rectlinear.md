# seastate.tilers.rectlinear

Slippy Tile Generator for Satpy Scenes.

## Overview

This module generates filled contour slippy map tiles from data arrays
with parallel processing. Uses mercantile for robust tile calculations
and Delaunay triangulation for smooth contour rendering.

## Example Usage

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

# Generate tiles
generator.generate_tiles(
    scene_data=data_2d,
    scene_lats=lats_1d,
    scene_lons=lons_1d,
    output_dir="/path/to/tiles",
    zoom_levels=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    num_workers=10,
    cmap="viridis",
    levels=20,
    vmin=-1.0,
    vmax=1.0
)
```

## API Reference

::: seastate.tilers.rectlinear
    options:
      show_root_heading: false
      show_root_full_path: false
