# processor.tilers.chatgpt_satpy

Satpy-based tile generation utilities.

## Overview

This module provides functions for generating slippy map tiles from
Satpy scenes, including support for SSH and chlorophyll visualization
with various rendering methods.

## Available Functions

### High-Level

- `satpy_chl_to_tiles`: Generate chlorophyll tiles
- `satpy_ssh_to_tiles`: Generate SSH tiles with matplotlib contours

### Optimized Variants

- `satpy_ssh_to_tiles_2`: Threaded rendering
- `satpy_ssh_to_tiles_3`: PIL-based rendering
- `satpy_ssh_to_tiles_4`: PIL with contour lines

## Example Usage

```python
from processor.tilers.chatgpt_satpy import satpy_ssh_to_tiles
from processor.data_sources import cmems_ssh

# Load scene
scn = cmems_ssh.open_scene(dtm="2026-01-15")

# Generate tiles
satpy_ssh_to_tiles(
    scn,
    dtm="2026-01-15",
    min_zoom=0,
    max_zoom=8,
    cmap='RdBu',
    add_contour_lines=True
)
```

## API Reference

::: processor.tilers.chatgpt_satpy
    options:
      show_root_heading: false
      show_root_full_path: false
      members:
        - enhance_chl
        - reproject
        - satpy_chl_to_tiles
        - satpy_ssh_to_tiles
        - satpy_ssh_to_tiles_2
        - satpy_ssh_to_tiles_3
        - satpy_ssh_to_tiles_4
